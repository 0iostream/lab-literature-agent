"""端到端评测（Day 13）：Agentic RAG vs 固定「先搜再答」RAG。

指标分开算：
- 检索：对 should_search 且有 relevant_chunk_ids 的题，用 hybrid_rerank 算 Recall@k/MRR
- 工具：该搜则搜、不该搜则不搜；expect_tools 必须出现
- 答案：合法引用率（可回答且需检索）、拒答 Precision/Recall/F1、成功率
- 成本：平均步数、延迟、estimate_tokens（粗估，非 tiktoken）

写出 data/eval/last_report.json；失败 case 列表便于写 bad case。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.config import load_settings
from src.eval.retrieval_metrics import mrr, recall_at_k
from src.guardrails import estimate_tokens
from src.llm_client import LLMClient, LLMError
from src.memory.conversation import ConversationMemory
from src.rag.store import DocumentStore
from src.react_agent import ReactAgent, extract_citation_ids, looks_like_refuse
from src.tools.base import ToolRegistry
from src.tools.python_calc import PythonCalcTool
from src.tools.read_chunk import ReadChunkTool
from src.tools.rewrite_query import RewriteQueryTool
from src.tools.search_docs import SearchDocsTool

ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = ROOT / "data" / "docs"
GOLD_PATH = ROOT / "data" / "eval" / "gold.json"
REPORT_PATH = ROOT / "data" / "eval" / "last_report.json"
RETRIEVAL_KS = (1, 3, 5)
_RETRIEVAL_TOOLS = frozenset({"search_docs", "read_chunk"})


def build_agent() -> tuple[ReactAgent, DocumentStore]:
    settings = load_settings()
    llm = LLMClient(settings)
    store = DocumentStore()
    index_path = ROOT / "data" / "index" / "chunks.json"
    if index_path.exists():
        store.load(index_path)
    elif DOCS_DIR.exists():
        store.ingest_dir(DOCS_DIR)
    tools = ToolRegistry(
        [
            PythonCalcTool(),
            SearchDocsTool(store, mode="hybrid_rerank"),
            ReadChunkTool(store),
            RewriteQueryTool(llm),
        ]
    )
    agent = ReactAgent(
        llm=llm,
        tools=tools,
        max_steps=settings.max_steps,
        trace_dir=ROOT / "traces",
        store=store,
    )
    return agent, store


def tool_selection_ok(item: dict[str, Any], tool_calls: list[str]) -> bool:
    """该调用时有没有调用、不该调用时有没有乱调用。"""
    used = set(tool_calls)
    expect = set(item.get("expect_tools") or [])
    should_search = bool(item.get("should_search"))

    if should_search and not (used & _RETRIEVAL_TOOLS):
        return False
    if not should_search and (used & _RETRIEVAL_TOOLS):
        return False
    # rewrite_query 允许作为检索辅助，不算进强制集合
    required = expect - {"rewrite_query"}
    if required and not required.issubset(used):
        return False
    return True


def case_success(item: dict[str, Any], row: dict[str, Any]) -> bool:
    """任务级成功：拒答题要拒；算术要调 calc；闲聊不检索；事实题要合法引用。"""
    if item.get("should_refuse"):
        return bool(row["refused"])
    if item.get("type") == "calc":
        return "python_calc" in row["tools"] and not row["refused"]
    if item.get("type") == "chitchat":
        return not (set(row["tools"]) & _RETRIEVAL_TOOLS)
    if item.get("should_search"):
        return bool(row["has_valid_citation"])
    return True


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * p))))
    return ordered[idx]


def _refuse_f1(gold: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, float]:
    tp = fp = fn = 0
    for item, row in zip(gold, rows, strict=True):
        gold_pos = bool(item.get("should_refuse"))
        pred_pos = bool(row["refused"])
        if gold_pos and pred_pos:
            tp += 1
        elif not gold_pos and pred_pos:
            fp += 1
        elif gold_pos and not pred_pos:
            fn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _retrieval_metrics(store: DocumentStore, gold: list[dict[str, Any]]) -> dict[str, Any]:
    """与 Agent 答案解耦：只用 query + 金标 chunk 评检索器。"""
    subset = [
        item
        for item in gold
        if item.get("should_search") and item.get("relevant_chunk_ids")
    ]
    if not subset:
        return {"n": 0}
    recalls = {k: [] for k in RETRIEVAL_KS}
    mrrs: list[float] = []
    for item in subset:
        relevant = set(item["relevant_chunk_ids"])
        hits = store.search(item["question"], k=max(RETRIEVAL_KS), mode="hybrid_rerank")
        ranked = [h.chunk_id for h in hits]
        for k in RETRIEVAL_KS:
            recalls[k].append(recall_at_k(ranked, relevant, k))
        mrrs.append(mrr(ranked, relevant))
    n = len(subset)
    return {
        "n": n,
        "mode": "hybrid_rerank",
        **{f"recall@{k}": sum(recalls[k]) / n for k in RETRIEVAL_KS},
        "mrr": sum(mrrs) / n,
    }


def run_fixed_rag(store: DocumentStore, llm: LLMClient, question: str) -> dict[str, Any]:
    """固定流水线：先 hybrid_rerank 再一次性生成（对照 Agentic）。"""
    t0 = time.perf_counter()
    hits = store.search(question, k=4, mode="hybrid_rerank")
    if not hits:
        answer = "本地文献库没有检索到相关内容，我不知道，不能编造。"
        latency_ms = (time.perf_counter() - t0) * 1000
        return {
            "answer": answer,
            "tools": ["search_docs"],
            "steps": 1,
            "stopped_reason": "fixed_rag_no_hits",
            "latency_ms": latency_ms,
            "tokens_est": estimate_tokens(question, answer),
            "citation_ids": [],
            "has_valid_citation": False,
            "refused": True,
        }

    blocks = "\n\n".join(f"[{h.chunk_id}]\n{h.text}" for h in hits)
    prompt = (
        "你是文献问答助手。只能根据下列检索片段回答。"
        "最终答案必须引用真实 chunk_id，格式 [doc_id#数字]。"
        "若片段不足以回答，明确说不知道。\n\n"
        f"检索片段：\n{blocks}\n\n问题：{question}"
    )
    try:
        response = llm.chat([{"role": "user", "content": prompt}], temperature=0.0)
        answer = (response.choices[0].message.content or "").strip()
        stopped = "fixed_rag"
    except LLMError as exc:
        answer = f"模型调用失败：{exc}"
        stopped = "llm_error"
    latency_ms = (time.perf_counter() - t0) * 1000
    cite_ids = extract_citation_ids(answer)
    has_valid = any(store.get(cid) is not None for cid in cite_ids)
    return {
        "answer": answer,
        "tools": ["search_docs"],
        "steps": 1,
        "stopped_reason": stopped,
        "latency_ms": latency_ms,
        "tokens_est": estimate_tokens(question, blocks, answer),
        "citation_ids": cite_ids,
        "has_valid_citation": has_valid,
        "refused": looks_like_refuse(answer),
    }


def _summarize_pipeline(
    name: str,
    gold: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    n = len(gold)
    answerable = [i for i, x in enumerate(gold) if not x.get("should_refuse") and x.get("should_search")]
    cited_ok = sum(1 for i in answerable if rows[i]["has_valid_citation"])
    tool_ok_n = sum(1 for item, row in zip(gold, rows, strict=True) if row["tool_selection_ok"])
    success_n = sum(1 for item, row in zip(gold, rows, strict=True) if row["success"])
    latencies = [row["latency_ms"] for row in rows]
    steps = [row["steps"] for row in rows]
    tokens = [row["tokens_est"] for row in rows]
    failures = [
        {
            "id": item["id"],
            "type": item.get("type"),
            "question": item["question"],
            "reason": _failure_reason(item, row),
            "tools": row["tools"],
            "stopped_reason": row["stopped_reason"],
            "answer_preview": (row["answer"] or "")[:160],
        }
        for item, row in zip(gold, rows, strict=True)
        if not row["success"]
    ]
    return {
        "pipeline": name,
        "n": n,
        "success_rate": success_n / max(1, n),
        "tool_selection_accuracy": tool_ok_n / max(1, n),
        "citation_rate_on_answerable_search": cited_ok / max(1, len(answerable)),
        "refuse": _refuse_f1(gold, rows),
        "avg_steps": sum(steps) / max(1, n),
        "avg_latency_ms": sum(latencies) / max(1, n),
        "p95_latency_ms": _percentile(latencies, 0.95),
        "avg_tokens_est": sum(tokens) / max(1, n),
        "fail_n": len(failures),
        "failures": failures,
    }


def _failure_reason(item: dict[str, Any], row: dict[str, Any]) -> str:
    if item.get("should_refuse") and not row["refused"]:
        return "应拒答却未拒答"
    if item.get("type") == "calc" and "python_calc" not in row["tools"]:
        return "算术未调用 python_calc"
    if item.get("type") == "chitchat" and (set(row["tools"]) & _RETRIEVAL_TOOLS):
        return "闲聊误触发检索"
    if item.get("should_search") and not item.get("should_refuse") and not row["has_valid_citation"]:
        return "可回答题缺少合法引用"
    if not row["tool_selection_ok"]:
        return "工具选择不符合标注"
    return "未达成功标准"


def run_eval(*, compare_fixed_rag: bool = True) -> dict[str, Any]:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    agent, store = build_agent()
    settings = load_settings()
    llm = LLMClient(settings)

    agent_rows: list[dict[str, Any]] = []
    for item in gold:
        agent.memory = ConversationMemory()
        t0 = time.perf_counter()
        result = agent.run(item["question"])
        latency_ms = (time.perf_counter() - t0) * 1000
        cited_ids = extract_citation_ids(result.answer)
        has_valid_cite = agent.has_valid_citation(result.answer)
        refused = looks_like_refuse(result.answer)
        row = {
            "id": item["id"],
            "type": item.get("type"),
            "question": item["question"],
            "answer": result.answer,
            "steps": result.steps,
            "tools": result.tool_calls,
            "stopped_reason": result.stopped_reason,
            "trace_id": result.trace_id,
            "citation_ids": cited_ids,
            "has_valid_citation": has_valid_cite,
            "should_refuse": bool(item.get("should_refuse")),
            "should_search": bool(item.get("should_search")),
            "refused": refused,
            "latency_ms": round(latency_ms, 2),
            "tokens_est": estimate_tokens(item["question"], result.answer),
            "tool_selection_ok": False,
            "success": False,
        }
        row["tool_selection_ok"] = tool_selection_ok(item, result.tool_calls)
        row["success"] = case_success(item, row)
        agent_rows.append(row)

    report: dict[str, Any] = {
        "n": len(gold),
        "gold_path": str(GOLD_PATH.relative_to(ROOT)),
        "model": settings.model,
        "retrieval": _retrieval_metrics(store, gold),
        "agentic": _summarize_pipeline("agentic", gold, agent_rows),
        "cases": {"agentic": agent_rows},
    }

    if compare_fixed_rag:
        fixed_rows: list[dict[str, Any]] = []
        for item in gold:
            # 固定 RAG 对算术/闲聊仍会先搜——对照基线的已知偏差，用来衬托 Agentic
            raw = run_fixed_rag(store, llm, item["question"])
            row = {
                "id": item["id"],
                "type": item.get("type"),
                "question": item["question"],
                **raw,
                "should_refuse": bool(item.get("should_refuse")),
                "should_search": bool(item.get("should_search")),
                "trace_id": "",
                "tool_selection_ok": False,
                "success": False,
            }
            row["tool_selection_ok"] = tool_selection_ok(item, raw["tools"])
            row["success"] = case_success(item, row)
            fixed_rows.append(row)
        report["fixed_rag"] = _summarize_pipeline("fixed_rag", gold, fixed_rows)
        report["cases"]["fixed_rag"] = fixed_rows
        report["comparison"] = {
            "agentic_success_rate": report["agentic"]["success_rate"],
            "fixed_rag_success_rate": report["fixed_rag"]["success_rate"],
            "agentic_tool_acc": report["agentic"]["tool_selection_accuracy"],
            "fixed_rag_tool_acc": report["fixed_rag"]["tool_selection_accuracy"],
            "agentic_avg_latency_ms": report["agentic"]["avg_latency_ms"],
            "fixed_rag_avg_latency_ms": report["fixed_rag"]["avg_latency_ms"],
            "note": "同题、同库、同检索器；固定 RAG 始终先搜，故 calc/chitchat 工具准确率会明显更低。",
        }

    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def print_summary(report: dict[str, Any]) -> None:
    print(f"n={report['n']}  model={report.get('model')}")
    r = report.get("retrieval") or {}
    if r.get("n"):
        print(
            f"retrieval(n={r['n']} {r.get('mode')}): "
            f"R@1={r['recall@1']:.3f} R@3={r['recall@3']:.3f} MRR={r['mrr']:.3f}"
        )
    for key in ("agentic", "fixed_rag"):
        if key not in report:
            continue
        s = report[key]
        refuse = s["refuse"]
        print(
            f"\n[{s['pipeline']}] success={s['success_rate']:.3f} "
            f"tool_acc={s['tool_selection_accuracy']:.3f} "
            f"cite={s['citation_rate_on_answerable_search']:.3f} "
            f"refuse_f1={refuse['f1']:.3f} "
            f"steps={s['avg_steps']:.2f} "
            f"avg_ms={s['avg_latency_ms']:.0f} p95_ms={s['p95_latency_ms']:.0f} "
            f"tok_est={s['avg_tokens_est']:.0f} fail={s['fail_n']}"
        )
        for fail in s["failures"][:8]:
            print(f"  - {fail['id']}: {fail['reason']}")
        if s["fail_n"] > 8:
            print(f"  ... 另有 {s['fail_n'] - 8} 条，见报告")
    if "comparison" in report:
        c = report["comparison"]
        print("\nA/B:", json.dumps(c, ensure_ascii=False))
    print(f"\n完整报告: {REPORT_PATH.relative_to(ROOT)}")
    print("人工抽查：从 cases.agentic 抽 ≥10 条，核对答案是否被引用片段支持（引用率≠忠实度）。")


if __name__ == "__main__":
    print_summary(run_eval(compare_fixed_rag=True))
