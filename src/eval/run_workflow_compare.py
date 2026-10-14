"""Day 15：固定 LangGraph Workflow vs 手写 ReAct，同题对比。

默认取 gold 中 10 道 should_search 题（含拒答），避免算术/闲聊对「必搜」Workflow 不公平。
报告写入 data/eval/workflow_compare_report.json。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.config import load_settings
from src.eval.run_eval import build_agent, case_success, tool_selection_ok
from src.guardrails import estimate_tokens
from src.llm_client import LLMClient
from src.memory.conversation import ConversationMemory
from src.react_agent import extract_citation_ids, looks_like_refuse
from src.workflow.langgraph_qa import FixedCitationWorkflow, WORKFLOW_NOTES

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "data" / "eval" / "gold.json"
REPORT_PATH = ROOT / "data" / "eval" / "workflow_compare_report.json"

# 10 题：事实 / 专有名词 / 多跳 / 拒答各若干
DEFAULT_IDS = [
    "react_1",
    "react_2",
    "rag_1",
    "rag_2",
    "attn_1",
    "wf_1",
    "mcp_1",
    "multi_1",
    "refuse_printer",
    "refuse_author",
]


def _pick_gold(limit: int = 10) -> list[dict[str, Any]]:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    by_id = {item["id"]: item for item in gold}
    selected = [by_id[i] for i in DEFAULT_IDS if i in by_id]
    if len(selected) < limit:
        for item in gold:
            if item.get("should_search") and item["id"] not in {x["id"] for x in selected}:
                selected.append(item)
            if len(selected) >= limit:
                break
    return selected[:limit]


def _error_type(item: dict[str, Any], row: dict[str, Any]) -> str:
    if row["success"]:
        return "ok"
    if item.get("should_refuse") and not row["refused"]:
        return "应拒答却未拒答"
    if item.get("should_search") and not item.get("should_refuse") and not row["has_valid_citation"]:
        return "缺合法引用"
    if row["stopped_reason"] == "llm_error":
        return "llm_error"
    if row["stopped_reason"] == "no_citation":
        return "引用校验耗尽"
    return row["stopped_reason"] or "其他失败"


def _summarize(name: str, gold: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(gold)
    success_n = sum(1 for r in rows if r["success"])
    steps = [r["steps"] for r in rows]
    latencies = [r["latency_ms"] for r in rows]
    tokens = [r["tokens_est"] for r in rows]
    error_counts: dict[str, int] = {}
    for item, row in zip(gold, rows, strict=True):
        et = _error_type(item, row)
        error_counts[et] = error_counts.get(et, 0) + 1
    return {
        "pipeline": name,
        "n": n,
        "success_rate": success_n / max(1, n),
        "avg_steps": sum(steps) / max(1, n),
        "avg_latency_ms": sum(latencies) / max(1, n),
        "avg_tokens_est": sum(tokens) / max(1, n),
        "error_types": error_counts,
    }


def run_workflow_compare(*, limit: int = 10) -> dict[str, Any]:
    gold = _pick_gold(limit)
    settings = load_settings()
    agent, store = build_agent()
    llm = LLMClient(settings)
    workflow = FixedCitationWorkflow(llm, store, mode="hybrid_rerank")

    agent_rows: list[dict[str, Any]] = []
    for item in gold:
        agent.memory = ConversationMemory()
        t0 = time.perf_counter()
        result = agent.run(item["question"])
        latency_ms = (time.perf_counter() - t0) * 1000
        cited = extract_citation_ids(result.answer)
        row = {
            "id": item["id"],
            "type": item.get("type"),
            "question": item["question"],
            "answer": result.answer,
            "steps": result.steps,
            "tools": result.tool_calls,
            "stopped_reason": result.stopped_reason,
            "citation_ids": cited,
            "has_valid_citation": agent.has_valid_citation(result.answer),
            "refused": looks_like_refuse(result.answer),
            "latency_ms": round(latency_ms, 2),
            "tokens_est": estimate_tokens(item["question"], result.answer),
            "tool_selection_ok": False,
            "success": False,
        }
        row["tool_selection_ok"] = tool_selection_ok(item, result.tool_calls)
        row["success"] = case_success(item, row)
        agent_rows.append(row)

    wf_rows: list[dict[str, Any]] = []
    for item in gold:
        t0 = time.perf_counter()
        result = workflow.run(item["question"])
        latency_ms = (time.perf_counter() - t0) * 1000
        cited = extract_citation_ids(result.answer)
        has_valid = any(store.get(cid) is not None for cid in cited)
        row = {
            "id": item["id"],
            "type": item.get("type"),
            "question": item["question"],
            "answer": result.answer,
            "steps": result.steps,
            "tools": result.tool_calls,
            "stopped_reason": result.stopped_reason,
            "citation_ids": cited,
            "has_valid_citation": has_valid,
            "refused": looks_like_refuse(result.answer),
            "latency_ms": round(latency_ms, 2),
            "tokens_est": result.tokens_est or estimate_tokens(item["question"], result.answer),
            "hit_ids": result.hit_ids,
            "tool_selection_ok": False,
            "success": False,
        }
        row["tool_selection_ok"] = tool_selection_ok(item, result.tool_calls)
        row["success"] = case_success(item, row)
        wf_rows.append(row)

    report: dict[str, Any] = {
        "n": len(gold),
        "gold_ids": [g["id"] for g in gold],
        "gold_path": str(GOLD_PATH.relative_to(ROOT)),
        "model": settings.model,
        "selection_note": (
            "仅 should_search 题；Workflow 每题必搜，不适合算术/闲聊对照。"
        ),
        "workflow_notes": WORKFLOW_NOTES.strip(),
        "agentic": _summarize("agentic_react", gold, agent_rows),
        "workflow": _summarize("langgraph_fixed", gold, wf_rows),
        "cases": {"agentic": agent_rows, "workflow": wf_rows},
        "takeaway": (
            "Workflow：步骤固定、延迟更稳，适合「必检索」文献问答；"
            "Agent：可跳过检索/改写/多跳，适合分支不确定任务。"
            "生产常外层 Workflow 控边界，局部节点再用 Agent。"
        ),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def print_compare(report: dict[str, Any]) -> None:
    print(f"同题对比 n={report['n']}  ids={report['gold_ids']}")
    print(f"model={report['model']}")
    print(report["selection_note"])
    for key in ("agentic", "workflow"):
        s = report[key]
        print(
            f"[{s['pipeline']}] success={s['success_rate']:.2f} "
            f"avg_steps={s['avg_steps']:.2f} "
            f"avg_latency_ms={s['avg_latency_ms']:.0f} "
            f"avg_tokens_est={s['avg_tokens_est']:.0f} "
            f"errors={s['error_types']}"
        )
    print("takeaway:", report["takeaway"])
    print(f"报告已写：{REPORT_PATH}")
