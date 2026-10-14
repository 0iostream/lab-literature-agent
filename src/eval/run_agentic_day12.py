"""Day12 Agentic 行为检查（会调 LLM API）。

1) 三条「先搜 NO_HITS → rewrite_query → 再搜」case，写入报告
2) 算术 / 闲聊各一题，检查是否误触发 search_docs
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import load_settings
from src.eval.run_eval import build_agent
from src.llm_client import LLMClient
from src.memory.conversation import ConversationMemory
from src.rag.embedder import get_local_embedder
from src.rag.store import DocumentStore
from src.tools.rewrite_query import RewriteQueryTool

ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "data" / "index" / "chunks.json"
DOCS_DIR = ROOT / "data" / "docs"
OUT_PATH = ROOT / "data" / "eval" / "agentic_day12_report.json"

# 故意用「字面难撞上、改写后应能搜到」的口语问法
REWRITE_CASES = [
    {
        "id": "rw_react",
        "question": "那个又想又动手、中间还看工具返回的框架叫啥来着？",
        "expect_chunk_prefix": "react#",
    },
    {
        "id": "rw_rag",
        "question": "把外面文档捞进来塞进提示词再生成答案，这种套路专业名是什么？",
        "expect_chunk_prefix": "rag#",
    },
    {
        "id": "rw_mcp",
        "question": "Agent 和工具之间那种标准垂直接口协议叫什么缩写？",
        "expect_chunk_prefix": "workflow_agent#",
    },
]

NO_RETRIEVAL_CASES = [
    {"id": "calc", "question": "计算 (3+5)*2", "kind": "arithmetic"},
    {"id": "chitchat", "question": "你好，今天怎么样？", "kind": "chitchat"},
]


def _load_store() -> DocumentStore:
    store = DocumentStore()
    if INDEX_PATH.exists():
        store.load(INDEX_PATH)
    else:
        store.ingest_dir(DOCS_DIR)
    if not store.has_embeddings:
        store.build_embeddings(get_local_embedder())
    return store


def _pipeline_rewrite_cases(store: DocumentStore, llm: LLMClient) -> list[dict[str, Any]]:
    rewriter = RewriteQueryTool(llm)
    rows: list[dict[str, Any]] = []
    for case in REWRITE_CASES:
        q = case["question"]
        before = store.search(q, k=3, mode="hybrid_rerank")
        before_ids = [h.chunk_id for h in before]
        # 若 hybrid 已能命中，仍强制看 lexical 是否 NO_HITS，体现「改写价值」
        lex_before = store.search_lexical(q, k=3)
        rewritten = rewriter.run(question=q)
        after = store.search(rewritten, k=3, mode="hybrid_rerank")
        after_ids = [h.chunk_id for h in after]
        prefix = case["expect_chunk_prefix"]
        rows.append(
            {
                "id": case["id"],
                "question": q,
                "lexical_before": [h.chunk_id for h in lex_before],
                "lexical_no_hits": len(lex_before) == 0,
                "hybrid_before": before_ids,
                "rewritten": rewritten,
                "hybrid_after": after_ids,
                "hit_expected_after": any(cid.startswith(prefix) for cid in after_ids),
            }
        )
    return rows


def _agent_no_retrieval_cases() -> list[dict[str, Any]]:
    agent, _ = build_agent()
    rows: list[dict[str, Any]] = []
    for case in NO_RETRIEVAL_CASES:
        agent.memory = ConversationMemory()
        result = agent.run(case["question"])
        used_search = "search_docs" in result.tool_calls
        rows.append(
            {
                "id": case["id"],
                "kind": case["kind"],
                "question": case["question"],
                "tool_calls": result.tool_calls,
                "wrongly_searched": used_search,
                "answer": result.answer,
                "trace_id": result.trace_id,
            }
        )
    return rows


def run_agentic_day12() -> dict[str, Any]:
    settings = load_settings()
    llm = LLMClient(settings)
    store = _load_store()
    rewrite_rows = _pipeline_rewrite_cases(store, llm)
    no_retr = _agent_no_retrieval_cases()
    report = {
        "rewrite_pipeline": rewrite_rows,
        "no_retrieval_guard": no_retr,
        "summary": {
            "rewrite_n": len(rewrite_rows),
            "rewrite_hit_after": sum(1 for r in rewrite_rows if r["hit_expected_after"]),
            "lexical_no_hits_n": sum(1 for r in rewrite_rows if r["lexical_no_hits"]),
            "wrong_search_n": sum(1 for r in no_retr if r["wrongly_searched"]),
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def print_report(report: dict[str, Any]) -> None:
    print("=== rewrite: NO_HITS(lexical) → rewrite → hybrid_rerank ===")
    for row in report["rewrite_pipeline"]:
        print(f"[{row['id']}]")
        print(f"  Q: {row['question']}")
        print(f"  lexical_no_hits={row['lexical_no_hits']} before={row['lexical_before']}")
        print(f"  rewritten: {row['rewritten']}")
        print(f"  after={row['hybrid_after']} hit_expected={row['hit_expected_after']}")
    print("\n=== arithmetic / chitchat 是否误检索 ===")
    for row in report["no_retrieval_guard"]:
        flag = "FAIL 误触发 search_docs" if row["wrongly_searched"] else "OK"
        print(f"[{row['id']}] {flag} tools={row['tool_calls']}")
        print(f"  A: {row['answer'][:120]}")
    print("\nsummary:", json.dumps(report["summary"], ensure_ascii=False))
    print(f"完整报告: {OUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    print_report(run_agentic_day12())
