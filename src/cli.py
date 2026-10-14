"""命令行入口。

- cmd_chat: 多轮纯对话，不调工具
- cmd_agent: 启动 ReAct Agent（可带一道题或交互）
- cmd_ingest: 切分 data/docs，写出 chunks 索引（可选本地 bge embedding）
- cmd_search: 词法 / 向量 / 混合检索 top-k（可对比）
- cmd_retrieval_eval: 检索评测（含 hybrid vs hybrid_rerank + 延迟）
- cmd_agentic_day12: rewrite 三连 + 算术/闲聊防误检索
- cmd_eval: Day13 端到端评测（Agentic vs 固定 RAG）
- cmd_serve: Day14 FastAPI（/health、/ask）
- cmd_workflow_compare: Day15 LangGraph 固定链 vs ReAct 同题对比
- cmd_demo: 连续跑 3 个演示题（每题独立 memory）
- main: 解析子命令并分发
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_settings
from src.eval.run_agentic_day12 import print_report as print_agentic_report
from src.eval.run_agentic_day12 import run_agentic_day12
from src.eval.run_eval import build_agent
from src.eval.run_eval import print_summary as print_e2e_summary
from src.eval.run_eval import run_eval
from src.eval.run_retrieval_eval import print_summary, run_retrieval_eval
from src.eval.run_workflow_compare import print_compare, run_workflow_compare
from src.llm_client import LLMClient, LLMError
from src.memory.conversation import ConversationMemory
from src.rag.embedder import get_local_embedder
from src.rag.store import DocumentStore, SearchHit

ROOT = Path(__file__).resolve().parents[1]
DOCS_DIR = ROOT / "data" / "docs"
INDEX_PATH = ROOT / "data" / "index" / "chunks.json"
DEMO_QUESTIONS = [
    "计算 (3+5)*2",
    "ReAct 里三个字母分别代表什么？请引用文献。",
    "实验室打印机的维修电话是多少？",
]


def cmd_chat() -> None:
    settings = load_settings()
    llm = LLMClient(settings)
    history: list[dict[str, str]] = [
        {"role": "system", "content": "你是简洁的中文助手。"},
    ]
    print("多轮对话已启动，输入 /exit 离开。")
    while True:
        user = input("you> ").strip()
        if not user or user == "/exit":
            break
        history.append({"role": "user", "content": user})
        try:
            response = llm.chat(history)
            text = response.choices[0].message.content or ""
        except LLMError as exc:
            print(f"assistant> 模型调用失败：{exc}\n")
            history.pop()
            continue
        history.append({"role": "assistant", "content": text})
        print(f"assistant> {text}\n")


def cmd_agent(question: str | None) -> None:
    agent, _ = build_agent()
    if question:
        result = agent.run(question)
        print(result.answer)
        print(f"[steps={result.steps} tools={result.tool_calls} stop={result.stopped_reason} trace={result.trace_id}]")
        return
    print("Agent 已启动，输入 /exit 离开。")
    while True:
        user = input("you> ").strip()
        if not user or user == "/exit":
            break
        result = agent.run(user)
        print(f"assistant> {result.answer}")
        print(f"[steps={result.steps} tools={result.tool_calls} stop={result.stopped_reason} trace={result.trace_id}]\n")


def _local_embedder():
    # 读 LOCAL_EMBEDDING_MODEL；无 .env 密钥时仍可用默认模型名
    try:
        name = load_settings().local_embedding_model
    except RuntimeError:
        name = "BAAI/bge-small-zh-v1.5"
    return get_local_embedder(name)


def cmd_ingest(with_embeddings: bool) -> None:
    store = DocumentStore()
    n = store.ingest_dir(DOCS_DIR)
    if with_embeddings:
        print("加载本地 bge-small-zh 并编码…（首次会下载模型）")
        emb_n = store.build_embeddings(_local_embedder())
        print(f"已编码 {emb_n} 个向量")
    store.dump(INDEX_PATH)
    print(f"已切分 {n} 个 chunk，写入 {INDEX_PATH}")


def _print_hits(label: str, hits: list[SearchHit]) -> None:
    print(f"--- {label} ---")
    if not hits:
        print("NO_HITS")
        return
    for hit in hits:
        print(f"[{hit.chunk_id}] score={hit.score:.3f}\n{hit.text}\n")


def cmd_search(query: str, k: int, mode: str) -> None:
    store = DocumentStore()
    if INDEX_PATH.exists():
        store.load(INDEX_PATH)
    else:
        store.ingest_dir(DOCS_DIR)

    need_emb = mode in ("vector", "hybrid", "hybrid_rerank", "both", "all")
    emb = _local_embedder() if need_emb else None
    if mode in ("lexical", "both", "all"):
        _print_hits("lexical", store.search_lexical(query, k=k))
    if mode in ("vector", "both", "all"):
        _print_hits("vector", store.search_vector(query, k=k, embedder=emb))
    if mode in ("hybrid", "all"):
        _print_hits("hybrid", store.search_hybrid(query, k=k, embedder=emb))
    if mode in ("hybrid_rerank", "all"):
        _print_hits("hybrid_rerank", store.search_hybrid_rerank(query, k=k, embedder=emb))


def cmd_retrieval_eval() -> None:
    print_summary(run_retrieval_eval())


def cmd_agentic_day12() -> None:
    print_agentic_report(run_agentic_day12())


def cmd_eval(compare_fixed_rag: bool) -> None:
    print_e2e_summary(run_eval(compare_fixed_rag=compare_fixed_rag))


def cmd_serve(host: str, port: int) -> None:
    import uvicorn

    print(f"启动 API：http://{host}:{port}  (GET /health, POST /ask)")
    uvicorn.run("src.api.app:app", host=host, port=port, reload=False)


def cmd_workflow_compare(limit: int) -> None:
    print_compare(run_workflow_compare(limit=limit))


def cmd_demo() -> None:
    agent, _ = build_agent()
    for question in DEMO_QUESTIONS:
        # 每题独立会话：否则第 2、3 题会带上前面的答案/引用
        agent.memory = ConversationMemory()
        print("=" * 60)
        print(f"Q: {question}")
        result = agent.run(question)
        print(f"A: {result.answer}")
        print(f"[steps={result.steps} tools={result.tool_calls} stop={result.stopped_reason} trace={result.trace_id}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="实验室文献问答 Agent")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chat", help="Day 1-2：纯多轮对话")
    agent_p = sub.add_parser("agent", help="带工具的 ReAct Agent")
    agent_p.add_argument("question", nargs="?", default=None)
    ingest_p = sub.add_parser("ingest", help="切分 data/docs 并写出索引")
    ingest_p.add_argument(
        "--with-embeddings",
        action="store_true",
        help="用本地 BAAI/bge-small-zh-v1.5 编码并写入索引",
    )
    search_p = sub.add_parser("search", help="检索 top-k（含 hybrid_rerank）")
    search_p.add_argument("query")
    search_p.add_argument("-k", type=int, default=4)
    search_p.add_argument(
        "--mode",
        choices=("lexical", "vector", "hybrid", "hybrid_rerank", "both", "all"),
        default="lexical",
        help="Day12 可用 hybrid_rerank 或 all",
    )
    sub.add_parser("retrieval-eval", help="检索评测：含 hybrid vs hybrid_rerank + 延迟")
    sub.add_parser("agentic-day12", help="Day12：rewrite 三连 + 防误检索（需 API）")
    eval_p = sub.add_parser("eval", help="Day13：端到端评测（需 API）")
    eval_p.add_argument(
        "--no-fixed-rag",
        action="store_true",
        help="只跑 Agentic，跳过固定 RAG 对照（更快）",
    )
    serve_p = sub.add_parser("serve", help="Day14：启动 FastAPI（/health、/ask）")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    wf_p = sub.add_parser(
        "workflow-compare",
        help="Day15：LangGraph 固定链 vs ReAct 同题对比（需 API）",
    )
    wf_p.add_argument(
        "--limit",
        type=int,
        default=10,
        help="同题条数，默认 10（仅 should_search）",
    )
    sub.add_parser("demo", help="Day 20：一键跑 3 个 case")
    args = parser.parse_args()

    if args.cmd == "chat":
        cmd_chat()
    elif args.cmd == "agent":
        cmd_agent(args.question)
    elif args.cmd == "ingest":
        cmd_ingest(args.with_embeddings)
    elif args.cmd == "search":
        cmd_search(args.query, args.k, args.mode)
    elif args.cmd == "retrieval-eval":
        cmd_retrieval_eval()
    elif args.cmd == "agentic-day12":
        cmd_agentic_day12()
    elif args.cmd == "eval":
        cmd_eval(compare_fixed_rag=not args.no_fixed_rag)
    elif args.cmd == "serve":
        cmd_serve(args.host, args.port)
    elif args.cmd == "workflow-compare":
        cmd_workflow_compare(args.limit)
    elif args.cmd == "demo":
        cmd_demo()
    else:
        parser.error(f"未知命令 {args.cmd}")


if __name__ == "__main__":
    main()
