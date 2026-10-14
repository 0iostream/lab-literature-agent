"""检索评测：同一批金标上对比 lexical / vector / hybrid / hybrid_rerank。

- 读 data/eval/retrieval_gold.json
- 算 Recall@1/3/5、MRR，以及每路平均延迟（ms）
- 写出 data/eval/retrieval_report.json（含失败查询，不按题手改答案）
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.eval.retrieval_metrics import mrr, recall_at_k
from src.rag.embedder import get_local_embedder
from src.rag.reranker import get_reranker
from src.rag.store import DocumentStore, SearchMode

ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = ROOT / "data" / "index" / "chunks.json"
DOCS_DIR = ROOT / "data" / "docs"
GOLD_PATH = ROOT / "data" / "eval" / "retrieval_gold.json"
REPORT_PATH = ROOT / "data" / "eval" / "retrieval_report.json"
MODES: tuple[SearchMode, ...] = ("lexical", "vector", "hybrid", "hybrid_rerank")
KS = (1, 3, 5)


def _load_store() -> DocumentStore:
    store = DocumentStore()
    if INDEX_PATH.exists():
        store.load(INDEX_PATH)
    else:
        store.ingest_dir(DOCS_DIR)
    return store


def _eval_mode(
    store: DocumentStore,
    gold: list[dict[str, Any]],
    mode: SearchMode,
    embedder,
    reranker,
) -> dict[str, Any]:
    recalls = {k: [] for k in KS}
    mrrs: list[float] = []
    latencies_ms: list[float] = []
    failures: list[dict[str, Any]] = []
    per_case: list[dict[str, Any]] = []

    for item in gold:
        query = item["query"]
        relevant = set(item["relevant_chunk_ids"])
        t0 = time.perf_counter()
        hits = store.search(
            query,
            k=max(KS),
            mode=mode,
            embedder=embedder,
            reranker=reranker,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies_ms.append(elapsed_ms)
        ranked = [h.chunk_id for h in hits]
        case_metrics = {f"recall@{k}": recall_at_k(ranked, relevant, k) for k in KS}
        case_mrr = mrr(ranked, relevant)
        for k in KS:
            recalls[k].append(case_metrics[f"recall@{k}"])
        mrrs.append(case_mrr)
        row = {
            "id": item["id"],
            "type": item.get("type"),
            "query": query,
            "relevant_chunk_ids": sorted(relevant),
            "ranked": ranked,
            "mrr": case_mrr,
            "latency_ms": round(elapsed_ms, 2),
            **case_metrics,
        }
        per_case.append(row)
        if case_metrics["recall@3"] == 0.0:
            failures.append(
                {
                    "id": item["id"],
                    "type": item.get("type"),
                    "query": query,
                    "relevant_chunk_ids": sorted(relevant),
                    "ranked": ranked,
                }
            )

    n = max(1, len(gold))
    summary = {
        "n": len(gold),
        "mode": mode,
        **{f"recall@{k}": sum(recalls[k]) / n for k in KS},
        "mrr": sum(mrrs) / n,
        "avg_latency_ms": sum(latencies_ms) / n,
        "fail_recall@3": len(failures),
        "reranker": getattr(reranker, "name", None) if mode == "hybrid_rerank" else None,
    }
    return {"summary": summary, "failures": failures, "cases": per_case}


def run_retrieval_eval() -> dict[str, Any]:
    gold = json.loads(GOLD_PATH.read_text(encoding="utf-8"))
    store = _load_store()
    embedder = get_local_embedder()
    if not store.has_embeddings:
        store.build_embeddings(embedder)
    reranker = get_reranker()

    by_mode: dict[str, Any] = {}
    for mode in MODES:
        by_mode[mode] = _eval_mode(store, gold, mode, embedder, reranker)

    by_type: dict[str, dict[str, dict[str, float]]] = {}
    types = sorted({item.get("type", "unknown") for item in gold})
    for qtype in types:
        subset = [item for item in gold if item.get("type") == qtype]
        by_type[qtype] = {}
        for mode in MODES:
            cases = [c for c in by_mode[mode]["cases"] if c.get("type") == qtype]
            n = max(1, len(cases))
            by_type[qtype][mode] = {
                "n": len(subset),
                **{f"recall@{k}": sum(c[f"recall@{k}"] for c in cases) / n for k in KS},
                "mrr": sum(c["mrr"] for c in cases) / n,
                "avg_latency_ms": sum(c["latency_ms"] for c in cases) / n,
            }

    report = {
        "gold_path": str(GOLD_PATH.relative_to(ROOT)),
        "n_queries": len(gold),
        "reranker": getattr(reranker, "name", None),
        "modes": {mode: by_mode[mode]["summary"] for mode in MODES},
        "by_type": by_type,
        "failures": {mode: by_mode[mode]["failures"] for mode in MODES},
        "cases": {mode: by_mode[mode]["cases"] for mode in MODES},
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def print_summary(report: dict[str, Any]) -> None:
    print(f"n_queries={report['n_queries']}  reranker={report.get('reranker')}")
    print(f"{'mode':<14} {'R@1':>6} {'R@3':>6} {'R@5':>6} {'MRR':>6} {'ms':>8} {'fail@3':>7}")
    for mode, s in report["modes"].items():
        print(
            f"{mode:<14} {s['recall@1']:6.3f} {s['recall@3']:6.3f} "
            f"{s['recall@5']:6.3f} {s['mrr']:6.3f} {s['avg_latency_ms']:8.1f} "
            f"{s['fail_recall@3']:7d}"
        )
    print("\n重点对比 hybrid vs hybrid_rerank:")
    for mode in ("hybrid", "hybrid_rerank"):
        s = report["modes"][mode]
        print(
            f"  {mode}: R@1={s['recall@1']:.3f} R@3={s['recall@3']:.3f} "
            f"MRR={s['mrr']:.3f} avg_ms={s['avg_latency_ms']:.1f}"
        )
    print(f"\n完整报告: {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    print_summary(run_retrieval_eval())
