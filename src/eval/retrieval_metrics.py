"""检索指标：Recall@k、MRR（Day 11）。

- recall_at_k: |相关 ∩ top-k| / |相关|，看有没有召回全
- mrr: 第一个相关结果排名的倒数，兼顾排序位置
"""

from __future__ import annotations


def recall_at_k(ranked_ids: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    hit = set(ranked_ids[:k]) & relevant
    return len(hit) / len(relevant)


def mrr(ranked_ids: list[str], relevant: set[str]) -> float:
    for i, cid in enumerate(ranked_ids, start=1):
        if cid in relevant:
            return 1.0 / i
    return 0.0
