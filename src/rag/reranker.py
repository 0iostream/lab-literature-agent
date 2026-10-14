"""重排：召回后对候选再打分（Day 12）。

- CrossEncoderReranker: BAAI/bge-reranker-base，query+chunk 同时看，更准更慢
- TokenOverlapReranker: 轻量 baseline（Jaccard），不下载模型；仅兜底/对照
- get_reranker: 默认 cross；RERANKER_BACKEND=light 或加载失败时退回 light
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from functools import lru_cache

from src.rag.store import SearchHit

_TOKEN = re.compile(r"[\w\u4e00-\u9fff]+")
DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-base"


class Reranker(ABC):
    name: str

    @abstractmethod
    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        raise NotImplementedError


class TokenOverlapReranker(Reranker):
    """轻量 baseline：query 与 chunk 的 token Jaccard。标注清晰，便于对照 cross-encoder。"""

    name = "token_overlap"

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        q = set(m.group(0).lower() for m in _TOKEN.finditer(query))
        if not hits:
            return []
        scored: list[SearchHit] = []
        for hit in hits:
            t = set(m.group(0).lower() for m in _TOKEN.finditer(hit.text))
            if not q or not t:
                score = 0.0
            else:
                score = len(q & t) / len(q | t)
            scored.append(SearchHit(chunk_id=hit.chunk_id, text=hit.text, score=float(score)))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]


class CrossEncoderReranker(Reranker):
    name = "cross_encoder"

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]:
        if not hits:
            return []
        pairs = [(query, hit.text) for hit in hits]
        scores = self._model.predict(pairs, show_progress_bar=False)
        scored = [
            SearchHit(chunk_id=hit.chunk_id, text=hit.text, score=float(scores[i]))
            for i, hit in enumerate(hits)
        ]
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:top_k]


@lru_cache(maxsize=2)
def get_reranker(backend: str | None = None, model_name: str | None = None) -> Reranker:
    choice = (backend or os.getenv("RERANKER_BACKEND", "cross")).strip().lower()
    if choice in ("light", "token", "overlap", "baseline"):
        return TokenOverlapReranker()
    name = model_name or os.getenv("LOCAL_RERANKER_MODEL", DEFAULT_RERANKER_MODEL).strip()
    try:
        return CrossEncoderReranker(model_name=name or DEFAULT_RERANKER_MODEL)
    except Exception as exc:  # noqa: BLE001 — 下载/权限失败时降级，保证评测可跑
        print(f"[reranker] cross-encoder 加载失败，降级为 token_overlap：{exc}")
        return TokenOverlapReranker()
