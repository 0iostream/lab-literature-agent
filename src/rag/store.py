"""本地文献库：切分入库 + 词法/向量/混合检索。

- SearchHit: 一条检索结果（chunk_id、正文、分数）
- _tokens: 中英简单分词
- DocumentStore.ingest_dir / ingest_file: 读 md，切成 chunk 放入内存
- DocumentStore.get: 按 chunk_id 取片段（引用校验、read_chunk 用）
- DocumentStore.search: lexical / vector / hybrid / hybrid_rerank
- DocumentStore.build_embeddings: 用本地 bge 为全部 chunk 编码
- DocumentStore.search_hybrid: 词法+向量候选，RRF 融合（不直接加原始分）
- DocumentStore.search_hybrid_rerank: hybrid 召回 top-N 后再 cross-encoder 精排
- DocumentStore.dump / load: 读写 JSON 索引（可含 embedding）
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from src.rag.chunker import Chunk, split_text
from src.rag.embedder import LocalEmbedder

_TOKEN = re.compile(r"[\w\u4e00-\u9fff]+")
SearchMode = Literal["lexical", "vector", "hybrid", "hybrid_rerank"]
RRF_K = 60
DEFAULT_RERANK_FETCH_N = 12


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    text: str
    score: float


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in _TOKEN.finditer(text)]


class DocumentStore:
    """内存文献库。词法 baseline 始终可用；建好 embedding 后可向量检索。"""

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._embeddings: dict[str, np.ndarray] = {}

    @property
    def has_embeddings(self) -> bool:
        return bool(self._embeddings) and len(self._embeddings) == len(self._chunks)

    def ingest_dir(self, docs_dir: Path, chunk_size: int = 400, overlap: int = 80) -> int:
        count = 0
        for path in sorted(docs_dir.glob("*.md")):
            count += self.ingest_file(path, chunk_size=chunk_size, overlap=overlap)
        return count

    def ingest_file(self, path: Path, chunk_size: int = 400, overlap: int = 80) -> int:
        doc_id = path.stem
        pieces = split_text(path.read_text(encoding="utf-8"), chunk_size, overlap)
        for i, text in enumerate(pieces):
            chunk_id = f"{doc_id}#{i}"
            self._chunks[chunk_id] = Chunk(chunk_id=chunk_id, doc_id=doc_id, text=text)
        # 文本变了，旧向量作废
        self._embeddings.clear()
        return len(pieces)

    def get(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)

    def build_embeddings(self, embedder: LocalEmbedder) -> int:
        if not self._chunks:
            return 0
        ids = list(self._chunks.keys())
        texts = [self._chunks[i].text for i in ids]
        matrix = embedder.encode(texts)
        self._embeddings = {cid: matrix[i] for i, cid in enumerate(ids)}
        return len(self._embeddings)

    def search(
        self,
        query: str,
        k: int = 4,
        mode: SearchMode = "lexical",
        embedder: LocalEmbedder | None = None,
        reranker=None,
        fetch_n: int = DEFAULT_RERANK_FETCH_N,
    ) -> list[SearchHit]:
        if mode == "vector":
            return self.search_vector(query, k=k, embedder=embedder)
        if mode == "hybrid":
            return self.search_hybrid(query, k=k, embedder=embedder)
        if mode == "hybrid_rerank":
            return self.search_hybrid_rerank(
                query, k=k, embedder=embedder, reranker=reranker, fetch_n=fetch_n
            )
        return self.search_lexical(query, k=k)

    def search_lexical(self, query: str, k: int = 4) -> list[SearchHit]:
        q = set(_tokens(query))
        if not q:
            return []
        scored: list[SearchHit] = []
        for chunk in self._chunks.values():
            tokens = _tokens(chunk.text)
            if not tokens:
                continue
            overlap = len(q & set(tokens))
            if overlap == 0:
                continue
            score = overlap / math.sqrt(len(tokens))
            scored.append(SearchHit(chunk_id=chunk.chunk_id, text=chunk.text, score=score))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    def search_vector(self, query: str, k: int = 4, embedder: LocalEmbedder | None = None) -> list[SearchHit]:
        if not self._chunks:
            return []
        if embedder is None:
            from src.rag.embedder import get_local_embedder

            embedder = get_local_embedder()
        if not self.has_embeddings:
            self.build_embeddings(embedder)
        q = embedder.encode([query])[0]
        scored: list[SearchHit] = []
        for chunk_id, vec in self._embeddings.items():
            # 已归一化 → 点积即余弦
            score = float(np.dot(q, vec))
            chunk = self._chunks[chunk_id]
            scored.append(SearchHit(chunk_id=chunk.chunk_id, text=chunk.text, score=score))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    def search_hybrid(
        self,
        query: str,
        k: int = 4,
        embedder: LocalEmbedder | None = None,
        fetch_k: int | None = None,
        rrf_k: int = RRF_K,
    ) -> list[SearchHit]:
        """词法与向量各取候选，用 RRF 按排名融合（分数量纲不同，不能直接相加）。"""
        pool = fetch_k if fetch_k is not None else max(k * 3, len(self._chunks))
        lex = self.search_lexical(query, k=pool)
        vec = self.search_vector(query, k=pool, embedder=embedder)
        fused: dict[str, float] = {}
        for rank, hit in enumerate(lex, start=1):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        for rank, hit in enumerate(vec, start=1):
            fused[hit.chunk_id] = fused.get(hit.chunk_id, 0.0) + 1.0 / (rrf_k + rank)
        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
        return [
            SearchHit(chunk_id=cid, text=self._chunks[cid].text, score=score)
            for cid, score in ordered
            if cid in self._chunks
        ]

    def search_hybrid_rerank(
        self,
        query: str,
        k: int = 4,
        embedder: LocalEmbedder | None = None,
        reranker=None,
        fetch_n: int = DEFAULT_RERANK_FETCH_N,
    ) -> list[SearchHit]:
        """hybrid 召回 top-N，再用 reranker 精排到 top-k。"""
        if reranker is None:
            from src.rag.reranker import get_reranker

            reranker = get_reranker()
        candidates = self.search_hybrid(query, k=max(fetch_n, k), embedder=embedder)
        return reranker.rerank(query, candidates, top_k=k)

    def dump(self, path: Path) -> None:
        payload = []
        for c in self._chunks.values():
            row: dict = {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text}
            emb = self._embeddings.get(c.chunk_id)
            if emb is not None:
                row["embedding"] = emb.tolist()
            payload.append(row)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, path: Path) -> int:
        raw = json.loads(path.read_text(encoding="utf-8"))
        self._chunks.clear()
        self._embeddings.clear()
        for row in raw:
            chunk = Chunk(chunk_id=row["chunk_id"], doc_id=row["doc_id"], text=row["text"])
            self._chunks[chunk.chunk_id] = chunk
            if "embedding" in row and row["embedding"] is not None:
                self._embeddings[chunk.chunk_id] = np.asarray(row["embedding"], dtype=np.float32)
        return len(self._chunks)
