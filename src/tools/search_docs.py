"""文献检索工具 search_docs。

- SearchDocsTool.run: 默认 hybrid_rerank；0 命中返回 NO_HITS，否则带 [chunk_id] 正文
- 若仍 NO_HITS，应再调 rewrite_query 后重搜
"""

from __future__ import annotations

from typing import Any

from src.rag.store import DocumentStore, SearchMode
from src.tools.base import Tool


class SearchDocsTool(Tool):
    name = "search_docs"
    description = (
        "在本地文献库中检索与问题相关的片段。回答专业事实/论文内容时必须先检索；"
        "纯算术或闲聊不要调用。"
        "若返回 NO_HITS，再用 rewrite_query 改写后重搜。"
        "最终回答需引用 chunk_id。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索查询，尽量具体"},
            "k": {"type": "integer", "description": "返回条数，默认 4"},
        },
        "required": ["query"],
    }

    def __init__(self, store: DocumentStore, mode: SearchMode = "hybrid_rerank") -> None:
        self.store = store
        self.mode = mode

    def run(self, **kwargs: Any) -> str:
        query = str(kwargs.get("query", "")).strip()
        k = int(kwargs.get("k", 4) or 4)
        hits = self.store.search(query, k=k, mode=self.mode)
        if not hits:
            return "NO_HITS。可调用 rewrite_query 改写原问题后再 search_docs。"
        lines = [f"[{hit.chunk_id}] score={hit.score:.3f}\n{hit.text}" for hit in hits]
        return "\n\n".join(lines)

    

