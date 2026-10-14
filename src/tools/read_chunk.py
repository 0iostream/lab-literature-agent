"""按 id 精读片段工具 read_chunk。

- ReadChunkTool.run: DocumentStore.get(chunk_id)；找不到则 ERROR
"""

from __future__ import annotations

from typing import Any

from src.rag.store import DocumentStore
from src.tools.base import Tool


class ReadChunkTool(Tool):
    name = "read_chunk"
    description = "按 chunk_id 读取完整片段。在 search_docs 之后、需要更多上下文时使用。"
    parameters = {
        "type": "object",
        "properties": {
            "chunk_id": {"type": "string", "description": "形如 doc_name#12"},
        },
        "required": ["chunk_id"],
    }

    def __init__(self, store: DocumentStore) -> None:
        self.store = store

    def run(self, **kwargs: Any) -> str:
        chunk_id = str(kwargs.get("chunk_id", "")).strip()
        chunk = self.store.get(chunk_id)
        if chunk is None:
            return f"ERROR: 找不到 chunk {chunk_id}"
        return f"[{chunk.chunk_id}]\n{chunk.text}"
