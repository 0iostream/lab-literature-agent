"""文本切分。

- Chunk: 一个片段的数据结构（chunk_id / doc_id / text）
- split_text: 按字符切分，保留 overlap，避免答案刚好落在边界上被切断
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def split_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """按字符切分，保留 overlap，避免句子被拦腰截断得太狠。"""
    text = text.strip()
    if not text:
        return []
    if chunk_size <= overlap:
        raise ValueError("chunk_size 必须大于 overlap")
    pieces: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        pieces.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - overlap
    return [p for p in pieces if p]
