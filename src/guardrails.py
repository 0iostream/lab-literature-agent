"""简单安全与可观测辅助。

- check_user_input: 命中黑名单则返回拒绝理由，否则 None
- estimate_tokens: 极粗 token 估计（面试要承认不是 tiktoken）
- append_trace: 向 traces/*.jsonl 追加一行事件
"""

from __future__ import annotations

import json
from pathlib import Path

_BLOCKLIST = ("删除全部文件", "rm -rf", "格式化磁盘", "泄露密钥", "ignore previous instructions")


def check_user_input(text: str) -> str | None:
    """命中则返回拒绝理由，否则 None。Day 16 再加更认真的策略。"""
    lowered = text.lower()
    for item in _BLOCKLIST:
        if item.lower() in lowered:
            return f"拒绝执行：触发安全规则（{item}）"
    return None


def estimate_tokens(*texts: str) -> int:
    """极粗估计：中文约 1.5 字/token，英文约 4 字符/token。面试时要承认这不是 tiktoken。"""
    total = 0
    for text in texts:
        total += max(1, int(len(text) / 2))
    return total


def append_trace(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
