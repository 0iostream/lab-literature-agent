"""会话记忆：多轮对话历史（可截断成摘要）。

- ConversationMemory.add: 追加一轮 user/assistant；过长则压进 summary
- ConversationMemory.as_messages: 转成可塞进 LLM 的 messages 列表
长期记忆（跨会话）本月只口述设计，不在此文件实现。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConversationMemory:
    """会话滚动窗口；超出 max_turns 的旧轮次压成 summary。"""

    max_turns: int = 8
    summary: str = ""
    turns: list[tuple[str, str]] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.turns.append((role, content))
        overflow = len(self.turns) - self.max_turns
        if overflow > 0:
            old = self.turns[:overflow]
            digest = "；".join(f"{r}: {c[:80]}" for r, c in old)
            self.summary = (self.summary + " " + digest).strip()
            self.turns = self.turns[overflow:]

    def as_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self.summary:
            messages.append(
                {
                    "role": "system",
                    "content": f"此前对话摘要（可能不完整）：{self.summary}",
                }
            )
        for role, content in self.turns:
            messages.append({"role": role, "content": content})
        return messages
