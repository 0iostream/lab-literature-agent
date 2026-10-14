"""手写 ReAct Agent：模型选工具，本地执行，Observation 回灌。

- extract_citation_ids: 从答案抽出 [doc#chunk] 引用 id
- looks_like_refuse: 判断是否像拒答
- AgentResult: 一次 run 的返回（答案、步数、工具、停止原因、trace_id）
- ReactAgent.run: 主循环；含引用门禁、重复调用检测、max_steps 收尾
- ReactAgent.has_valid_citation: 引用 id 是否在文献库真实存在
- ReactAgent._run_tool: 解析 JSON 参数并执行工具，失败变成 ERROR 字符串
- ReactAgent._new_trace / _append_trace: 唯一轨迹文件与写日志
"""

from __future__ import annotations

import json
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.guardrails import append_trace, check_user_input, estimate_tokens
from src.llm_client import LLMClient, LLMError
from src.memory.conversation import ConversationMemory
from src.rag.store import DocumentStore
from src.tools.base import ToolRegistry

SYSTEM_PROMPT = """你是实验室文献问答助手。

规则：
1. 需要事实或论文内容时，先调用 search_docs，必要时再 read_chunk。
2. search_docs 返回 NO_HITS 时，调用 rewrite_query（保留专有名词/数字），再用新查询 search_docs；不要一上来就改写。
3. 最终回答必须引用检索到的 chunk_id，格式 [doc_id#chunk]，例如 [react#0]。
4. 检索不到就明确说不知道，禁止编造文献。
5. 算术用 python_calc，不要心算；寒暄/闲聊直接回答，不要调用 search_docs。
6. 工具失败时根据 Observation 调整，不要重复完全相同的调用超过 2 次。
"""

# 匹配 [react#0] / [workflow_agent#12] 这类引用
_CITATION_RE = re.compile(r"\[([^\]]+?#\d+)\]")
# 避免单独匹配「不确定」：语料/答案讨论「是否需要检索并不确定」时会误伤
_REFUSE_MARKERS = (
    "我不知道",
    "我不确定",
    "检索不到",
    "查不到",
    "没有找到",
    "未找到",
    "找不到",
    "无法基于",
    "无法回答",
    "无法提供",
    "没有相关文献",
    "没有相关内容",
    "没有在本地文献",
    "文献库中没有",
    "不能编造",
    "证据不足",
)
_RETRIEVAL_TOOLS = frozenset({"search_docs", "read_chunk"})
_CITATION_RETRY_PROMPT = (
    "系统校验失败：你的回答缺少可验证的文献引用。"
    "请只使用本次检索中真实存在的 chunk_id，格式必须是 [doc_id#数字]，例如 [react#0]；"
    "若证据不足请明确说不知道，不要编造引用。"
)


def extract_citation_ids(text: str) -> list[str]:
    return _CITATION_RE.findall(text or "")


def looks_like_refuse(text: str) -> bool:
    return any(marker in (text or "") for marker in _REFUSE_MARKERS)


@dataclass
class AgentResult:
    answer: str
    steps: int
    tool_calls: list[str] = field(default_factory=list)
    stopped_reason: str = "final"
    trace_id: str = ""


class ReactAgent:
    """手写 ReAct：Thought/Action 由模型以 tool_call 形式给出，Observation 由本地工具返回。"""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        max_steps: int = 6,
        trace_dir: Path | None = None,
        memory: ConversationMemory | None = None,
        store: DocumentStore | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps
        self.trace_dir = trace_dir or Path("traces")
        self.memory = memory or ConversationMemory()
        self.store = store

    def _new_trace(self) -> tuple[str, Path]:
        """秒级时间戳 + 短 uuid，避免同一秒两次 run 写进同一个文件。"""
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        trace_id = f"{stamp}-{uuid.uuid4().hex[:8]}"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        return trace_id, self.trace_dir / f"{trace_id}.jsonl"

    def _append_trace(self, path: Path, trace_id: str, event: dict[str, Any]) -> None:
        event = {"trace_id": trace_id, **event}
        append_trace(path, event)

    def has_valid_citation(self, answer: str) -> bool:
        """答案里至少有一个能在文献库查到的 [doc_id#chunk]。"""
        if self.store is None:
            return False
        for chunk_id in extract_citation_ids(answer):
            if self.store.get(chunk_id) is not None:
                return True
        return False

    def _needs_citation_gate(self, tool_calls: list[str]) -> bool:
        """本轮用过检索类工具时，最终答案必须合法引用或明确拒答。"""
        return self.store is not None and any(name in _RETRIEVAL_TOOLS for name in tool_calls)

    def _citation_acceptable(self, answer: str) -> bool:
        return self.has_valid_citation(answer) or looks_like_refuse(answer)

    def run(self, user_input: str) -> AgentResult:
        blocked = check_user_input(user_input)
        if blocked:
            return AgentResult(answer=blocked, steps=0, stopped_reason="guardrail")

        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(self.memory.as_messages())
        messages.append({"role": "user", "content": user_input})

        tool_calls: list[str] = []
        call_counter: Counter[str] = Counter()
        citation_retry_used = False
        trace_id, trace_path = self._new_trace()

        try:
            return self._run_loop(
                user_input=user_input,
                messages=messages,
                tool_calls=tool_calls,
                call_counter=call_counter,
                citation_retry_used=citation_retry_used,
                trace_id=trace_id,
                trace_path=trace_path,
            )
        except LLMError as exc:
            answer = f"模型调用失败：{exc}"
            self._append_trace(
                trace_path,
                trace_id,
                {"step": 0, "type": "final", "answer": answer, "stopped_reason": "llm_error"},
            )
            return AgentResult(
                answer=answer,
                steps=0,
                tool_calls=tool_calls,
                stopped_reason="llm_error",
                trace_id=trace_id,
            )

    def _run_loop(
        self,
        user_input: str,
        messages: list[dict[str, Any]],
        tool_calls: list[str],
        call_counter: Counter[str],
        citation_retry_used: bool,
        trace_id: str,
        trace_path: Path,
    ) -> AgentResult:
        for step in range(1, self.max_steps + 1):
            response = self.llm.chat(messages, tools=self.tools.schemas())
            choice = response.choices[0]
            message = choice.message
            messages.append(message)

            if not message.tool_calls:
                answer = (message.content or "").strip() or "（模型没有返回内容）"
                if self._needs_citation_gate(tool_calls) and not self._citation_acceptable(answer):
                    if not citation_retry_used:
                        citation_retry_used = True
                        messages.append({"role": "user", "content": _CITATION_RETRY_PROMPT})
                        self._append_trace(
                            trace_path,
                            trace_id,
                            {
                                "step": step,
                                "type": "citation_retry",
                                "answer": answer,
                                "extracted": extract_citation_ids(answer),
                            },
                        )
                        continue
                    answer = (
                        "检索过程未能形成可验证的文献引用，我不能编造出处。"
                        "关于该问题，我目前无法基于本地文献库给出有依据的答案。"
                    )
                    self.memory.add("user", user_input)
                    self.memory.add("assistant", answer)
                    self._append_trace(
                        trace_path,
                        trace_id,
                        {
                            "step": step,
                            "type": "final",
                            "answer": answer,
                            "tokens_est": estimate_tokens(user_input, answer),
                            "stopped_reason": "no_citation",
                        },
                    )
                    return AgentResult(
                        answer=answer,
                        steps=step,
                        tool_calls=tool_calls,
                        stopped_reason="no_citation",
                        trace_id=trace_id,
                    )

                self.memory.add("user", user_input)
                self.memory.add("assistant", answer)
                self._append_trace(
                    trace_path,
                    trace_id,
                    {
                        "step": step,
                        "type": "final",
                        "answer": answer,
                        "tokens_est": estimate_tokens(user_input, answer),
                        "stopped_reason": "final",
                    },
                )
                return AgentResult(
                    answer=answer,
                    steps=step,
                    tool_calls=tool_calls,
                    stopped_reason="final",
                    trace_id=trace_id,
                )

            for call in message.tool_calls:
                name = call.function.name
                raw_args = call.function.arguments or "{}"
                signature = f"{name}:{raw_args}"
                tool_calls.append(name)
                call_counter[signature] += 1

                if call_counter[signature] > 2:
                    observation = "ERROR: 检测到重复工具调用，请基于已有结果直接给出最终回答。"
                else:
                    observation = self._run_tool(name, raw_args)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": observation,
                    }
                )
                self._append_trace(
                    trace_path,
                    trace_id,
                    {
                        "step": step,
                        "type": "tool",
                        "tool": name,
                        "arguments": raw_args,
                        "observation": observation[:2000],
                    },
                )

        fallback = "已达到最大步数。请根据已有 Observation 给出当前最好的回答，并注明不确定之处。"
        if self._needs_citation_gate(tool_calls):
            fallback += " 若作答，必须带真实存在的 [doc_id#chunk] 引用；否则明确说不知道。"
        messages.append({"role": "user", "content": fallback})
        last = self.llm.chat(messages, tools=None)
        answer = (last.choices[0].message.content or "").strip()
        stopped = "max_steps"
        if self._needs_citation_gate(tool_calls) and not self._citation_acceptable(answer):
            answer = (
                "已达到最大步数，且最终回答缺少可验证的文献引用。"
                "我不能编造出处，因此无法给出有依据的答案。"
            )
            stopped = "no_citation"
        self.memory.add("user", user_input)
        self.memory.add("assistant", answer)
        self._append_trace(
            trace_path,
            trace_id,
            {
                "step": self.max_steps,
                "type": "final",
                "answer": answer,
                "tokens_est": estimate_tokens(user_input, answer),
                "stopped_reason": stopped,
            },
        )
        return AgentResult(
            answer=answer,
            steps=self.max_steps,
            tool_calls=tool_calls,
            stopped_reason=stopped,
            trace_id=trace_id,
        )

    def _run_tool(self, name: str, raw_args: str) -> str:
        try:
            args = json.loads(raw_args) if raw_args else {}
            if not isinstance(args, dict):
                return "ERROR: 工具参数必须是 JSON 对象，请重试。"
            return self.tools.get(name).run(**args)
        except json.JSONDecodeError as exc:
            return f"ERROR: 工具参数不是合法 JSON（{exc}）。请只输出合法 JSON。"
        except Exception as exc:  # 工具失败必须回传，禁止吞掉
            return f"ERROR: 工具 {name} 执行失败：{exc}"
