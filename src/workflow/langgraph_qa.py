"""Day 15：LangGraph 固定「检索 → 生成 → 校验引用」工作流。

控制流在代码里，不由模型选下一步。与 ReactAgent 对照用。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from src.guardrails import estimate_tokens
from src.llm_client import LLMClient, LLMError
from src.rag.store import DocumentStore, SearchMode
from src.react_agent import extract_citation_ids, looks_like_refuse

WORKFLOW_NOTES = """
控制流在代码里，不由模型选择下一步：
1. search
2. generate
3. check citation（失败只重试 generate 一次，仍失败则证据不足）

和 ReactAgent 的差别：
- Workflow 适合步骤稳定、要可预期延迟的链路
- Agent 适合「要不要搜、搜什么、要不要再搜」这种分支不确定的任务
落地时常混合：外层 Workflow，内层个别节点交给 Agent。
"""

_GENERATE_SYSTEM = (
    "你是文献问答助手。只能根据给定检索片段回答。"
    "最终答案必须引用真实 chunk_id，格式 [doc_id#数字]，例如 [react#0]。"
    "若片段不足以回答，明确说不知道，禁止编造引用。"
)
_RETRY_HINT = (
    "系统校验失败：上轮回答缺少可验证引用。"
    "请只使用片段中真实存在的 chunk_id；证据不足请明确说不知道。"
)
_REFUSE_NO_HITS = "本地文献库没有检索到相关内容，我不知道，不能编造。"
_REFUSE_NO_CITATION = "证据不足，无法基于检索片段给出带合法引用的回答。我不知道。"


class WorkflowState(TypedDict, total=False):
    question: str
    hits_text: str
    hit_ids: list[str]
    answer: str
    citation_ok: bool
    retry_used: bool
    stopped_reason: str
    steps: int
    tokens_est: int
    error: str


@dataclass
class WorkflowResult:
    answer: str
    steps: int
    tool_calls: list[str] = field(default_factory=list)
    stopped_reason: str = "final"
    tokens_est: int = 0
    hit_ids: list[str] = field(default_factory=list)
    citation_ok: bool = False
    error: str = ""


class FixedCitationWorkflow:
    """三节点固定链：search → generate → check_citation。"""

    def __init__(
        self,
        llm: LLMClient,
        store: DocumentStore,
        *,
        mode: SearchMode = "hybrid_rerank",
        k: int = 4,
    ) -> None:
        self.llm = llm
        self.store = store
        self.mode = mode
        self.k = k
        self._graph = self._build_graph()

    def _build_graph(self) -> Any:
        graph = StateGraph(WorkflowState)
        graph.add_node("search", self._node_search)
        graph.add_node("generate", self._node_generate)
        graph.add_node("check_citation", self._node_check)
        graph.add_edge(START, "search")
        graph.add_conditional_edges(
            "search",
            self._route_after_search,
            {"generate": "generate", "end": END},
        )
        graph.add_edge("generate", "check_citation")
        graph.add_conditional_edges(
            "check_citation",
            self._route_after_check,
            {"ok": END, "retry": "generate", "fail": END},
        )
        return graph.compile()

    def _node_search(self, state: WorkflowState) -> dict[str, Any]:
        question = state["question"]
        hits = self.store.search(question, k=self.k, mode=self.mode)
        steps = int(state.get("steps") or 0) + 1
        if not hits:
            return {
                "hits_text": "",
                "hit_ids": [],
                "answer": _REFUSE_NO_HITS,
                "citation_ok": True,
                "stopped_reason": "no_hits",
                "steps": steps,
                "tokens_est": estimate_tokens(question, _REFUSE_NO_HITS),
            }
        blocks = "\n\n".join(f"[{h.chunk_id}]\n{h.text}" for h in hits)
        return {
            "hits_text": blocks,
            "hit_ids": [h.chunk_id for h in hits],
            "steps": steps,
            "stopped_reason": "searched",
        }

    def _node_generate(self, state: WorkflowState) -> dict[str, Any]:
        question = state["question"]
        hits_text = state.get("hits_text") or ""
        user_parts = [f"检索片段：\n{hits_text}", f"问题：{question}"]
        if state.get("retry_used") and state.get("stopped_reason") == "citation_retry":
            user_parts.insert(0, _RETRY_HINT)
        messages = [
            {"role": "system", "content": _GENERATE_SYSTEM},
            {"role": "user", "content": "\n\n".join(user_parts)},
        ]
        steps = int(state.get("steps") or 0) + 1
        try:
            response = self.llm.chat(messages, temperature=0.0)
            answer = (response.choices[0].message.content or "").strip()
            err = ""
            stopped = "generated"
        except LLMError as exc:
            answer = f"模型调用失败：{exc}"
            err = str(exc)
            stopped = "llm_error"
        return {
            "answer": answer,
            "steps": steps,
            "stopped_reason": stopped,
            "error": err,
            "tokens_est": estimate_tokens(question, hits_text, answer),
            "citation_ok": False,
        }

    def _node_check(self, state: WorkflowState) -> dict[str, Any]:
        answer = state.get("answer") or ""
        if state.get("stopped_reason") == "llm_error":
            return {"citation_ok": False, "stopped_reason": "llm_error"}

        cite_ids = extract_citation_ids(answer)
        has_valid = any(self.store.get(cid) is not None for cid in cite_ids)
        ok = has_valid or looks_like_refuse(answer)
        if ok:
            return {"citation_ok": True, "stopped_reason": "final"}

        if not state.get("retry_used"):
            return {
                "citation_ok": False,
                "retry_used": True,
                "stopped_reason": "citation_retry",
            }

        return {
            "citation_ok": False,
            "answer": _REFUSE_NO_CITATION,
            "stopped_reason": "no_citation",
            "tokens_est": estimate_tokens(state.get("question") or "", _REFUSE_NO_CITATION),
        }

    @staticmethod
    def _route_after_search(state: WorkflowState) -> Literal["generate", "end"]:
        if state.get("stopped_reason") == "no_hits":
            return "end"
        return "generate"

    @staticmethod
    def _route_after_check(state: WorkflowState) -> Literal["ok", "retry", "fail"]:
        if state.get("citation_ok"):
            return "ok"
        if state.get("stopped_reason") == "citation_retry":
            return "retry"
        return "fail"

    def run(self, question: str) -> WorkflowResult:
        init: WorkflowState = {
            "question": question,
            "hits_text": "",
            "hit_ids": [],
            "answer": "",
            "citation_ok": False,
            "retry_used": False,
            "stopped_reason": "",
            "steps": 0,
            "tokens_est": 0,
            "error": "",
        }
        final = self._graph.invoke(init)
        return WorkflowResult(
            answer=final.get("answer") or "",
            steps=int(final.get("steps") or 0),
            tool_calls=["search_docs"],
            stopped_reason=final.get("stopped_reason") or "final",
            tokens_est=int(final.get("tokens_est") or 0),
            hit_ids=list(final.get("hit_ids") or []),
            citation_ok=bool(final.get("citation_ok")),
            error=final.get("error") or "",
        )


def not_implemented() -> None:
    """兼容旧占位调用；请改用 FixedCitationWorkflow。"""
    raise NotImplementedError("已实现 FixedCitationWorkflow，见 src/workflow/langgraph_qa.py")
