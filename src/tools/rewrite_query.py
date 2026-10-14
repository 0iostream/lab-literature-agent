"""查询改写工具 rewrite_query。

- RewriteQueryTool.run: 让 LLM 把口语问题改成短检索词（保留专有名词）
search_docs 返回 NO_HITS 后再用，避免一上来就改写。
"""

from __future__ import annotations

from typing import Any

from src.llm_client import LLMClient
from src.tools.base import Tool

_REWRITE_PROMPT = """把下面的用户问题改写成更适合向量/关键词检索的短查询。
必须保留专有名词、缩写、数字和约束条件，不要改偏原意；去掉客套话。
只输出查询本身，不要解释。

问题：{question}
"""


class RewriteQueryTool(Tool):
    name = "rewrite_query"
    description = "把口语化问题改写成短检索词。search_docs 返回 NO_HITS 时再调用。"
    parameters = {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "原始用户问题"},
        },
        "required": ["question"],
    }

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def run(self, **kwargs: Any) -> str:
        question = str(kwargs.get("question", "")).strip()
        response = self.llm.chat(
            [{"role": "user", "content": _REWRITE_PROMPT.format(question=question)}],
            temperature=0.0,
        )
        return (response.choices[0].message.content or question).strip()
