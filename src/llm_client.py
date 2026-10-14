"""调用大模型 Chat Completions（OpenAI 兼容）。

- LLMError: 超时重试耗尽、空响应等不可恢复错误
- LLMClient.chat: 发 messages（可选 tools）；含 timeout、有限重试、空 choices 检查
"""

from __future__ import annotations

import time
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from src.config import Settings

# 可重试：临时网络/超时/限流/部分 5xx。参数错误、鉴权失败不重试。
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """LLM 调用失败且已无法恢复时抛出，供上层受控展示。"""


class LLMClient:
    """OpenAI 兼容 Chat Completions 客户端（DeepSeek / 通义 / OpenAI 均可）。"""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.llm_timeout,
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        attempts = self.settings.llm_max_retries + 1
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self._client.chat.completions.create(**kwargs)
                if not getattr(response, "choices", None):
                    raise LLMError("LLM 返回空 choices，无法继续。")
                return response
            except LLMError:
                raise
            except (APITimeoutError, APIConnectionError, RateLimitError) as exc:
                last_error = exc
            except APIStatusError as exc:
                if exc.status_code in _RETRYABLE_STATUS:
                    last_error = exc
                else:
                    raise LLMError(f"LLM 请求被拒绝（HTTP {exc.status_code}）：{exc}") from exc
            except Exception as exc:  # 未知错误：不盲重试，直接受控失败
                raise LLMError(f"LLM 调用异常：{exc}") from exc

            if attempt + 1 >= attempts:
                break
            # 简单指数退避：0.5s, 1s, 2s...
            time.sleep(0.5 * (2**attempt))

        raise LLMError(f"LLM 请求失败（已重试 {self.settings.llm_max_retries} 次）：{last_error}") from last_error
