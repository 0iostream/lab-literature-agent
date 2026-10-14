"""运行配置。所有密钥只从环境变量读取。

- _optional: 读可选环境变量，空则 None
- Settings: 冻结的配置数据类（API、模型、max_steps、timeout 等）
- load_settings: 从 .env / 环境变量加载并校验 Settings
- Settings.has_embedding: 是否配置了 embedding API（向量检索用）
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _optional(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    model: str
    embedding_api_key: str | None
    embedding_base_url: str | None
    embedding_model: str | None
    local_embedding_model: str
    max_steps: int
    llm_timeout: float
    llm_max_retries: int
    ask_timeout: float
    max_question_chars: int
    log_level: str

    @property
    def has_embedding(self) -> bool:
        """是否配置了远程 embedding API（本地 bge 不依赖此项）。"""
        return bool(self.embedding_api_key and self.embedding_model)


def load_settings() -> Settings:
    api_key = os.getenv("LLM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("缺少 LLM_API_KEY。请复制 .env.example 为 .env 并填入密钥。")
    max_steps = int(os.getenv("MAX_STEPS", "6"))
    if max_steps < 1:
        raise RuntimeError("MAX_STEPS 必须是 >= 1 的整数。")
    timeout = float(os.getenv("LLM_TIMEOUT", "60"))
    if timeout <= 0:
        raise RuntimeError("LLM_TIMEOUT 必须是正数（秒）。")
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
    if max_retries < 0:
        raise RuntimeError("LLM_MAX_RETRIES 必须是 >= 0 的整数。")
    ask_timeout = float(os.getenv("ASK_TIMEOUT", "120"))
    if ask_timeout <= 0:
        raise RuntimeError("ASK_TIMEOUT 必须是正数（秒）。")
    max_question_chars = int(os.getenv("MAX_QUESTION_CHARS", "2000"))
    if max_question_chars < 1:
        raise RuntimeError("MAX_QUESTION_CHARS 必须是 >= 1 的整数。")
    return Settings(
        api_key=api_key,
        base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com").rstrip("/"),
        model=os.getenv("LLM_MODEL", "deepseek-flash"),
        embedding_api_key=_optional("EMBEDDING_API_KEY"),
        embedding_base_url=_optional("EMBEDDING_BASE_URL"),
        embedding_model=_optional("EMBEDDING_MODEL"),
        local_embedding_model=os.getenv("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5").strip()
        or "BAAI/bge-small-zh-v1.5",
        max_steps=max_steps,
        llm_timeout=timeout,
        llm_max_retries=max_retries,
        ask_timeout=ask_timeout,
        max_question_chars=max_question_chars,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
