"""FastAPI 薄封装：GET /health、POST /ask。

- API 层：输入校验、request_id、整题超时、HTTP 状态码
- Agent 层：推理循环、工具、引用门禁（见 src/react_agent.py）
- 每个 /ask 使用独立 ConversationMemory，避免请求串会话
"""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from src.config import Settings, load_settings
from src.eval.run_eval import build_agent
from src.llm_client import LLMError
from src.memory.conversation import ConversationMemory
from src.react_agent import ReactAgent, extract_citation_ids
from src.rag.store import DocumentStore

_executor = ThreadPoolExecutor(max_workers=4)
_agent: ReactAgent | None = None
_store: DocumentStore | None = None
_settings: Settings | None = None


class AskRequest(BaseModel):
    question: str = Field(..., description="用户问题")

    @field_validator("question")
    @classmethod
    def strip_nonempty(cls, value: str) -> str:
        text = (value or "").strip()
        if not text:
            raise ValueError("question 不能为空")
        return text


class AskResponse(BaseModel):
    answer: str
    citations: list[str]
    trace_id: str
    stopped_reason: str
    request_id: str
    steps: int
    tools: list[str]


class HealthResponse(BaseModel):
    status: str
    model: str
    chunk_count: int
    has_embeddings: bool


class ErrorBody(BaseModel):
    error: str
    detail: str
    request_id: str


def _ensure_ready() -> tuple[ReactAgent, DocumentStore, Settings]:
    if _agent is None or _store is None or _settings is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "service_unavailable", "detail": "Agent 尚未完成初始化"},
        )
    return _agent, _store, _settings


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _agent, _store, _settings
    _settings = load_settings()
    _agent, _store = build_agent()
    yield
    _executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="实验室文献问答 Agent",
    description="API 边界薄封装；核心仍是手写 ReAct loop。",
    version="0.1.0",
    lifespan=lifespan,
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) or request.headers.get(
        "x-request-id", ""
    )
    if isinstance(exc.detail, dict):
        payload = {"request_id": request_id, **exc.detail}
    else:
        payload = {
            "error": "http_error",
            "detail": str(exc.detail),
            "request_id": request_id,
        }
    return JSONResponse(status_code=exc.status_code, content=payload)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-Id"] = request_id
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """只检查服务与文献库是否就绪，不跑 Agent。"""
    _, store, settings = _ensure_ready()
    return HealthResponse(
        status="ok",
        model=settings.model,
        chunk_count=len(store._chunks),
        has_embeddings=store.has_embeddings,
    )


@app.post("/ask", response_model=AskResponse)
def ask(
    body: AskRequest,
    request: Request,
    x_request_id: str | None = Header(default=None),
) -> AskResponse:
    """接收问题，返回答案与可观测字段；不在此层做推理。"""
    agent, _, settings = _ensure_ready()
    request_id = x_request_id or getattr(request.state, "request_id", uuid.uuid4().hex)

    if len(body.question) > settings.max_question_chars:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "validation_error",
                "detail": f"question 最长 {settings.max_question_chars} 字符",
                "request_id": request_id,
            },
        )

    # 每请求独立会话，避免并发/连续请求串记忆
    agent.memory = ConversationMemory()

    def _run() -> Any:
        return agent.run(body.question)

    future = _executor.submit(_run)
    try:
        result = future.result(timeout=settings.ask_timeout)
    except FuturesTimeout as exc:
        raise HTTPException(
            status_code=504,
            detail={
                "error": "ask_timeout",
                "detail": f"Agent 在 {settings.ask_timeout:.0f}s 内未完成",
                "request_id": request_id,
            },
        ) from exc
    except LLMError as exc:
        raise HTTPException(
            status_code=502,
            detail={
                "error": "llm_error",
                "detail": str(exc),
                "request_id": request_id,
            },
        ) from exc
    except Exception as exc:  # noqa: BLE001 — 边界层统一成 500
        raise HTTPException(
            status_code=500,
            detail={
                "error": "internal_error",
                "detail": str(exc),
                "request_id": request_id,
            },
        ) from exc

    # llm_error 已在 Agent 内收成答案时，仍 200 但 stopped_reason 标明
    citations = extract_citation_ids(result.answer)
    return AskResponse(
        answer=result.answer,
        citations=citations,
        trace_id=result.trace_id,
        stopped_reason=result.stopped_reason,
        request_id=request_id,
        steps=result.steps,
        tools=result.tool_calls,
    )
