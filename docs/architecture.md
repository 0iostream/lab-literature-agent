# 架构概览与心智模型

## 这是什么系统

面向「本地 Markdown 文献库」的问答 Agent。模型通过 **Function Calling** 选择工具；工具在本地执行，结果以 Observation 回灌；循环受 `MAX_STEPS`、重复调用检测、引用门禁约束。

```
用户问题
  → ReactAgent.run（Thought / tool_call / Observation）
      ├─ python_calc      安全算术
      ├─ search_docs      本地检索（默认 hybrid_rerank）
      ├─ read_chunk       按 chunk_id 精读
      └─ rewrite_query    NO_HITS 后改写查询
  → 最终答案（须含合法 [doc#chunk] 或明确拒答）
  → traces/*.jsonl 事件日志
```

## 核心概念

| 概念 | 含义 |
| --- | --- |
| ReAct | 模型交替推理与行动；本仓库手写循环，不依赖 LangChain 主路径 |
| Chunk | 文献切分单元，id 形如 `react#0`（`doc_id#index`） |
| NO_HITS | `search_docs` 无结果时的约定字符串，触发改写重搜 |
| 合法引用 | 答案中的 citation id 必须能在 `DocumentStore.get` 查到 |
| 会话记忆 | `ConversationMemory`；Demo/评测每题应隔离，避免串话 |
| Trace | 每次 run 写独立 `traces/<timestamp>-<uuid>.jsonl` |

## 组件关系

- **LLMClient**（`src/llm_client.py`）：OpenAI 兼容 Chat Completions；超时与有限重试
- **ToolRegistry**（`src/tools/`）：JSON Schema → Function Calling；`run` 返回字符串 Observation
- **DocumentStore**（`src/rag/store.py`）：切分入库；lexical / vector / hybrid / hybrid_rerank
- **ReactAgent**（`src/react_agent.py`）：主循环、引用校验、重复调用、max_steps 收尾
- **CLI**（`src/cli.py`）：chat / agent / ingest / search / demo / 各类 eval 入口
- **Eval**（`src/eval/`）：检索指标与端到端（工具选择、引用、拒答、延迟等）

## 明确非目标

从零预训练、完整 RLHF、多模态训练、复杂 Multi-Agent 编排、做成网页套壳聊天。MCP / FastAPI / LangGraph 为可选加深，不是主线门禁。
