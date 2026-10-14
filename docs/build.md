# Build / 跑通指南

在仓库根目录执行。以下均假设使用 venv 解释器。

## 一次性准备

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
# 编辑 .env 填入 LLM_API_KEY
```

## 构建文献索引

```powershell
# 切分 data/docs → data/index/chunks.json
.\.venv\Scripts\python.exe -m src.cli ingest

# 需要向量 / hybrid 时再编码（首次会下载 bge-small-zh）
.\.venv\Scripts\python.exe -m src.cli ingest --with-embeddings
```

文献语料现有：`data/docs/rag.md`、`react.md`、`transformer.md`、`workflow_agent.md`。

## 常用运行命令

| 目的 | 命令 |
| --- | --- |
| 多轮纯对话 | `python -m src.cli chat` |
| 跑 Agent | `python -m src.cli agent "计算 (3+5)*2"` |
| 交互 Agent | `python -m src.cli agent` |
| 检索试跑 | `python -m src.cli search "什么是 ReAct"` |
| 连续 Demo | `python -m src.cli demo` |
| Attention demo | `python -m src.scratch.attention_demo` |
| FastAPI 服务 | `python -m src.cli serve` → `GET /health`、`POST /ask` |
| Workflow vs ReAct | `python -m src.cli workflow-compare`（需 API；报告 `data/eval/workflow_compare_report.json`） |

（Windows：把 `python` 换成 `.\.venv\Scripts\python.exe`。）

## 注意事项

- 缺 `LLM_API_KEY` 会在 `load_settings()` 直接失败
- 无索引时部分命令会现场 `ingest_dir`；正式评测前应先落盘索引
- Agent 默认 `max_steps` 来自环境变量；超时会有限重试后再抛 `LLMError`
