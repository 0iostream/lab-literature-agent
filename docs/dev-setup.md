# 开发环境与技术栈

## 技术栈

- **语言**：Python 3.11+
- **LLM**：OpenAI 兼容 API（推荐 DeepSeek / 通义 Qwen；见 `.env.example`）
- **核心依赖**：`openai`, `python-dotenv`, `pydantic`, `numpy`, `httpx`
- **本地 Embedding**：`sentence-transformers` + `BAAI/bge-small-zh-v1.5`（Day 10+）
- **可选**：`langgraph` / `langchain-core`（Day 15）、`mcp`（Day 14）
- **重排**：默认 `RERANKER_BACKEND=cross`（`BAAI/bge-reranker-base`），可降级 light

## 环境配置

1. 在**仓库根目录**（clone 后的项目根，即含 `src/`、`requirements.txt` 的目录）操作
2. 创建并安装：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

3. 编辑 `.env`：至少填 `LLM_API_KEY`；按需改 `LLM_BASE_URL` / `LLM_MODEL`
4. 常用旋钮：`MAX_STEPS`（默认 6）、`LLM_TIMEOUT`、`LLM_MAX_RETRIES`、`LOCAL_EMBEDDING_MODEL`、`RERANKER_BACKEND`

## 约定

- **不要**提交 `.env`、`.venv/`、索引与 traces（见 `.gitignore`）
- Activate.ps1 若被策略禁用：始终用 `.\.venv\Scripts\python.exe -m ...`
- 首次向量/重排会下载本地模型，需网络与磁盘空间
