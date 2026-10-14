# 实验室文献问答 Agent（Agentic RAG）

手写 **ReAct** 循环的本地文献问答：模型自主决定是否检索、改写 query、精读片段或计算；答案须带可验证 `[doc_id#chunk]` 引用，证据不足则拒答。

License：[MIT](LICENSE) · 仓库：https://github.com/0iostream/lab-literature-agent

## 能力概览

- **Agent Runtime**：Thought → tool_call → Observation；最大步数、重复调用检测、错误 Observation 回灌、引用门禁、LLM 超时重试
- **工具**：`search_docs` / `read_chunk` / `rewrite_query` / `python_calc`
- **检索**：词法 · 本地 bge-small-zh 向量 · RRF 混合 · 可选 cross-encoder rerank
- **评测**：检索金标与端到端金标；报告在 `data/eval/`
- **对照**：LangGraph 固定「检索→生成→引用检查」vs 手写 ReAct（`workflow-compare`）
- **HTTP**：FastAPI `GET /health`、`POST /ask`（本地 demo，无鉴权）

## 评测数字摘要（报告可复现）

语料为可控本地 Markdown 笔记；扩库后请重跑评测再引用数字。下列摘自已保存报告：

| 实验 | 规模 | 结果 |
| --- | --- | --- |
| 检索对照 | 22 条金标 | Recall@3：词法 / 向量 / 混合 = **0.57 / 0.68 / 0.75**；混合+rerank = **0.96**；MRR 混合 / +rerank = **0.68 / 0.91**（`retrieval_report.json`） |
| 端到端 | 25 条金标 | 工具选择 **1.00**、合法引用率 **1.00**、拒答 F1 **0.89**（`last_report.json`，本地生成） |
| Workflow vs ReAct | 10 道检索题 | 成功率 **0.90 / 1.00**；平均步数 **2.0 / 2.4**；平均延迟约 **6.8s / 19s**（`workflow_compare_report.json`） |

数字对应当前本地笔记库与自建金标；开放域未测。

## 非目标

- 从零预训练 / 完整 RLHF
- 多模态训练与复杂 Multi-Agent 编排
- 声称高并发生产部署（当前为同步单机 demo）

## 目录

```
src/react_agent.py   手写 ReAct
src/tools/           工具与 JSON Schema
src/rag/             切分、检索、embedding、rerank
src/eval/            评测脚本
src/api/             FastAPI
src/workflow/        LangGraph 固定链对照
data/docs/           本地文献（Markdown）
data/eval/           金标与评测报告
docs/                架构与跑通说明
```

## 环境

Python 3.11+。在**仓库根目录**：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

编辑 `.env`，填入 `LLM_API_KEY`（勿提交 `.env`）。推荐 DeepSeek 或通义 Qwen。

首次向量 / rerank 会下载本地模型。索引不入库，需自行：

```powershell
.\.venv\Scripts\python.exe -m src.cli ingest --with-embeddings
```

## 常用命令

```powershell
.\.venv\Scripts\python.exe -m src.cli agent "ReAct 三个字母代表什么？请引用文献。"
.\.venv\Scripts\python.exe -m src.cli search "什么是 ReAct" --mode hybrid_rerank
.\.venv\Scripts\python.exe -m src.cli retrieval-eval
.\.venv\Scripts\python.exe -m src.cli eval
.\.venv\Scripts\python.exe -m src.cli workflow-compare
.\.venv\Scripts\python.exe -m src.cli serve
```

更多说明见 [docs/build.md](docs/build.md)、[docs/architecture.md](docs/architecture.md)、[docs/testing.md](docs/testing.md)。

### HTTP API

| 接口 | 说明 |
| --- | --- |
| `GET /health` | 就绪检查（不跑 Agent） |
| `POST /ask` | `{"question":"..."}` → answer / citations / trace_id 等 |

错误码：`422` 校验失败；`504` 整题超时；`502` LLM 失败。每请求独立 memory；不做鉴权。

引用格式：`[doc_id#chunk]`，例如 `[react#0]`。
