# 代码组织规范

## 顶层目录

```
src/           运行时代码（包根）
data/docs/     本地文献（*.md）
data/index/    切分索引 chunks.json（可生成，勿手改为主）
data/eval/     金标与评测报告
traces/        运行轨迹 jsonl
notes/         bad case 等公开笔记（求职私货见 .gitignore）
docs/          架构 / 跑通 / 测试说明（本目录）
README.md / AGENTS.md / LICENSE
```

## `src/` 模块地图

| 路径 | 职责 |
| --- | --- |
| `react_agent.py` | 手写 ReAct 主循环、引用门禁、trace |
| `cli.py` | 命令行入口与子命令分发 |
| `config.py` | Settings / `.env` 加载 |
| `llm_client.py` | Chat Completions 客户端 |
| `guardrails.py` | 输入黑名单、粗 token 估计、写 trace |
| `tools/` | `python_calc` / `search_docs` / `read_chunk` / `rewrite_query` |
| `rag/` | chunker、embedder、reranker、DocumentStore |
| `memory/` | ConversationMemory |
| `eval/` | 检索评测、端到端、Day12 agentic 检查 |
| `api/` | FastAPI：`/health`、`/ask`（薄边界，不含 loop） |
| `workflow/` | 可选 LangGraph 固定链 |
| `mcp_server/` | 可选 MCP（主线可不实现） |
| `scratch/` | 教学 demo（如 attention） |

## 命名与约定

- 工具 **name** 用 snake_case，与 Function Calling 名称一致
- Chunk id：`{doc_stem}#{index}`，与引用格式一致
- 评测金标：`data/eval/gold.json`（端到端）、`retrieval_gold.json`（检索）
- 报告：`data/eval/last_report.json` 等；改指标后应重跑并保存
- 生成物（`__pycache__`、traces、index）不作为「源码真相」

## LOC / 改动建议

- 优先小步改：先跑通相关 CLI，再改循环或检索
- 新增工具：继承 `Tool`，注册进 `ToolRegistry`（见 `build_agent`）
- 避免为大段可选 Day（FastAPI/MCP）提前膨胀主循环
