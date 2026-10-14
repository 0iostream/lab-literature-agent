# AGENTS.md

本文件是给人类开发者和 AI coding agent 的**项目入口索引**。保持精简：必要说明与硬性规则在此；细节请跟链接进入 `docs/`。

## 项目一句话

实验室文献问答的 **Agentic RAG**：手写 ReAct 循环，模型自主决定是否检索 / 改写 query / 精读片段 / 计算，回答须带可验证引用，证据不足则拒答。不是 ChatGPT 套壳聊天。

## 硬性规则（改代码前必读）

1. **不要**把主线改成「单轮 RAG 聊天」或堆 LangChain/CrewAI 名词；核心是 `src/react_agent.py` 手写循环。
2. 需要事实时必须先 `search_docs`；`NO_HITS` 再用 `rewrite_query`，不要一上来就改写。
3. 最终回答引用格式必须是可在文献库查到的 `[doc_id#chunk]`（如 `[react#0]`）；证据不足要明确说不知道。
4. 算术用 `python_calc`，不要心算；闲聊/寒暄不要误调检索。
5. 密钥只放 `.env`，禁止提交；Windows 优先用 `.\.venv\Scripts\python.exe`，不要用系统 Python。
6. 对外展示的数字必须来自保存的评测报告（题数、模型、语料、配置齐全）；未完成的功能不要写成已完成。
7. 主线优先级：可靠性（引用校验 + LLM 超时/重试）> 词法 vs 向量同集对比 > 端到端评测 > 混合检索 > FastAPI > LangGraph > MCP。

## 文档入口（按需打开）

| 类别 | 入口 |
| --- | --- |
| 架构与心智模型 | [docs/architecture.md](docs/architecture.md) |
| 开发环境与技术栈 | [docs/dev-setup.md](docs/dev-setup.md) |
| 代码组织规范 | [docs/code-structure.md](docs/code-structure.md) |
| Build / 跑通指南 | [docs/build.md](docs/build.md) |
| 测试与评测指南 | [docs/testing.md](docs/testing.md) |
| 历史问题与教训 | [docs/lessons.md](docs/lessons.md) |
| 对外 README | [README.md](README.md) |

学习计划、简历字弹、JD/面经等材料仅存本地（见 `.gitignore`），不进入公开仓库。

## Agent 改代码时的建议顺序

1. 读本文件规则 + [docs/architecture.md](docs/architecture.md)
2. 改功能前对照对应模块（见 code-structure）
3. 用 [docs/build.md](docs/build.md) / [docs/testing.md](docs/testing.md) 的命令本地验证
4. 若踩坑，把根因补进 [docs/lessons.md](docs/lessons.md)
