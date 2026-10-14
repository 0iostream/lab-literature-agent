# 测试与评测指南

本仓库以**评测脚本 + 金标**为主，不是传统 pytest 套件为主线。

## 检索评测

- 金标：`data/eval/retrieval_gold.json`（及计划中的检索题）
- 入口：`python -m src.eval.run_retrieval_eval` 或 CLI `cmd_retrieval_eval`
- 指标：Recall@k、MRR；可对比 lexical / vector / hybrid / hybrid_rerank 与延迟

## 端到端评测（Day 13）

- 金标：`data/eval/gold.json`（目标约 30 条，保底 20）
- 入口：`python -m src.eval.run_eval`
- 报告：`data/eval/last_report.json`
- 关注：工具选择是否得当、合法引用率、拒答 F1、平均步数、延迟、粗 token

`build_agent()` 组装：`PythonCalc` + `SearchDocs(hybrid_rerank)` + `ReadChunk` + `RewriteQuery`。

## Day 12 Agentic 检查

- `python -m src.eval.run_agentic_day12`：rewrite 三连、算术/闲聊防误检索等
- 报告示例：`data/eval/agentic_day12_report.json`

## 手工冒烟（改完循环后建议）

1. `agent "计算 (3+5)*2"` → 应走 `python_calc`，不搜文献
2. `agent` 问 ReAct 含义 → 应检索并带 `[react#…]` 类引用
3. 问库中不存在的事实（如打印机电话）→ 应拒答/说不知道
4. 看对应 `traces/*.jsonl` 是否记录 tool 与 stopped_reason

## 验证修改的原则

- 改检索：重跑检索评测，保存报告再改简历数字
- 改 Agent 行为：重跑端到端或至少 Demo 三题 + 抽查 trace
- 引用门禁变更：确认 `has_valid_citation` 与金标 `should_refuse` 一致
