# 评估复现

项目提供可重复的分层评估，覆盖分类与路由、本地混合检索、确定性审核、修复子 Agent、Cooking Session 和端到端路径。最终结果见 [快速评测报告](../docs/快速评测报告.md)。

## 运行命令

```bash
python -m pytest -q -s
python -m evaluation.run --suite quick --offline
python -m evaluation.run --suite quick
```

## 运行模式

离线模式不发送网络请求，适合复现分类、检索、审核、Cooking Session 和端到端路径。

在线模式最多运行 3 个 DeepSeek 领域门禁成本探针和 6 个修复子 Agent 样本；Spoonacular 每轮运行最多调用 5 次。

评估产物写入 `evaluation/results/`，同时保存逐样本 JSON 和 Markdown 报告。评测报告移动到 `docs/` 后，可直接查看 [快速评测报告](../docs/快速评测报告.md)。
