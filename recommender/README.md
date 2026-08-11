# 本地混合菜谱检索

本模块为 `search_recipes` 提供本地优先的 SQLite FTS5 + FAISS + RRF 检索，并在本地结果不足 3 条时由现有 Spoonacular、TheMealDB 和 DeepSeek Web Search 依次补齐。

## 数据与许可

- 数据固定来自 `josephrmartinez/recipe-dataset` 提交 `c5a1e8dab69d46841d914be1f20cca6fbe4b6260`。
- 输入为完整 `13k-recipes.csv`。
- 上游数据声明许可为 CC BY-SA 3.0。

## 一次性构建

构建入口只存在于 `recommender/build.py`，由开发或部署环境显式运行，主应用不会下载 CSV、创建数据库、生成向量或重建索引。

```bash
python -m recommender.build \
  --source /absolute/path/to/13k-recipes.csv \
  --model /absolute/path/to/bge-small-en-v1.5 \
  --output-dir recommender/artifacts
```

正式产物为：

- `recipes.db`：SQLite recipes 表与 FTS5 索引；
- `recipes.faiss`：以 SQLite recipe ID 为外部 ID 的 FAISS `IndexIDMap2`；
- `metadata.json`：模型、版本、哈希、维度、数量和 UTC 构建时间。

embedding 模型固定为 `BAAI/bge-small-en-v1.5` revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`，最大序列长度 512，向量维度 384。运行时只从部署前准备好的本地模型目录加载，并使用 `local_files_only=True`，不会隐式联网下载。

## 运行时行为

- FTS5 与 FAISS 各召回 top 100。
- RRF 固定 `k=60`，rank 从 1 开始。
- 排序支持 `default`、`max-used-ingredients`、`min-missing-ingredients`。
- 过敏原与排除食材在 RRF 后执行结构化硬过滤。
- 饮食类型、菜系、餐型和制作时间只进入语义查询文本，不执行结构化硬过滤。
- 本地至少 3 条时不调用外部来源；本地 1–2 条时保留本地结果并由 API/Web Search 补齐。
- 全部来源不足 3 条时返回真实可用结果并标记 degraded，不虚构菜谱。
- 本阶段不实现 `bge-reranker`，也不保留空调用。
