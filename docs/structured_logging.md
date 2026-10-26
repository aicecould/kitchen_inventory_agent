# Query 结构化日志

一个 Query 对应一个 JSON 文档，在 SQLite 的 `query_logs` 表中占一行。
组件、模型和工具调用保存在这条记录的 `steps` 内，不产生独立日志行。
日志用于保存事实，供后续分析、评测或构建数据集使用。

## 存储与生命周期

默认启用，数据库路径为 `data/llm_logs.db`。可通过 `LLM_LOG_DB_PATH` 修改位置，
通过 `LLM_LOGGING_ENABLED=false` 关闭。数据来源由 `LLM_LOG_SOURCE` 指定，默认为 `application`。

SQLite 普通列保存 `id`、`user_id`、`segment_id`、`source`、`status`、`started_at`、`ended_at`，
用于索引；`record_json` 保存完整文档。

请求开始插入一行，状态为 `running`。执行过程中将已完成的明细更新到同一行；
结束时在同一行写入最终结果、结束时间及用量汇总。更新使用事务，避免并行工具相互覆盖。
进程中断时保留 `running` 状态和此前已写入的执行过程。

日志写入异常只输出错误类型，业务继续执行。不会记录 LangChain 内部调度链节点。

## JSON 内容

| 字段 | 内容 |
| --- | --- |
| `id`, `schema_version`, `name`, `source` | Query ID、结构版本、请求类型、数据来源 |
| `user_id`, `segment_id` | 用户与对话段引用 |
| `request`, `prompt_hash` | 原始输入和实际输入指纹 |
| `runtime` | 代码指纹、源码文件指纹、Python/依赖版本、配置白名单、模型与检索元数据指纹；每条 Query 保存一次 |
| `metadata` | 已有运行/样本引用、触发来源等上下文 |
| `steps` | 有业务意义的组件执行、模型调用、工具调用和决策事件 |
| `result` | 最终内容、业务返回信息、执行状态和错误类型 |
| `time` | UTC 开始/结束时间和毫秒耗时 |
| `models` | 服务响应声明的模型名称；缺失身份的调用不以配置名称填充 |
| `tokens` | 已知用量之和、模型调用次数、已返回用量的调用数、用量是否完整 |

每个 `steps` 明细保留自身的 ID、父明细 ID、类型、名称、输入、输出、参数、状态、耗时。
模型明细额外包含模型名、完整消息、工具请求、响应元数据、原始 usage、token 用量、
system prompt 指纹和参数指纹。工具结果保留 `tool_call_id`，可与模型发出的工具请求对应。

主 Agent、修复、回答重写、菜谱恢复、画像总结、过期推断、主动提醒和 DeepSeek 网页搜索
分别标识职责，但都归属当前 Query。修复/重试和网页搜索续轮不会增加 Query 记录。
独立后台调用或单个评测输入在自身的执行范围内产生一条 Query 记录。

`tokens.complete=false` 表示部分调用缺少供应商用量，汇总仅表示已知部分。
单次调用缺失的 token 字段为 `null`；完全没有模型调用时总用量为 0。
调用成功或 Query 执行成功不代表答案正确。SDK 内部 HTTP 自动重试计入一次模型调用，
应用层重试在当前 Query 内另记调用明细。

请求还保存目标语言、任务引用、当时的画像内容；最终结果保留工具历史和提案。
图片原始字节按 SHA-256 去重，保存在日志库同名的 `.assets/` 目录中，JSON 内保存摘要、大小、路径。
备份或迁移时需一并保留该目录。没有实际读取的库存或外部状态不会额外快照。

## 查看与导出

```bash
.venv/bin/python -m app.log_analysis --query-id QUERY_ID
.venv/bin/python -m app.log_analysis --segment-id SEGMENT_ID
.venv/bin/python -m app.log_analysis --source application --name response_repair
.venv/bin/python -m app.log_analysis --kind llm --model MODEL_NAME
.venv/bin/python -m app.log_analysis --limit 1000 > queries.jsonl
.venv/bin/python -m app.log_analysis --summary
```

JSONL 每行是一个完整 Query。按组件或模型筛选也只筛选 Query，不拆开输出组件明细。
默认返回最近 100 条 Query。`--summary` 汇总匹配 Query 的已知用量和平均请求耗时，
同时提供 `queries`、`model_calls`、`calls_with_usage`、`usage_complete`。
筛选某个模型会选中包含该模型的 Query，Query 内的其他模型调用及其用量仍然保留。

也可直接查询 SQLite：

```sql
SELECT id, status, json_extract(record_json, '$.result.content') AS content
FROM query_logs
ORDER BY started_at DESC;
```

测试使用独立临时日志库，不向应用日志库写入构造的数据。实际业务数据不作为测试夹具。
日志保留完整输入和上下文，当前不自动清理。不采集请求头或客户端对象；结构化凭据字段会替换，
正文中的敏感信息不做自动识别。异常只记录类型，避免把供应商错误正文中的凭据写入日志。

## 模型身份（schema_version=2）

每个模型步骤的 `model_identity` 区分 `configured_model`、`requested_model` 和
`response_model`。`model` 与 `models` 只汇总响应声明；`source=response_body` 表示
读取 SDK 补齐前的原始响应字段，缺失时为 `unknown`。`sdk_model` 保留 SDK 汇总名称，
不能用来确认实际模型。`independently_verified=false` 表示未独立核验远端部署权重。
旧日志仅能依据已保存的消息元数据恢复，来源标为 `message_response_metadata`。

可在备份后纠正旧日志的混合字段：

```bash
.venv/bin/python -m app.log_migration --db data/llm_logs.db
```

迁移自动使用 SQLite backup 保存完整数据库，旧值留在 `legacy_model`，只处理 v1，
不修改模型消息、工具结果或历史回答。重复执行不重复修改已迁移记录。
新请求的 `result.public_response` 保存实际提供给前端的字段，原始 `tool_history`
和内部审计原因只保留在日志与内部处理链中。完整业务规则见 [business_flow.md](business_flow.md)。
