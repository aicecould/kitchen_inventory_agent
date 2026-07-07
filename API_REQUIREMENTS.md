# 外部 API 调用要求

本文档记录厨房库存 Agent 原型使用的外部 API、鉴权方式、请求要求、返回字段和本地配置项。所有密钥统一保存在项目根目录的 `.env`，不得写入代码、文档、日志或提交到 Git。

## 1. 通用要求

- 所有请求使用 HTTPS。
- 请求超时由 `HTTP_TIMEOUT_SECONDS` 统一控制，默认 20 秒。
- 日志可以记录服务名称、状态码和错误信息，但不得记录 API Key、Secret Key、Access Token 或完整签名。
- `.env` 已加入 `.gitignore`；仓库只提交不含真实密钥的 `.env.example`。
- 外部响应进入 Agent 前必须转换为项目内部统一结构。
- 图片、查询文本和用户画像可能包含隐私信息，不得在错误日志中输出完整内容。
- Agent 的库存写工具只能创建待确认操作，不得直接写数据库；确认接口由本地白名单执行器处理。

## 2. DeepSeek API

### 用途

作为 LangChain Agent 的模型后端，负责理解上下文、自主选择库存或食谱工具，并基于工具结果生成最终回答。

### 官方资料

- API 入门：https://api-docs.deepseek.com/
- Function Calling：https://api-docs.deepseek.com/guides/function_calling/

### 本地配置

```dotenv
DEEPSEEK_API_KEY=
DEEPSEEK_MODEL=deepseek-v4-flash
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MAX_OUTPUT_TOKENS=800
DEEPSEEK_MAX_INPUT_CHARS=12000
```

### 调用要求

- DeepSeek 提供 OpenAI 兼容接口，项目通过 `langchain-openai` 的 `ChatOpenAI` 接入。
- `base_url` 使用 `https://api.deepseek.com`。
- 请求必须携带 `DEEPSEEK_API_KEY`。
- 当前默认模型为 `deepseek-v4-flash`。
- Agent 工具必须提供名称、说明和 JSON Schema 参数。
- 模型产生工具调用后，由本地 Python 函数执行；模型本身不得直接修改库存。
- 工具执行结果必须作为 Tool Message 返回模型，模型再生成最终回答。
- 使用 `AGENT_RECURSION_LIMIT` 限制 Agent 循环次数，防止无限工具调用。
- 单次模型调用最多生成 `DEEPSEEK_MAX_OUTPUT_TOKENS` 个 token。
- 发送给 Agent 的结构化输入不得超过 `DEEPSEEK_MAX_INPUT_CHARS` 个字符；超限会在调用 API 前拒绝。
- API 失败最多重试一次，避免故障期间重复消耗额度。

### 项目内位置

- 模型初始化：`app/agent.py`
- System Prompt：`app/prompts.py`
- Agent 上下文：`app/context.py`

## 3. 百度图像识别 API

### 用途

识别用户图片中的物体或场景，返回候选食材名称与置信度。

### 官方资料

- 通用物体和场景识别：https://ai.baidu.com/ai-doc/IMAGERECOGNITION/Mk3bcxfbi

### 本地配置

```dotenv
BAIDU_IMAGE_API_KEY=
BAIDU_IMAGE_SECRET_KEY=
BAIDU_IMAGE_ENDPOINT=https://aip.baidubce.com/rest/2.0/image-classify/v1/classify/ingredient
```

### 鉴权要求

1. 使用 API Key 和 Secret Key 请求 OAuth Access Token：

   `POST https://aip.baidubce.com/oauth/2.0/token`

2. 查询参数：

   - `grant_type=client_credentials`
   - `client_id=BAIDU_IMAGE_API_KEY`
   - `client_secret=BAIDU_IMAGE_SECRET_KEY`

3. 调用识别接口时通过 `access_token` 查询参数传递 Token。
4. Access Token 在内存中缓存，并在过期前刷新；不得写入数据库或日志。

### 图片请求要求

- 请求方法：`POST`。
- Content-Type：`application/x-www-form-urlencoded`。
- 图片以 Base64 字符串放入 `image` 字段。
- 调用 API 前先在本地完成非空、大小和可解码性检查。
- 原型按果蔬识别接口限制将原始图片上限设为 3 MiB，保证 Base64 编码后不超过 4 MiB。
- 图片仅支持 JPG、PNG、BMP；边长为 15–4096px，长宽比不超过 3:1。

### 使用的返回字段

| 字段 | 用途 |
|---|---|
| `result[].name` | 候选果蔬名称 |
| `result[].score` | 0 到 1 的识别置信度 |
| `result[].root` | 上层分类，当前原型不强制使用 |
| `error_code` / `error_msg` | API 错误判断 |

识别结果进入 Agent 前根据高、低两个阈值标记为 `accepted`、`needs_confirmation` 或 `rejected`。

### 项目内位置

- API 客户端：`app/adapters/vision_api.py`
- 图片校验与置信度处理：`app/vision.py`

## 4. Spoonacular API

### 用途

作为菜谱搜索的第一路数据源，根据现有食材查找菜谱并获取完整配料与烹饪说明。

### 官方资料

- API 网站：https://spoonacular.com/food-api

### 本地配置

```dotenv
SPOONACULAR_API_KEY=
SPOONACULAR_BASE_URL=https://api.spoonacular.com
```

### 主搜索请求

```text
GET /recipes/complexSearch
```

使用参数：

| 参数 | 说明 |
|---|---|
| `apiKey` | Spoonacular API Key |
| `includeIngredients` | Agent生成的食材查询词，以逗号分隔 |
| `intolerances` | Spoonacular支持的过敏原大类 |
| `excludeIngredients` | 具体过敏食材或需要排除的单品 |
| `diet` / `cuisine` / `type` | 可选饮食偏好、菜系和餐型 |
| `maxReadyTime` | 可选最大烹饪时间 |
| `sort` | `max-used-ingredients`或`min-missing-ingredients` |
| `number=3` | 后端固定返回数量 |
| `instructionsRequired=true` | 只返回有烹饪步骤的菜谱 |
| `ignorePantry=true` | 忽略常见基础调料 |
| `addRecipeInformation=true` | 在搜索响应中返回菜谱详情 |
| `addRecipeInstructions=true` | 在搜索响应中返回结构化步骤 |
| `addRecipeNutrition=false` | 原型不请求营养数据 |

项目不再逐条调用详情接口。标题、说明、扩展配料、来源URL和图片URL直接从 `complexSearch` 响应读取。Spoonacular具有配额限制，实际可用额度以账户控制台为准。

### 项目内位置

- API 客户端：`app/adapters/recipe_api.py`
- 双路路由：`app/tools/recipe.py`

## 5. TheMealDB API

### 用途

作为菜谱搜索的第二路数据源。当 Spoonacular 失败、无结果或结果不足时，仍可返回菜谱候选。

### 官方资料

- API Guide：https://www.themealdb.com/docs_api_guide.php

### 本地配置

```dotenv
THEMEALDB_API_KEY=1
THEMEALDB_BASE_URL=https://www.themealdb.com/api/json/v1
```

开发与学习阶段可使用测试 Key `1`。公开或生产用途应按照 TheMealDB 的服务要求申请相应 Key。

### 搜索请求

```text
GET /{api_key}/filter.php?i={main_ingredient}
```

免费 V1 接口按一个主要食材过滤。项目使用 Agent 提供的第一个查询词作为条件。

### 详情请求

```text
GET /{api_key}/lookup.php?i={meal_id}
```

项目读取：

- `strMeal`：菜谱名称；
- `strInstructions`：烹饪步骤；
- `strIngredient1..20`：配料；
- `strMeasure1..20`：配料用量；
- `strSource`：来源；
- `strMealThumb`：图片。

### 项目内位置

- API 客户端：`app/adapters/recipe_api.py`
- 双路路由：`app/tools/recipe.py`

## 6. 双路菜谱路由规则

1. Agent 按菜谱 API 的要求生成食材查询词和过敏原词，并作为工具参数传入。
2. 先调用Spoonacular `complexSearch`，成功且存在安全结果时立即返回。
3. Spoonacular失败、无结果或过滤后为空时，才调用TheMealDB。
4. Spoonacular使用官方 `intolerances`大类和`excludeIngredients`单品进行搜索阶段过滤。
5. 两个来源返回后都使用大类词与单品词执行最终字符串复核。
6. 菜谱工具保留API原始结果，由Agent使用目标语言生成最终回答。
7. 两路均失败且没有结果时，向Agent返回明确错误。

## 7. 配置检查清单

运行完整链路前，在本地 `.env` 中确认：

- [ ] 已填写 `DEEPSEEK_API_KEY`；
- [ ] 已填写百度图像识别 API Key 与 Secret Key；
- [ ] 已填写 `SPOONACULAR_API_KEY`；
- [ ] TheMealDB Key 符合当前使用场景；
- [ ] 各服务已在对应控制台开通；
- [ ] 未将 `.env` 加入 Git 暂存区；
- [ ] 日志和报错中没有输出密钥或 Token。
