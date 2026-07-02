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
BAIDU_IMAGE_ENDPOINT=https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general
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
- 原型默认图片上限为 8 MiB；如百度账户对应接口限制更低，应以官方控制台限制为准。

### 使用的返回字段

| 字段 | 用途 |
|---|---|
| `result[].keyword` | 候选物体或食材名称 |
| `result[].score` | 0 到 1 的识别置信度 |
| `result[].root` | 上层分类，当前原型不强制使用 |
| `error_code` / `error_msg` | API 错误判断 |

识别结果进入 Agent 前根据高、低两个阈值标记为 `accepted`、`needs_confirmation` 或 `rejected`。

### 项目内位置

- API 客户端：`app/adapters/vision_api.py`
- 图片校验与置信度处理：`app/vision.py`

## 4. 百度通用翻译 API

### 用途

- 将中文食材名称翻译为英文，供国外菜谱 API 搜索。
- 将英文菜谱标题、配料和烹饪步骤翻译为目标语言。

### 官方资料

- 百度翻译开放平台：https://api.fanyi.baidu.com/
- 通用翻译产品：https://api.fanyi.baidu.com/product/11

### 本地配置

```dotenv
BAIDU_TRANSLATE_APP_ID=
BAIDU_TRANSLATE_SECRET_KEY=
BAIDU_TRANSLATE_ENDPOINT=https://fanyi-api.baidu.com/api/trans/vip/translate
```

### 签名要求

每次请求生成随机 `salt`，并按以下顺序拼接：

```text
appid + q + salt + secret_key
```

对拼接结果执行 MD5，得到 `sign`。计算签名前的 `q` 不进行 URL Encode。

### 请求参数

| 参数 | 说明 |
|---|---|
| `q` | 待翻译 UTF-8 文本 |
| `from` | 源语言，默认 `auto` |
| `to` | 目标语言，例如 `zh` 或 `en` |
| `appid` | 百度翻译 APPID |
| `salt` | 每次请求生成的随机数 |
| `sign` | MD5 签名 |

### 返回处理

- 从 `trans_result[].dst` 读取译文。
- 返回 `error_code` 时视为调用失败。
- 过敏原在菜谱过滤前也需要翻译为英文，避免中文过敏原无法匹配英文配料。

### 项目内位置

- API 客户端：`app/adapters/translation_api.py`
- 菜谱翻译和过敏原处理：`app/tools/recipe.py`

## 5. Spoonacular API

### 用途

作为菜谱搜索的第一路数据源，根据现有食材查找菜谱并获取完整配料与烹饪说明。

### 官方资料

- API 网站：https://spoonacular.com/food-api

### 本地配置

```dotenv
SPOONACULAR_API_KEY=
SPOONACULAR_BASE_URL=https://api.spoonacular.com
```

### 搜索请求

```text
GET /recipes/findByIngredients
```

使用参数：

| 参数 | 说明 |
|---|---|
| `apiKey` | Spoonacular API Key |
| `ingredients` | 英文食材名称，以逗号分隔 |
| `number` | 返回数量 |
| `ranking=1` | 优先减少缺失食材 |
| `ignorePantry=true` | 忽略常见基础调料 |

### 详情请求

对搜索结果中的每个菜谱 ID 请求：

```text
GET /recipes/{id}/information
```

项目使用标题、说明、扩展配料、来源 URL 和图片 URL。Spoonacular 具有配额限制，实际可用额度以账户控制台为准。

### 项目内位置

- API 客户端：`app/adapters/recipe_api.py`
- 双路路由：`app/tools/recipe.py`

## 6. TheMealDB API

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

免费 V1 接口按一个主要食材过滤。项目使用翻译后的第一个食材作为查询条件。

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

## 7. 双路菜谱路由规则

1. 使用百度翻译将食材和过敏原转换为英文。
2. 优先并独立调用 Spoonacular 与 TheMealDB。
3. 单一路由失败时保留另一路结果，不使整个工具立即失败。
4. 按 `source + id` 去重。
5. 在翻译输出前执行过敏原过滤。
6. 使用百度翻译将最终菜谱转换为指定语言。
7. 两路均失败且没有结果时，向 Agent 返回明确错误。

## 8. 配置检查清单

运行完整链路前，在本地 `.env` 中确认：

- [ ] 已填写 `DEEPSEEK_API_KEY`；
- [ ] 已填写百度图像识别 API Key 与 Secret Key；
- [ ] 已填写百度翻译 APPID 与 Secret Key；
- [ ] 已填写 `SPOONACULAR_API_KEY`；
- [ ] TheMealDB Key 符合当前使用场景；
- [ ] 各服务已在对应控制台开通；
- [ ] 未将 `.env` 加入 Git 暂存区；
- [ ] 日志和报错中没有输出密钥或 Token。
