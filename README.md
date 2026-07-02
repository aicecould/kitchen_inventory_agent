# Kitchen Inventory Agent

以 Python 和 LangChain 为核心的厨房库存 Agent 后端原型。

当前仓库实现以下链路：

```text
输入 → 简单意图匹配 → 百度图像识别 API → Markdown 用户画像
    → DeepSeek LangChain Agent → 库存工具 / 双路食谱工具
    → 简单正则审核 → 输出
```

当前不实现：前端、订单管理、向量意图识别、RAG、Web Search、工具调用审查和第二模型复审。

详细设计见 [TECHNICAL_DESIGN.md](TECHNICAL_DESIGN.md)。

外部服务的鉴权、端点与请求要求见 [API_REQUIREMENTS.md](API_REQUIREMENTS.md)。

## 目录

```text
app/             Python 主包
app/adapters/    外部 API 适配器
app/tools/       Agent 可调用工具
data/            Markdown 用户画像及运行数据
tests/           原型测试
main.py          后端原型入口占位
```

## 初始化

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 中填写密钥后运行：

```powershell
python main.py --text "查询当前库存"
python main.py --text "识别图片中的食材并推荐菜谱" --image .\sample.jpg
python main.py --text "推荐一道菜" --language zh
```

启动展示页面：

```powershell
uvicorn app.web:app --reload
```

然后访问 `http://127.0.0.1:8000`。页面支持文字输入、图片上传、语言选择、库存查看、Agent 回答与工具调用轨迹展示。

展示库存确认流程时，可以输入：

```text
添加 2 个 番茄到库存
```

系统只会生成待确认卡片；点击“确认执行”后才会写入 SQLite，点击“取消”则不会修改库存。

所有密钥统一保存在 `.env`。该文件已由 `.gitignore` 排除；`.env.example` 只保存字段名和非敏感默认值。

DeepSeek 原型默认限制单轮模型输出最多 `800` token，送入 Agent 的结构化文本最多
`12000` 字符，并通过 `AGENT_RECURSION_LIMIT` 限制一次请求内的 Agent 循环次数。
这些限制可在 `.env` 中调整。

需要配置的服务凭据：

- `DEEPSEEK_API_KEY`：Agent 模型；
- `BAIDU_IMAGE_API_KEY` / `BAIDU_IMAGE_SECRET_KEY`：物体识别；
- `SPOONACULAR_API_KEY`：第一路菜谱搜索；
- `THEMEALDB_API_KEY`：第二路菜谱搜索，开发阶段默认使用测试键 `1`。
