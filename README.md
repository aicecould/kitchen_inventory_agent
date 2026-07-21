# Kitchen Inventory Agent

以 Python 和 LangChain 为核心的厨房库存 Agent

当前仓库实现以下链路：

```mermaid
flowchart TD
    A["用户文本 / 图片"] --> B["输入校验、鉴权、意图匹配、图片识别"]
    B --> C["上下文构建：画像、过敏原、历史、识别置信度"]
    C --> D["编排 Agent：理解、规划、生成工具参数"]

    D --> E["库存读取工具"]
    D --> F["库存变更提案"]
    D --> G["菜谱搜索编排器"]
    D --> H["信息不足：请求用户澄清"]

    F --> I["用户确认 / 取消"]
    I -->|确认| J["白名单执行器事务写库"]
    I -->|取消| K["取消结果"]

    G --> L["API / 离线库 / Web 等多路检索"]
    L --> M["标准化、去重、过敏原过滤、排序"]

    E --> N["结构化工具结果"]
    J --> N
    K --> N
    M --> N
    N --> D

    D --> O["生成统一回答"]
    O --> P["事实一致性、工具调用一致性、过敏原与安全审核"]
    P -->|通过| Q["最终响应"]
    P -->|可修正| O
    P -->|不能安全回答| R["拒绝、降级或请求澄清"]
```

## 目录

```text
app/             Python 主包
app/adapters/    外部 API 适配器
app/tools/       Agent 可调用工具
data/            SQLite 库存数据库及用户画像
frontend/        前端
scripts/         PowerShell 启动与停止脚本
tests/           测试
```

## 初始化

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

在 `.env` 中填写密钥后运行

启动展示页面：

```powershell
.\scripts\start.ps1
```

然后访问 `http://127.0.0.1:8000`

停止应用：

```powershell
.\scripts\stop.ps1
```
