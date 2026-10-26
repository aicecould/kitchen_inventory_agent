# Qwen3-1.7B GPU 推理

本目录独立保存 Qwen3-1.7B 的代码和权重，运行记录放在项目 `.run/`。
不导入 classifier、recommender 或厨房 Agent，也不改变应用的模型配置。
仅使用官方现成权重，不训练、不微调、不量化。

## 下载

在项目根目录执行：

```bash
.venv/bin/python -m llm.qwen3_1_7b.download
```

下载器直连 ModelScope 的 `Qwen/Qwen3-1.7B` 仓库，显式禁用代理，
包括环境变量 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY` 及其小写形式。
国内 CDN 重定向也使用直连。版本与每个文件的大小、SHA-256 固定在
`model-manifest.json`，下载后逐一校验。支持并行分段及中断续传。
权重保存在本目录 `models/Qwen3-1.7B/`，部分分段保存在其 `.parts/` 下。

## 环境

依赖按用户要求保留在项目已有的 `requirements-xpu.txt`，不新建依赖文件。
运行时统一使用项目根目录现有的 `.venv`，不创建模型专用虚拟环境。
模型代码和权重仍保存在本目录，不导入其他模型子集。缓存沿用项目默认设置。
测试统一放在根目录 `tests/test_llm_runtime.py` 和 `tests/test_llm_download.py`，
与 classifier 共用项目的 pytest 入口和配置。

安装依赖必须使用已有清单。以下命令只为该安装进程移除代理，并使用国内
PyPI 镜像；XPU 专用 wheel 地址由已有依赖清单指定：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  uv pip install --python .venv/bin/python \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements-xpu.txt
```

## 推理与服务

一键依次下载、校验 GPU 推理、启动服务：

```bash
.venv/bin/python -m llm.qwen3_1_7b.deploy
```

各阶段写入 `.run/qwen3_1_7b/deployment-status.json`。任何阶段失败都会停止，不会
回退到 CPU。后台部署的输出写入 `.run/qwen3_1_7b/deploy.log`，
进程组编号保存在 `.run/qwen3_1_7b/deploy.pid`。
检查是否真正就绪应使用 `/health`。

```bash
.venv/bin/python -m llm.qwen3_1_7b.smoke
.venv/bin/python -m llm.qwen3_1_7b.serve --port 8001
```

只允许 GPU 推理，GPU 不可用直接失败。Intel GPU 先初始化 XPU，再导入
Transformers，避开 WSL 驱动的 Sysman 枚举段错误。全部模型参数、缓冲区、
输入和前向计算使用 GPU FP16；没有 CPU 推理或权重卸载回退。
分词及权重读取使用主机内存。

`smoke.py` 保存实际设备、参数位置、输出及耗时至
`.run/qwen3_1_7b/deployment-result.json`。
服务只有完成模型加载后才会开放 `/health`。

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-1.7B","messages":[{"role":"user","content":"鸡蛋和西红柿可以做什么菜？"}],"max_tokens":128}'
```

服务只监听本机，无鉴权，支持文本非流式聊天、工具调用、JSON 对象输出和模型列表，
不实现完整 OpenAI API。默认关闭思考模式，串行生成，输入加输出上限 32768 token，
单次输出上限 4096 token。JSON 模式会校验返回对象；无效或不完整的工具调用会报错，
不会交给 Agent 执行。按 Ctrl+C 停止前台服务。


## Agent 配置切换

Agent 通过 HTTP 接口访问模型，不导入本目录的模型运行时代码。
项目根目录 `.env` 使用 `LLM_PROVIDER` 选择模型；默认仍为 `deepseek`。
切换到本地 Qwen 时，先保证上面的本地服务正在运行，再设置：

```dotenv
LLM_PROVIDER=qwen
QWEN_BASE_URL=http://127.0.0.1:8001/v1
QWEN_MODEL=Qwen/Qwen3-1.7B
QWEN_API_KEY=local
QWEN_TIMEOUT_SECONDS=120
QWEN_MAX_OUTPUT_TOKENS=800
QWEN_MAX_INPUT_CHARS=60000
```

修改后重启 Agent 应用。`QWEN_API_KEY` 是兼容客户端所用的占位值，本地服务不验证它。
主 Agent、回答修复、画像总结、过期时间推断和提醒均使用选中的模型。
Qwen 模式无需 DeepSeek 密钥，不启用 DeepSeek 网络搜索，也不在本地服务失败时
自动回退到远程模型。Agent 的原有工具审核与待确认写入规则保持生效。
本地模型的指令遵循、工具选择和回答质量仍需按实际业务评估。

切回远程模型时设置 `LLM_PROVIDER=deepseek`，并使用原有的 `DEEPSEEK_*` 配置。
`/api/status` 的 `llm_provider` 和 `llm_model` 显示当前选择；本地模型进程是否已就绪
由 `http://127.0.0.1:8001/health` 检查。
