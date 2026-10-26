# Qwen3-1.7B GPU 推理

本目录独立保存 Qwen3-1.7B 的代码、权重及运行记录。
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

服务只监听本机，无鉴权，支持文本非流式聊天和模型列表，不支持工具调用或
完整 OpenAI API。默认关闭思考模式，串行生成，输入加输出上限 4096 token，
单次输出上限 1024 token。按 Ctrl+C 停止前台服务。
