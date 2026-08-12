# 食谱/食材语义分类器

这是一个独立的二分类领域门禁：

- `1`：食谱或食材相关，包括食材库存、保存、安全、过敏原和烹饪问题。
- `0`：其他问题，包括餐厅/外卖、厨电维修、餐饮经营以及通用非食品问题。

训练读取逻辑会读取 `classifier/dataset` 目录下所有符合规范的 CSV，支持 UTF-8
逗号分隔和 GBK Tab 分隔。列名可以是 `query,label` 或 `text,label`，标签只能是
`0`、`1`，合并后会按规范化文本去重并检查标签冲突。

## 安装

应用内推理依赖已经包含在根目录 `requirements.txt` 中：

```bash
.venv/bin/python -m pip install -r requirements.txt
```

训练使用官方 FastText 0.9.2 CLI，应用内推理使用兼容 `.bin` 文件的 `fasttext-predict`：

```bash
./classifier/build_fasttext.sh
```

脚本会从官方仓库获取 FastText v0.9.2，并将 Linux CLI 构建到
`.run/fasttext-linux-build/fasttext`。也可以安装系统 FastText CLI，并通过
`--fasttext-binary` 指定位置。

## 训练

```bash
.venv/bin/python -m classifier.train
```

训练流程会：

1. 读取 dataset 目录下所有符合规范的 CSV 并合并去重；
2. 按类别进行 70%/15%/15% 确定性划分；
3. 使用 NFKC、英文小写和 jieba 分词；
4. 使用训练集和验证集训练部署模型；
5. 在部署模型验证集上按 F1 优先并设置最低置信度 0.1 选择阈值；
6. 将完全未参与训练和阈值选择的测试集指标写入
   `classifier/artifacts/evaluation.json`。

输出文件：

```text
classifier/artifacts/
├── food_query.bin
├── food_query.vec
├── metadata.json
├── evaluation.json
└── training/
```

应用只需要 `food_query.bin` 和 `metadata.json`。模型不可用或加载失败时，
请求会 fail-open 进入现有流水线，并在执行轨迹中记录降级状态。

## 独立评估

```bash
.venv/bin/python -m classifier.evaluate \
  --dataset classifier/dataset
```
