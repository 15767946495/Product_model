# CropNet Model — 县级玉米单产预测

基于**面板感知时间融合变换器（PA-TFT）**的县级玉米单产预测研究。

- **研究对象**：美国 29 个州 741 个玉米主产县，2017—2022 年平衡面板（4446 个县年）
- **数据**：CropNet 的 Sentinel-2、WRF-HRRR 与 USDA Corn 数据，外加 7 维县级土壤属性
- **方法**：县均值偏移 + 县内去均值植被指数 + 按样本规模设定的训练协议，TFT 作为时序后端
- **配套论文**：`paper.md`

## 目录结构

```
cropnet_model/
├── TFT_model/            # 主模型 PA-TFT 与对照配置 PVT-TFT（训练、推理、消融、干预与论文数值脚本）
├── UNet-ConvLSTM/        # 比较模型
├── GNN-RNN/              # 比较模型
├── PhenoYieldNet/        # 比较模型
├── train_dataset/        # 数据集构建脚本（JSONL / 网格缓存 / 土壤映射）
├── DataSrc/              # 数据下载与网格对齐
├── paper_figures/        # 论文数值结果（JSON）与图
├── docs/                 # 论文章节片段与表格生成脚本（本地，不进入 git）
├── Reference/            # 参考文献 PDF
├── paper.md              # 论文草稿
└── requirements.txt      # Python 依赖（conda env `product` 导出）
```

## 环境与依赖

```bash
# 依赖见 requirements.txt（由 conda 环境 product 导出）
pip install -r requirements.txt
# 主要依赖：torch>=2.0, numpy, pandas, scipy, scikit-learn, matplotlib
```

`requirements.txt` 中的 Torch 声明与本次已验证的 `product` 环境可能不一致，二者需要后续独立同步。本轮不自动升级/降级 Torch/CUDA；短训练和 CLI 验证使用当前 `product` 环境。

## 数据说明（重要）

所有模型只读取一个加工数据入口：`train_dataset/dataset.jsonl`。

| 路径 | 内容 |
|------|------|
| `/data/raid0/hqx/DataSrc/` | 唯一源数据根目录（USDA、WRF-HRRR、Sentinel-2 AG、gSSURGO 土壤、县界邻接、`download_audit.jsonl`） |
| `/data/raid0/hqx/train_dataset/dataset.jsonl` | 唯一加工数据实体文件，每行一个县年样本 |
| `train_dataset/dataset.jsonl` | 项目入口，必须是指向上述实体文件的符号链接 |

两个入口脚本：

```bash
P=/root/miniconda3/envs/product/bin/python
# 1) 下载与源数据审计（固定写入 /data/raid0/hqx/DataSrc）
$P DataSrc/download_cropnet.py --audit-only     # 只列远端清单，不下载
$P DataSrc/download_cropnet.py --download       # 增量下载，可中断重跑
$P DataSrc/download_cropnet.py --verify-local   # 只读审计，写 download_audit.jsonl
# 2) 源数据 -> 唯一 JSONL（原子写入并维护项目软链接）
$P train_dataset/build_dataset.py --validate-only
$P train_dataset/build_dataset.py
```

共享读取接口：`train_dataset/dataset.py`（schema、`CropNetDataset`、年份协议）、`collate.py`、`metrics.py`。

验证：

```bash
$P train_dataset/build_dataset.py --validate-only
readlink -f train_dataset/dataset.jsonl
```

`train_dataset/dataset.jsonl` 软链接被 `.gitignore` 忽略，所以 `git status` 中看不到，IDE 中显示为灰色；用上面的 `readlink -f` 确认它指向 RAID 实体文件。

旧运行时目录删除闸门：`$P train_dataset/migration_check.py` 只报告不删除；仅当输出
`READY_TO_DELETE_OLD_RUNTIME=true` 且用户再次明确批准后，才人工处理旧运行时目录。

正式入口只依赖 `train_dataset/dataset.jsonl` 和共享数据适配器，不依赖历史 manifest、旧缓存或旧运行时路径。本文档不把历史文件是否已物理删除作为入口契约。

## 推荐入口

所有正式入口均使用统一 JSONL，推荐直接运行各模型的 `run_train.sh`。除 TFT 使用 20 epoch 外，其余脚本默认 50 epoch；脚本固定使用 GPU 0--4，并把输出写入各自的模型目录，不使用公共输出根：

| 模型 | 推荐脚本 | GPU | 默认输出目录 |
|------|----------|-----|--------------|
| PhenoYieldNet | `./PhenoYieldNet/run_train.sh` | GPU 0 | `PhenoYieldNet/output_dir` |
| GNN-RNN | `./GNN-RNN/run_train.sh` | GPU 2 | `GNN-RNN/output_dir` |
| UNet-ConvLSTM | `./UNet-ConvLSTM/run_train.sh` | GPU 3 | `UNet-ConvLSTM/output_dir` |
| TFT | `./TFT_model/run_train.sh` | GPU 4 | `TFT_model/output_dir` |

所有脚本都接受环境变量覆盖，常用变量如下：

| 变量 | 默认值或作用 |
|------|--------------|
| `PYTHON` | `/root/miniconda3/envs/product/bin/python` |
| `DATASET_JSONL` | `$PROJECT_ROOT/train_dataset/dataset.jsonl` |
| `DATA_ROOT` | `/data/raid0/hqx/DataSrc` |
| `OUTPUT_ROOT` | 当前模型的 `output_dir` |
| `EPOCHS` | `50`（TFT 为 `20`） |
| `RUN_ID` | UTC 时间戳；用于输出 run 子目录 |
| `NUM_WORKERS` | `4` |
| `DRY_RUN` | `1` 只打印命令，不创建训练输出 |

模型专属变量包括 PhenoYieldNet 的 `STAGE1_EPOCHS`、`STAGE2_EPOCHS`、`GRAD_CHECKPOINTING`、`FROZEN_FM`，以及 GNN-RNN 的 `AGGREGATOR`、UNet-ConvLSTM 的 `COUNTY_BATCH_SIZE`/`IMAGE_SIZE`/`GRID_CHUNK`、TFT 的 `BATCH_SIZE`/`HIDDEN_SIZE`/`NUM_HEADS`/`NUM_LSTM_LAYERS`/`WARMUP_EPOCHS`/`CUTOFFS`。各模型 README 记录完整变量集合。

最小 smoke 只验证命令编排和输出契约：

```bash
DRY_RUN=1 EPOCHS=1 RUN_ID=smoke ./GNN-RNN/run_train.sh
DRY_RUN=1 EPOCHS=1 RUN_ID=smoke ./PhenoYieldNet/run_train.sh
```

PhenoYieldNet 脚本依次执行 Stage 1、Stage 2 和最终测试，两个阶段的 `checkpoint-final.pth` 自动通过模型目录中的 `OUTPUT_ROOT` 传递。

TFT 默认只从本地 `/data/raid0/hqx/pretrained/pvt_tiny.pth` 读取 PVT-Tiny，期望 SHA-256 为 `360ec6b56fe98f36bf9af6822f3e2a46a3acf2eee5bfe0c68f8580cf3c07ba0f`，正式路径不联网。文件缺失或 SHA 不匹配会直接失败；需要无预训练权重的离线 smoke 时使用 `VIT_PRETRAINED=0`，不会触发联网回退。

监督训练/最终测试 run 输出七个统一文件：`config.json`、`checkpoint-latest.pth`、`checkpoint-final.pth`、`train_metrics.jsonl`、`test_metrics.json`、`county_predictions.jsonl`、`run_metadata.json`。PhenoYieldNet Stage 1 只输出五个训练文件，不执行 2022 最终评估。上述短训练只用于验证入口、数据适配、优化步骤、checkpoint 和统一输出契约，不是性能结果，不得用于模型排名或论文结论。

## 论文数值复现

`paper.md` 只报告一折：2017—2021 年训练、2022 年测试，每个配置只跑一次、种子固定为 0（第5.1节）。唯一的例外是对照配置 PVT-TFT，它以三个种子重复训练以确认退化不是偶然。完整流程如下。

```bash
P=/root/miniconda3/envs/product/bin/python
export OUTPUT_ROOT=/data/raid0/hqx/TFT_out      # 训练输出务必外置，不要写进仓库

# 1) 训练：PA-TFT 与三个组件消融（2022 折、种子 0），以及 PVT-TFT 的三个种子
bash TFT_model/run_loyo.sh                                  # improved / noffset / nodemean / nors
CONFIGS=original SEEDS="42 1 2" bash TFT_model/run_loyo.sh  # PVT-TFT

# 2) 面板基线：仅县固定效应与线性面板估计器（第5.2、5.7节，写出 panel.json）
$P -I TFT_model/panel_report.py

# 3) 汇总：依次写 paper_figures/{loyo_summary,single_fold_2022,tft_main}.json
LOYO_SEEDS=0 $P -I TFT_model/loyo_summary.py
LOYO_SEEDS=0 $P -I TFT_model/paper_report_tft.py
LOYO_SEEDS=0 $P -I TFT_model/single_fold_report.py

# 4) 测试时遥感干预（第5.6节）
$P -I TFT_model/rs_intervention_improved.py
```

自助法（5000 次重抽样）从 `np.random.default_rng(0)` 取数，随机数流随比较行数移动：表5-1、表5-2 与消融区间的数值只在上面 1—3 步按原样跑完后才与论文一致，在 `paper_report_tft.py` 里增删任何对照行都会改变全部区间。

全样本置换检验（第5.8节）另用两步：`TFT_model/rs_signal_extract.py --output <npz>` 提取植被指数特征，`TFT_model/rs_signal_test.py --features <npz> --output <json>` 给出置换结果，`TFT_model/rs_info_test.py` 把结果整理成 `paper_figures/rs_info_test.json`。

## 引用

参考论文见 `Reference/`。研究细节见 `paper.md`。
