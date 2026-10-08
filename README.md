# CropNet Model — 县级玉米单产预测

基于**MMST 多模态空间前端 + 时间融合变换器（TFT）**的县级玉米单产预测研究。

- **研究对象**：美国玉米带（Corn Belt）县级玉米单产
- **数据**：官方 MMST Sentinel-2、WRF-HRRR 与 USDA Corn 数据，外加 7 维县级土壤属性
- **方法**：完整 PVTSimCLR 多模态前端、官方 SpatialTransformer、TFT temporal backend 与概率预测
- **配套论文**：`paper.md`

## 目录结构

```
cropnet_model/
├── TFT_model/            # 主模型代码（TFT + 空间注意力，训练/推理/注意力分析）
├── BaseLine_Model/       # 基线模型（CNN-RNN / ConvLSTM / GNNRNN / DeepCropNet）
├── ablation/             # 消融实验
├── grid_search/          # 超参数网格搜索
├── _txt/                 # 论文与实验说明文本
├── train_dataset/        # 数据集构建脚本（JSONL / 网格缓存 / 土壤映射）
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

## 数据说明（重要）

原始数据（约 31GB）**未提交到本仓库**，包括：

| 路径 | 内容 | 大小 |
|------|------|------|
| `DataSrc/cropnet_dataset/data/` | USDA 县级数据 + WRF-HRRR 网格气象（2017–2022） | ~28GB |
| `DataSrc/soil_dataset/` | 土壤属性数据 | — |
| `train_dataset/dataset.jsonl` | 县级粒度训练集（JSONL） | ~330MB |
| `train_dataset/grid_cache.pt` | 网格级气象二进制缓存 | ~1.5GB |
| `BaseLine_Model/output/` | 基线模型输出数据 | ~930MB |

这些文件因超出 Git 与 GitHub 的存储限制（单文件 >100MB 会被 GitHub 拒绝），改为**外置存放**。

> **下载地址：[ModelScope 数据集 wbsyz1/CropDataSet](https://www.modelscope.cn/datasets/wbsyz1/CropDataSet)**

数据以两个压缩包（tar.gz，内部保留原始目录结构）提供：

| 包 | 内容 | 大小 |
|----|------|------|
| `cropnet_weather_data.tar.gz` | WRF-HRRR 网格气象（2017–2022） | 约 5.8GB（原始 28GB） |
| `cropnet_aux_data.tar.gz` | USDA 数据、土壤、train_dataset、基线输出 | 约 1.8GB |

**解压方式**（在项目根目录执行，解压后即还原 `DataSrc/` 等原始结构）：

```bash
tar -xzf cropnet_weather_data.tar.gz
tar -xzf cropnet_aux_data.tar.gz
```

SHA256 校验值见数据集仓库的 `SHA256SUMS` 文件。

### 数据再生成

如需本地重新生成 `train_dataset` 中的数据，可按顺序运行：

```bash
# 1) 生成州级土壤静态特征映射
python train_dataset/gen_state_soil_map.py

# 2) 把 WRF-HRRR 气象数据按网格级处理成二进制缓存 grid_cache.pt
python train_dataset/prepare_grid.py

# 3) 把原始数据整理成县级粒度 JSONL 训练集 dataset.jsonl
python train_dataset/prepare_jsonl.py
```

> 注意：上述脚本依赖 `DataSrc/` 下的原始数据，需先从外部下载并放置到对应目录。

## 使用

```bash
# 1) 如尚无完整 PVTSimCLR checkpoint，先运行官方 MMST 预训练入口
bash MMST-ViT/run_pretrain.sh

# 2) 使用固定协议监督训练 TFT
bash TFT_model/run_finetune.sh

# 正式推理：239 个官方 Corn 验证县
python TFT_model/infer_mmst_frontend.py

# 基线模型训练（以 CNNRNN 为例）
bash BaseLine_Model/CNNRNN/train.sh

# 消融 / 网格搜索
python ablation/ablation.py
python grid_search/grid_search.py
```

### 正式 MMST-TFT 入口与迁移边界

正式训练入口是 `TFT_model/train_mmst_frontend.py`，正式推理入口是 `TFT_model/infer_mmst_frontend.py`。推理默认严格读取 `/data/raid0/hqx/TFT_train/mmst_frontend_2021/checkpoint-final.pth`，从 embedded `config` 和 `soil_stats` 重建 `MMSTTFTYieldModel` 并执行 strict state-dict load。评估固定使用官方 `MMST-ViT/data/corn_val.json` 的 239 个县、官方 `DataWrapper(train=False)` 变换和训练县土壤 z-score；正式数据适配器只读取 6 个月、每月 28 天的短期 HRRR。模型返回的最后有效月份 `final_dist[:, 0]` 是 log-yield mean，预测与 adapter 的 `yield_log` 均以 `exp` 还原后计算 RMSE、MAE、R² 和 Corr。

正式微调默认读取 `MMST-ViT/output_dir/pvt_simclr/checkpoint-latest.pth`，并对完整 `PVTSimCLR` state dict 执行严格加载。迁移内容包括 PVT backbone、视觉投影 `proj`、9 维天气投影 `proj_context`、天气归一化和 2 层多模态多头注意力 `mm_transformer`，不是仅迁移视觉骨干。县级空间模块直接采用官方 `SpatialTransformer` 类，在监督阶段初始化和训练；MMST 自监督预训练不训练该空间模块，也不训练 TFT temporal backend。

每个县的 Sentinel-2 网格影像与同网格短期 HRRR 先经完整 PVTSimCLR 融合，再由官方 SpatialTransformer 按月份聚合，输出 6 个形状为 512 维的县级多模态空间 token。后端以 7 维土壤静态特征条件化 TFT temporal 路径，包含 LSTM、因果时间注意力、独立 mean/variance heads，并以 Beta-NLL 加 temporal KL 训练。当前正式流程不依赖独立 TFT 预训练 checkpoint 或中间转换文件；仓库中的其他历史训练、推理和数据入口均不属于正式协议。

## 引用

参考论文见 `Reference/`。研究细节见 `paper.md`。
