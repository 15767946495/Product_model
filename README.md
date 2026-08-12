# CropNet Model — 县级玉米单产预测

基于**网格级空间注意力 + 时间融合变换器（TFT）**的县级玉米单产预测研究。

- **研究对象**：美国玉米带（Corn Belt）县级玉米单产
- **数据**：USDA 县级单产 + WRF-HRRR 网格级气象数据
- **方法**：时间融合变换器（Temporal Fusion Transformer）结合网格级空间注意力、变量选择网络、季节滚动预报
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
> 下载地址：*（待补充 — 数据已上传至 网盘/HuggingFace 后在此填写链接）*

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
# 训练主模型（TFT）
bash TFT_model/train_crucial.sh      # 或 train_mse.sh

# 推理
python TFT_model/infer.py

# 基线模型训练（以 CNNRNN 为例）
bash BaseLine_Model/CNNRNN/train.sh

# 消融 / 网格搜索
python ablation/ablation.py
python grid_search/grid_search.py
```

## 引用

参考论文见 `Reference/`。研究细节见 `paper.md`。
