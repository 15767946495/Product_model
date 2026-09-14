# TFT 五州 AG 遥感数据管线设计

## 目标

为 TFT 建立 CropNet 五州、AG-only 遥感数据管线，完成气象、土壤、产量与 AG 图像的样本身份对齐和加载能力。

本阶段只实现数据准备与加载，不修改 TFT 模型结构，不设计遥感与气象的融合方式，不训练或推理模型。

## 实验协议

### 区域

只保留 CropNet 数据集论文实验使用的五州：

```text
Illinois
Iowa
Louisiana
Mississippi
New York
```

所有生成的数据清单和运行时审计都基于解析后的州值验证五州范围。

### 年份划分

固定划分为：

- 训练集：2017--2020；
- 验证集：2021，用于早停和模型选择；
- 测试集：2022，仅用于最终评估。

2022 标签不能参与训练、早停、超参数选择或数据处理策略选择。

### 气象

短期气象保持当前统一协议：

- 月份：4--9 月；
- 每月日期：1--28 日；
- 时间步：168；
- 动态变量：当前 TFT 使用的网格级 WRF-HRRR 特征；
- 标准化统计量只能使用 2017--2020 训练样本计算。

### 遥感

只使用 Agriculture Imagery（AG），不使用 NDVI 或 Vegetation 文件。

每个县年样本引用两个季度文件：

```text
Agriculture_<state>_<year>-04-01_<year>-06-30.h5
Agriculture_<state>_<year>-07-01_<year>-09-30.h5
```

TFT 的 AG Dataset 每月保留 1 日和 15 日两幅影像，形成十二个时间点。MMST-ViT 官方 loader 只保留每月第一幅的六时间点选择仍作为参考，但不作为 TFT AG Dataset 的输出协议。

```text
04-01
04-15
05-01
05-15
06-01
06-15
07-01
07-15
08-01
08-15
09-01
09-15
```

每个样本的 AG 张量接口为：

```text
(12, G, 3, 224, 224)
```

其中 `G` 是县内 9 km 网格数量。本阶段不要求 AG 网格顺序与气象网格建立融合映射，但必须保存 FIPS、年份、日期、网格数量，并验证十二个时间点的 AG 网格数量一致。

## 数据加载接口

### 五州年份划分

在 `TFT_model/data.py` 中增加纯数据辅助函数，将输入样本按固定年份划分为：

```python
{
    "train": [...],
    "val": [...],
    "test": [...],
}
```

函数必须拒绝协议外州值和协议外年份，不能把未知年份静默放入任意 split。

### AG 路径构造

根据样本的 FIPS、州和年份构造两个 AG 季度文件路径。路径根目录可配置，默认指向：

```text
/data/raid0/hqx/Product_model_runtime/DataSrc/mmst_vit/download/
```

路径构造不包含 NDVI/Vegetation 回退逻辑。

### AG Dataset

新增独立 Dataset，职责仅为读取 AG 图像：

```python
class AgricultureImageDataset(Dataset):
    ...
```

构造参数至少包括：

- 样本记录；
- AG 数据根目录；
- `train` 图像处理模式；
- 随机种子，默认 0；
- 图像尺寸，默认 224。

每条输出至少包含：

- `ag_images`：`(12, G, 3, 224, 224)`；
- `ag_dates`：十二个 `(month, day)` 时间点；
- `FIPS`；
- `Year`；
- `State`；
- `County`；
- `grid_count`。

AG Dataset 不读取气象、土壤或产量，不改变当前 `GridTimeSeriesDataset` 的返回接口。

### 图像处理

图像处理与 MMST-ViT 官方 `DataWrapper` 保持一致。

训练模式：

```text
RandomResizedCrop(224)
RandomHorizontalFlip()
RandomApply(ColorJitter(0.8, 0.8, 0.8, 0.2), p=0.8)
RandomGrayscale(p=0.2)
GaussianBlur(kernel_size=9)
Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
```

验证和测试模式：

```text
CenterCrop(224)
Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])
```

随机种子默认使用 0。数据层不得修改全局随机状态；需要可复现增强时，应使用 worker 或样本级种子控制。

## 样本有效性

数据清单生成阶段必须验证：

- 两个 AG 季度文件都存在；
- HDF5 中存在目标 FIPS group；
- 目标 group 可以读取；
- 十二个目标日期都存在；
- 每个日期都有 `data` 数据集；
- 图像数据维度为 `(G, H, W, 3)`；
- 十二个时间点的 `G` 一致；
- 图像通道数为 3；
- 目标气象样本、土壤记录和产量标签存在。

无效样本不能在 Dataset 的训练迭代中静默跳过。应在清单构建阶段记录到独立的无效样本报告，包含 FIPS、年份、文件路径和错误原因。

## 运行时目录

五州数据使用独立目录，不覆盖现有八州历史 bundle：

```text
/data/raid0/hqx/Product_model_runtime/cropnet-five-state/
├── dataset/
├── manifests/
├── audit/
└── output/
```

至少生成：

```text
manifests/train.jsonl
manifests/val.jsonl
manifests/test.jsonl
audit/ag_integrity.json
audit/protocol.json
```

是否生成新的气象 JSONL 和网格缓存由实施阶段依据现有生成器最小适配决定，但五州运行产物不得覆盖当前 `task8-active`。

## 测试

单元测试使用临时 HDF5 fixture，不读取大型真实文件，覆盖：

- 五州允许集合；
- 2017--2020、2021、2022 的固定划分；
- 协议外州和年份拒绝；
- AG-only 两季度路径；
- 不读取 NDVI/Vegetation；
- 十二个目标日期选择；
- `(12, G, 3, 224, 224)` 输出；
- 训练与验证图像变换；
- 随机种子 0 的增强复现；
- AG 文件缺失；
- FIPS group 缺失；
- 日期或 `data` 缺失；
- 网格数量或通道数量不一致。

真实数据审计扫描五州 2017--2022 的 AG 文件，输出可用和不可用县年样本数量及原因，不训练模型。

## 验收标准

- `TFT_model/data.py` 可以独立构造五州固定年份 split；
- AG Dataset 可以读取临时和真实 HDF5 样本；
- AG 输出包含十二个时间点且形状正确；
- 图像处理与 MMST-ViT 官方配置一致；
- 五州真实 AG 完整性审计完成；
- 训练、验证、测试清单只包含有效 AG、气象、土壤和标签样本；
- 当前 `TFTEncoderForYieldPrediction` 和训练脚本未接入 AG 输入；
- 没有启动训练或推理；
- 没有删除或修改原始 Sentinel 文件；
- 当前八州 `task8-active` 保留不变。

## 非目标

- 不设计 AG 与每日气象的融合方法；
- 不新增视觉编码器；
- 不修改 TFT forward 签名；
- 不计算或缓存遥感特征；
- 不使用 NDVI/Vegetation；
- 不修改训练优化器、损失或随机种子配置；
- 不重新训练 MMST-ViT、TFT 或基线模型。
