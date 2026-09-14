# Task 2 报告：AG-only HDF5 数据加载

## 状态

已实现并通过验证。

## 实现内容

- `TFT_model/data.py`
  - 新增 `build_ag_paths(sample, ag_root)`，严格生成按时间排序的两个 `Agriculture` 季度 HDF5 路径。
  - 新增 `select_ag_dates(group)`，固定返回 12 个日期：`04-01`、`04-15`、`05-01`、`05-15`、`06-01`、`06-15`、`07-01`、`07-15`、`08-01`、`08-15`、`09-01`、`09-15`。
  - 新增 `AgricultureImageDataset`，只读取目标 FIPS、目标季度日期和 `data` 数据集，输出 `ag_images` 形状 `(12, G, 3, 224, 224)` 及 `ag_dates`、`FIPS`、`Year`、`State`、`County`、`grid_count` 元数据。
  - 严格拒绝 NDVI/Vegetation 路径、缺文件、缺 FIPS、缺日期、缺 `data`、错误数据形状/类型和跨时相网格数不一致。
  - 训练变换按 MMST-ViT 顺序使用 `RandomResizedCrop`、`RandomHorizontalFlip`、`RandomApply(ColorJitter)`、`RandomGrayscale`、`GaussianBlur`、官方 Normalize；验证/测试使用 `CenterCrop` 和官方 Normalize。
  - 使用 `torch.random.fork_rng` 和基于 `seed`、样本索引、时相索引的局部种子，Dataset 构造和取样不污染外部全局 Torch RNG 状态。

- `tests/test_tft_ag_data.py`
  - 新增临时 HDF5 fixture，覆盖两个季度、单 FIPS、12 个日期、两个网格。
  - 覆盖成功加载、路径构造、日期选择、NDVI 拒绝、缺文件/FIPS/日期/data、网格数不一致和全局 RNG 不变行为。

## TDD 记录

先添加 fixture 测试并运行，按预期因 `AgricultureImageDataset` 等接口不存在而在收集阶段失败；实现后再次运行通过。

## 验证结果

命令：

```text
/root/miniconda3/envs/hqx/bin/python -m pytest tests/test_tft_ag_data.py -q
32 passed, 1 warning

/root/miniconda3/envs/hqx/bin/python -m py_compile TFT_model/data.py tests/test_tft_ag_data.py
通过（无输出）
```

pytest 的一个既有环境告警来自 `torch.cuda`：`pynvml package is deprecated`；不是本任务代码产生的失败。

## 范围确认

- 未修改 `TFT_model/models.py`、训练入口或运行数据。
- 未运行模型训练或推理。
- 未访问、修改或审计真实 Sentinel 文件；所有 HDF5 测试数据均为临时 fixture。
- 工作区中原有的 `_txt` 删除、`BaseLine_Model/output` 和 `DataSrc` 未纳入本次提交。

## Concerns

- 当前实现按简报要求假定样本州属于五州协议，并通过州全名映射 USPS 缩写；不提供其他州或 NDVI/Vegetation fallback。
- Dataset 构造阶段会完整审计每个样本的两个 HDF5 文件，保证错误尽早暴露，但这会在构造时产生一次结构扫描，取样时仍保留防御性校验。
