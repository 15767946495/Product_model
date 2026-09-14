# Task 2 报告：AG-only HDF5 数据加载

## 状态

已实现并通过验证；本次修复针对审查发现的 AG-only 路径边界、实际取样 RNG 隔离、HDF5 结构错误、日期接口复用和测试覆盖问题进行了补强。

## 实现内容

- `TFT_model/data.py`
  - 新增 `build_ag_paths(sample, ag_root)`，只生成按时间排序的两个 `data/AG/.../Agriculture_*.h5` 季度路径；样本中存在 sentinel/path 字段时递归检查，并拒绝任何非 AG 路径。
  - 新增 `select_ag_dates(group)`，严格校验并返回 12 个日期：`04-01`、`04-15`、`05-01`、`05-15`、`06-01`、`06-15`、`07-01`、`07-15`、`08-01`、`08-15`、`09-01`、`09-15`。
  - 新增 `AgricultureImageDataset`，只读取目标 FIPS、目标季度日期和 `data` 数据集，输出 `ag_images` 形状 `(12, G, 3, 224, 224)` 及 `ag_dates`、`FIPS`、`Year`、`State`、`County`、`grid_count` 元数据。
  - 严格拒绝任意非 AG modality、非 `Agriculture_*.h5` 路径、缺文件、缺 FIPS、缺日期、非 Dataset 的 `data`、错误数据形状/类型和跨时相网格数不一致；构造和 `__getitem__` 均带 sample context 抛出 `ValueError`。
  - 训练变换按 MMST-ViT 顺序使用 `RandomResizedCrop`、`RandomHorizontalFlip`、`RandomApply(ColorJitter)`、`RandomGrayscale`、`GaussianBlur`、官方 Normalize；验证/测试使用 `CenterCrop` 和官方 Normalize。
  - 使用 `torch.random.fork_rng` 覆盖 CPU 和所有可用 CUDA 设备，并用基于 `seed`、样本索引、时相索引的局部种子；Dataset 构造和实际取样不污染外部全局 Torch RNG 状态。

- `tests/test_tft_ag_data.py`
  - 新增临时 HDF5 fixture，覆盖两个季度、单 FIPS、12 个日期、两个网格。
  - 覆盖成功加载、严格 AG 路径构造、严格日期选择、任意非 AG modality 拒绝、缺文件/FIPS/日期/data、非 Dataset `data`、shape/dtype 错误、网格数不一致、实际 `select_ag_dates()` 调用、seed=0 跨实例确定性，以及实际取样前后 CPU/CUDA RNG 不变行为。

## TDD 记录

先添加 fixture 测试并运行，按预期因 `AgricultureImageDataset` 等接口不存在而在收集阶段失败；实现后再次运行通过。

## 验证结果

命令：

```text
/root/miniconda3/envs/hqx/bin/python -m pytest tests/test_tft_ag_data.py -q
41 passed, 1 warning

/root/miniconda3/envs/hqx/bin/python -m py_compile TFT_model/data.py tests/test_tft_ag_data.py
通过（无输出）
```

pytest 的一个既有环境告警来自 `torch.cuda`：`pynvml package is deprecated`；不是本任务代码产生的失败。

## 范围确认

- 未修改 `TFT_model/models.py`、训练入口或运行数据。
- 未运行模型训练或推理。
- 未访问、修改或审计真实 Sentinel 文件；所有 HDF5 测试数据均为临时 fixture。
- 工作区中原有的 `_txt` 删除、`BaseLine_Model/output` 和 `DataSrc` 未纳入本次提交。

## 取舍

- 当前实现按简报要求假定样本州属于五州协议，并通过州全名映射 USPS 缩写；不提供其他州或 NDVI/Vegetation fallback。
- Dataset 构造阶段会完整审计每个样本的两个 HDF5 文件，保证错误尽早暴露；`__getitem__` 仍保留同等防御性校验。代价是构造时产生一次结构扫描。
