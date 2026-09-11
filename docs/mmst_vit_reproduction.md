# MMST-ViT CropNet 复现记录

## 官方预处理核对

固定源码版本：`fudong03/MMST-ViT@615666c8d9fcd704acb662c13065703cbf2eab70`。

- 气象：`dataset/hrrr_loader.py` 选择 9 个通道，短期读取 4--9 月每月前 28 天的 Daily 记录，长期读取月度记录；没有执行标准化、`StandardScaler` 或固定 mean/std，只转为 `float32`。
- `dataset/data_wrapper.py` 中的 `ScalarNorm` 使用 `sklearn.preprocessing.StandardScaler`，但官方微调入口没有调用它。因此当前复现的气象归一化必须记录为 `none`，不能擅自套用当前 TFT 的训练集统计量，否则不再是官方实现复现。
- 遥感：微调入口通过 `DataWrapper` 对图像使用 `Normalize([0.466, 0.471, 0.380], [0.195, 0.194, 0.192])`；训练还使用 SimCLR 增强，验证使用 CenterCrop。官方另一个 `sentinel_wrapper.py` 把 `ToTensor` 后的归一化注释掉，实际微调入口应以 `data_wrapper.py` 为准并在报告中注明。
- USDA：官方 `USDA_Dataset` 对生产量和单产取自然对数，评估时对模型输出和标签取指数；当前适配使用玉米 BU/ACRE 单产。

## 当前协议

- 训练：2017--2020
- 验证/早停：2021
- 最终测试：2022，一次性评估
- 气象归一化：不做归一化，严格保留官方原始数值流程
- Sentinel 归一化：官方固定三通道均值/标准差

## 运行顺序

```bash
python -m mmst_vit.manifest \
  --usda-dir /data/raid0/hqx/Product_model_runtime/DataSrc/cropnet_dataset/data/usda_corn \
  --weather-dir /data/raid0/hqx/Product_model_runtime/DataSrc/cropnet_dataset/data/weather \
  --output /data/raid0/hqx/Product_model_runtime/DataSrc/mmst_vit/manifests/valid.jsonl
```

随后使用有效 manifest 的 FIPS 调用 Sentinel 清单工具。必须先完成 dry-run，确认文件数、总字节数和 FIPS 数，再启动下载。官方 HF 文件是按州/季度打包的 HDF5，不能按单个县推算下载文件大小。

官方源码运行前必须生成三份官方 JSON 数组：训练、验证、测试。JSON 中的 `HRRR`、`USDA`、`sentinel` 路径必须相对于 `--root_dir` 可读；三个 Dataset 的样本顺序必须完全一致。当前 `mmst_vit.config.official_sample_record` 生成该 JSON 结构。

## 已知官方入口问题

官方 `main_finetune_mmst_vit.py` 的验证 DataLoader 使用了训练 sampler，`--eval` 分支也只评估训练 loader，并且入口没有独立的 2022 测试参数。复现实现必须在不修改 MMST-ViT 模型结构的前提下增加入口适配，显式构造 train/val/test loader，按 2021 指标保存最佳 checkpoint，最后只对 2022 输出 RMSE、R2、Pearson Corr 和 FIPS 级预测。

当前环境没有安装 `pytest`，验证可先使用 `python -m compileall mmst_vit` 和 `python -m unittest`；安装官方依赖后再执行 one-epoch/one-validation smoke test。
