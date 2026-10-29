# UNet-ConvLSTM, CropNet adaptation

PhenoYieldNet 论文报告了 UNet-ConvLSTM 基线，但官方仓库没有提供代码。本目录按其引用来源（Kamangir et al., *Large-scale spatio-temporal yield estimation via deep learning using satellite and management data fusion in vineyards*, <https://doi.org/10.1016/j.compag.2023.108439>）的结构思路独立实现。它不是作者基线的官方复现，结果统一标记为 `UNet-ConvLSTM, CropNet adaptation`。

## 协议

- 训练：2017–2021 每个县年都是独立样本，各自预测当年单产，不跨年传递隐藏状态。
- 测试：2022 县年，只评估 `checkpoint-final.pth`。
- 不使用长期气象、其他年份的隐藏状态或历史产量。
- 天气和标签的标准化统计量只来自训练年份。训练轮数固定，不设验证集，也不按指标挑选 checkpoint。

## 数据流

```text
每月每网格 Sentinel-2（惰性读取并缩放到 image_size）
  -> 共享 U-Net encoder（4 级收缩路径；多尺度特征池化到最深层后经 1x1 卷积融合）
同月每网格 28 天 x 11 维天气 -> 1D 卷积天气编码器 -> 广播到特征图
视觉 + 天气 1x1 卷积融合 -> 6 个月的特征图序列 -> ConvLSTM（逐网格跨月）
  -> 末月隐状态空间平均 -> 有效网格 masked pooling -> 县级当年单产
```

U-Net 只用编码器特征，不生成分割图。编码器只处理有效网格，padding 网格不参与计算。

## 运行

```bash
./UNet-ConvLSTM/run_train.sh
```

脚本默认 `EPOCHS=50`、`GPU 3`，输出根为 `UNet-ConvLSTM/output_dir`，训练后自动执行 2022 最终测试。可覆盖 `EPOCHS`、`RUN_ID`、`OUTPUT_ROOT`、`NUM_WORKERS`、`COUNTY_BATCH_SIZE`、`IMAGE_SIZE`、`GRID_CHUNK`、`WARMUP_EPOCHS` 和 `DRY_RUN`；例如 `DRY_RUN=1 EPOCHS=1 RUN_ID=smoke ./UNet-ConvLSTM/run_train.sh`。

超参数写在 `config.py`，都是本项目自行设定的。
