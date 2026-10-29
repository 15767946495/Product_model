"""UNet-ConvLSTM, CropNet adaptation 的固定配置（规范 10.5）。

PhenoYieldNet 论文报告了该基线但未公开代码；结构参考 Kamangir et al., 2024
(https://doi.org/10.1016/j.compag.2023.108439)。以下参数为本项目显式定义，不声称复现原作者设置。
"""

import os

MODEL_NAME = "unet_convlstm"
ADAPTATION = "UNet-ConvLSTM, CropNet adaptation"


def _years(env, default):
    """环境变量给出逗号分隔年份时覆盖默认折；未设置时保持默认（2022 主协议）。"""
    raw = os.environ.get(env, "").strip()
    return tuple(int(x) for x in raw.split(",") if x.strip()) if raw else default


TRAIN_YEARS = _years("UNET_TRAIN_YEARS", (2017, 2018, 2019, 2020, 2021))
TEST_YEARS = _years("UNET_TEST_YEARS", (2022,))

DEFAULTS = {
    "image_size": 128,          # HDF5 惰性读取时缩放到 128x128（U-Net 4 级编码 -> 16x16）
    "base_channels": 32,        # U-Net 编码器各级通道 32/64/128/256
    "weather_dim": 64,          # 月度 28 天 x 11 维天气编码维度
    "hidden_channels": 64,      # ConvLSTM 隐状态通道
    "dropout": 0.1,
    "county_batch_size": 2,     # 每个优化步的县年数
    "grid_chunk": 128,          # 每次 U-Net 前向的 (网格 x 月) 图像数上限，只影响显存
    "epochs": 30,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "warmup_epochs": 2,
}

INPUTS = ("sentinel2_monthly", "weather_daily_grid")
