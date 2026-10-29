"""PhenoYieldNet-unified 固定配置（规范 10.3）。

数值取自官方仓库 roroyo/PhenoYieldNet@5a14ed7 的 train_da.py 默认值与 scripts/run_finetune.sh；
README（Stage 2 写 60 epochs）与 run_finetune.sh（EPOCHS=40）不一致，这里显式固定为脚本值 40。
"""

import os

MODEL_NAME = "phenoyieldnet"
STAGE1_NAME = "phenoyieldnet_stage1"
ADAPTATION = "PhenoYieldNet-unified (PhenoYieldNet-SC, corn only, unified CropNet)"

def _years(env, default):
    """环境变量给出逗号分隔年份时覆盖默认折；未设置时保持默认（2017—2020 预训练、2021 微调、2022 测试）。"""
    raw = os.environ.get(env, "").strip()
    return tuple(int(x) for x in raw.split(",") if x.strip()) if raw else default


PRETRAIN_YEARS = _years("PHENO_PRETRAIN_YEARS", (2017, 2018, 2019, 2020))
FINETUNE_YEARS = _years("PHENO_FINETUNE_YEARS", (2021,))
TEST_YEARS = _years("PHENO_TEST_YEARS", (2022,))

BACKBONE = "vit_base_patch8_128_channel3_frame1_tpatch1"
EMBED_DIM = 768
IMG_SIZE = 128            # 骨干输入；HDF5 原图 224 经随机裁剪 / 中心裁剪到 128（与上游一致）
SOURCE_IMAGE_SIZE = 224
PIXEL_SCALE = 255.0       # 上游直接把 uint8 像素转 float（0--255，不除 255）；统一 Dataset 输出 [0,1]，乘回
NUM_GRID = 322            # 上游默认 64；统一数据单县最多 321 网格（+1 cls）
CORN = 1                  # 上游 crop_query_bank / mlp_heads 中 corn 的索引
NUM_CROPS = 4             # 保留上游四作物参数形状，只使用 corn
MONTHS = 6                # 统一数据只有 4--9 月；上游 corn 窗口为 4--12 月，此处取交集 4--9 月

SPECTRALGPT_URL = "https://zenodo.org/records/8412455/files/SpectralGPT+.pth?download=1"
SPECTRALGPT_PATH = "/data/raid0/hqx/pretrained/SpectralGPT+.pth"
SPECTRALGPT_SHA256 = "981730de133f15090978d60a9eded6d04d970174834b4e7e16b93e1d39b8a437"
SPECTRALGPT_MD5 = "aeaa08388f0937e3f07ecbd7757f20dc"  # 与 Zenodo API 记录一致
SPECTRALGPT_LICENSE = "CC-BY-4.0"

STAGE1 = {"epochs": 200, "blr": 1e-3, "batch_size": 64, "warmup_epochs": 40, "weight_decay": 0.05,
          "drop_path": 0.2, "frame_drop_rate": 0.2}
STAGE2 = {"epochs": 40, "blr": 3e-4, "batch_size": 16, "warmup_epochs": 5, "weight_decay": 0.05,
          "drop_path": 0.2, "frozen_fm": False}

INPUTS = ("sentinel2_monthly", "weather_daily_grid_9var")

# 原 MMST / PhenoYieldNet 使用的 9 个 HRRR 变量（不含 U/V 风分量），按名称从 11 维选取。
CONTEXT_FEATURES = (
    "avg_temperature_k", "max_temperature_k", "min_temperature_k",
    "precipitation_kg_m2", "relative_humidity_pct", "wind_gust_m_s",
    "wind_speed_m_s", "downward_shortwave_radiation_w_m2", "vapor_pressure_deficit_kpa",
)
