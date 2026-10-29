"""PhenoYieldNet-unified 唯一数据适配边界：统一 CropNetDataset -> 上游 PhenoYieldNet 输入。

batch（规范 10.3）：
    images (B,6,G,3,224,224) 像素尺度 0--255（与上游直接 float(uint8) 一致）
    image_mask (B,6,G)、time_mask (B,6)、grid_mask (B,G)
    weather (B,6,G,28,11)、weather_context (B,6,G,28,9)（按变量名选 9 维）
    labels (B,)、sample_id、fips、year
不使用长期气象。上游模型没有网格 mask，因此训练 / 测试 DataLoader 固定每步一个县年（与上游 batch_size=1
的县级 DataLoader 一致），不产生 padding 网格；多县 batch 只用于 smoke 检查。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (  # noqa: E402
    CONTEXT_FEATURES, FINETUNE_YEARS, NUM_GRID, PIXEL_SCALE, PRETRAIN_YEARS, SOURCE_IMAGE_SIZE, TEST_YEARS,
)
from train_dataset.collate import collate_county_year, monthly_weather_view  # noqa: E402
from train_dataset.dataset import DATA_ROOT, WEATHER_FEATURES, CropNetDataset  # noqa: E402


def context_indices(feature_names=WEATHER_FEATURES) -> list[int]:
    names = list(feature_names)
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate weather feature {duplicates}")
    missing = [n for n in CONTEXT_FEATURES if n not in names]
    if missing:
        raise ValueError(f"missing context feature {missing}")
    return [names.index(n) for n in CONTEXT_FEATURES]


_CONTEXT_INDEX = torch.tensor(context_indices())


def collate(items) -> dict:
    base = collate_county_year(items)
    if "images" not in base:
        raise ValueError("PhenoYieldNet requires images; build the dataset with load_images=True")
    grids = base["grid_mask"].shape[1]
    if grids + 1 > NUM_GRID:  # +1 为空间 cls token
        raise ValueError(f"{base['sample_id']}: {grids} grids exceeds PhenoYieldNet num_grid {NUM_GRID} - 1")
    weather = monthly_weather_view(base["weather"])
    return {
        "images": base["images"] * PIXEL_SCALE,
        "image_mask": base["image_mask"],
        "time_mask": base["image_mask"].any(dim=2),
        "grid_mask": base["grid_mask"],
        "weather": weather,
        "weather_context": weather[..., _CONTEXT_INDEX],
        "labels": base["labels"],
        "sample_id": base["sample_id"], "fips": base["fips"], "year": base["year"],
    }


def build_dataset(jsonl_path, data_root, years) -> CropNetDataset:
    return CropNetDataset(Path(jsonl_path), tuple(years), load_images=True, data_root=Path(data_root),
                          image_size=SOURCE_IMAGE_SIZE)


def build_loader(dataset, shuffle: bool, *, num_workers: int = 0, sampler=None, generator=None):
    """每步一个县年（上游协议），因此不存在 padding 网格。"""
    return torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=shuffle if sampler is None else False, sampler=sampler,
        collate_fn=collate, num_workers=num_workers, generator=generator, persistent_workers=num_workers > 0)


def require_unpadded(batch) -> None:
    if not batch["grid_mask"].all():
        raise ValueError(f"{batch['sample_id']}: PhenoYieldNet has no grid mask; use one county-year per step")


__all__ = ["PRETRAIN_YEARS", "FINETUNE_YEARS", "TEST_YEARS", "DATA_ROOT", "build_dataset", "build_loader",
           "collate", "context_indices", "require_unpadded"]
