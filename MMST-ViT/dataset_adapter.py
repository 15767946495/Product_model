"""MMST-ViT 唯一数据适配边界：统一 CropNetDataset -> MMST batch。

只消费 train_dataset.CropNetDataset / collate_county_year，不读取旧 manifest、不扫描源数据。
batch 字段：
    images (B,6,G,3,S,S) / image_mask (B,6,G)        仅 load_images=True
    weather (B,6,G,28,11)、weather_context (B,6,G,28,9)
    long_term (B,4,12,9) 县内有效网格平均；2017--2020 为 None
    grid_mask (B,G)、coords (B,G,2)、labels (B,)、sample_id、fips、year
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from train_dataset.collate import collate_county_year, monthly_weather_view  # noqa: E402
from train_dataset.dataset import (  # noqa: E402
    DATA_ROOT, FINETUNE_YEARS, PRETRAIN_YEARS, TEST_YEARS, WEATHER_FEATURES, CropNetDataset,
)

MMST_YEARS_PRETRAIN = PRETRAIN_YEARS
MMST_YEARS_FINETUNE = FINETUNE_YEARS
MMST_YEARS_TEST = TEST_YEARS
MMST_MAX_GRIDS = 80
DEFAULT_DATASET_JSONL = _PROJECT_ROOT / "train_dataset" / "dataset.jsonl"

# 原 MMST-ViT 使用的 9 个 HRRR 变量（排除 U/V 风分量），按名称选择。
MMST_CONTEXT_FEATURES = (
    "avg_temperature_k", "max_temperature_k", "min_temperature_k",
    "precipitation_kg_m2", "relative_humidity_pct", "wind_gust_m_s",
    "wind_speed_m_s", "downward_shortwave_radiation_w_m2", "vapor_pressure_deficit_kpa",
)


def context_indices(feature_names: Sequence[str]) -> list[int]:
    names = list(feature_names)
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate weather feature {duplicates}")
    missing = [n for n in MMST_CONTEXT_FEATURES if n not in names]
    if missing:
        raise ValueError(f"missing context feature {missing}")
    return [names.index(n) for n in MMST_CONTEXT_FEATURES]


def select_context_features(values, feature_names: Sequence[str]):
    """最后一维按变量名从 len(feature_names) 维选出 MMST_CONTEXT_FEATURES 的 9 维。"""
    index = context_indices(feature_names)
    if values.shape[-1] != len(feature_names):
        raise ValueError(f"last dimension {values.shape[-1]} != {len(feature_names)} weather features")
    if isinstance(values, torch.Tensor):
        return values[..., torch.tensor(index, device=values.device)]
    return np.asarray(values)[..., index]


def _pad_dim(tensor: torch.Tensor, dim: int, size: int) -> torch.Tensor:
    extra = size - tensor.shape[dim]
    if extra == 0:
        return tensor
    shape = list(tensor.shape)
    shape[dim] = extra
    return torch.cat([tensor, tensor.new_zeros(shape)], dim=dim)


# 字段 -> 网格维；右侧 zero padding，mask 补 False。
_GRID_DIMS = {"images": 2, "image_mask": 2, "weather": 2, "weather_context": 2, "coords": 1, "grid_mask": 1}


def pad_mmst_grids(batch: dict, max_grids: int = MMST_MAX_GRIDS) -> dict:
    """把网格维补齐到 max_grids；超限直接报错，不截断、不丢县。"""
    counts = batch["grid_mask"].sum(dim=1).tolist()
    G = batch["grid_mask"].shape[1]
    if G > max_grids:
        over = [f"{sid} ({int(n)} grids)" for sid, n in zip(batch["sample_id"], counts) if n > max_grids]
        raise ValueError(f"{', '.join(over) or batch['sample_id']}: {G} grids exceeds MMST limit {max_grids}")
    out = dict(batch)
    for key, dim in _GRID_DIMS.items():
        if key in out:
            out[key] = _pad_dim(out[key], dim, max_grids)
    return out


def collate_mmst(items: Sequence[dict], max_grids: int | None = None) -> dict:
    """CropNetDataset items -> MMST batch；max_grids 给定时补齐网格维。"""
    has_long = [item["long_term"] is not None for item in items]
    if any(has_long) and not all(has_long):
        mixed = [item["sample_id"] for item in items]
        raise ValueError(f"batch mixes samples with and without long-term weather: {mixed}")
    base = collate_county_year(items)
    weather = monthly_weather_view(base["weather"])
    grid_mask = base["grid_mask"]
    long_term = None
    if all(has_long):
        # (B,G,4,12,11) -> 县内有效网格平均 (B,4,12,9)，与原 MMST 县级长期气象输入一致
        weights = base["long_term_mask"].to(base["long_term"].dtype).view(*grid_mask.shape, 1, 1, 1)
        pooled = (base["long_term"] * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        long_term = select_context_features(pooled, WEATHER_FEATURES)
    batch = {
        "weather": weather,
        "weather_context": select_context_features(weather, WEATHER_FEATURES),
        "long_term": long_term,
        "grid_mask": grid_mask,
        "coords": base["coords"],
        "labels": base["labels"],
        "sample_id": base["sample_id"],
        "fips": base["fips"],
        "year": base["year"],
    }
    if "images" in base:
        batch["images"], batch["image_mask"] = base["images"], base["image_mask"]
    if max_grids is not None:
        batch = pad_mmst_grids(batch, max_grids)
    return batch


def build_mmst_dataset(jsonl_path, data_root, years, *, load_images: bool = True) -> CropNetDataset:
    return CropNetDataset(Path(jsonl_path), tuple(years), load_images=load_images, data_root=Path(data_root))


class _MMSTCollate:
    """可 pickle 的 collate（DataLoader 多进程需要）。"""

    def __init__(self, max_grids):
        self.max_grids = max_grids

    def __call__(self, items):
        return collate_mmst(items, max_grids=self.max_grids)


def build_mmst_dataloader(jsonl_path, data_root, years, batch_size, shuffle, load_images=True, *,
                          num_workers: int = 0, pin_memory: bool = False, sampler=None,
                          drop_last: bool = False, max_grids: int = MMST_MAX_GRIDS, dataset=None):
    dataset = dataset or build_mmst_dataset(jsonl_path, data_root, years, load_images=load_images)
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle if sampler is None else False, sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last,
        collate_fn=_MMSTCollate(max_grids), persistent_workers=num_workers > 0,
    )


def make_batch(path, data_root=DATA_ROOT, years=MMST_YEARS_FINETUNE, *, load_images: bool = False,
               max_grids: int | None = None) -> dict:
    """把所选年份全部样本合成一个 batch（测试 / smoke 用）。"""
    dataset = build_mmst_dataset(path, data_root, years, load_images=load_images)
    return collate_mmst([dataset[i] for i in range(len(dataset))], max_grids=max_grids)
