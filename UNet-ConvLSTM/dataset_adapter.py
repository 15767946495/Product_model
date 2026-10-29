"""UNet-ConvLSTM 唯一数据适配边界：统一 CropNetDataset -> 县年 batch。

每个县年是独立样本：训练 2017--2021，测试 2022；不使用长期气象。
遥感像素通过 HDF5 引用惰性读取并缩放到 image_size；天气标准化统计量只来自训练年份。
batch：images (B,6,G,3,S,S)、image_mask (B,6,G)、weather (B,6,G,28,11)、grid_mask (B,G)、
labels (B,)、sample_id、fips、year。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import TEST_YEARS, TRAIN_YEARS  # noqa: E402
from train_dataset.collate import collate_county_year, monthly_weather_view  # noqa: E402
from train_dataset.dataset import DATA_ROOT, CropNetDataset  # noqa: E402

EPS = 1e-6


def collate(items) -> dict:
    base = collate_county_year(items)
    if "images" not in base:
        raise ValueError("UNet-ConvLSTM requires images; build the dataset with load_images=True")
    return {
        "images": base["images"], "image_mask": base["image_mask"],
        "weather": monthly_weather_view(base["weather"]), "grid_mask": base["grid_mask"],
        "labels": base["labels"], "sample_id": base["sample_id"], "fips": base["fips"], "year": base["year"],
    }


def build_dataset(jsonl_path, data_root, years, *, image_size: int, load_images: bool = True) -> CropNetDataset:
    return CropNetDataset(Path(jsonl_path), tuple(years), load_images=load_images,
                          data_root=Path(data_root), image_size=image_size)


def build_loader(dataset, batch_size: int, shuffle: bool, *, num_workers: int = 0, generator=None):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                                       num_workers=num_workers, generator=generator,
                                       persistent_workers=num_workers > 0)


def fit_stats(train_dataset: CropNetDataset) -> dict:
    """只用训练年份：天气按 (网格, 天) 求每变量均值/方差；标签取 log 后的均值/方差。"""
    if {s["year"] for s in train_dataset.samples} - set(TRAIN_YEARS):
        raise ValueError("stats must be fitted on training years only")
    total = np.zeros(11, dtype=np.float64)
    sq = np.zeros(11, dtype=np.float64)
    count = 0
    for s in train_dataset.samples:
        w = s["weather"].reshape(-1, s["weather"].shape[-1]).astype(np.float64)
        total += w.sum(axis=0)
        sq += (w ** 2).sum(axis=0)
        count += w.shape[0]
    mean = total / count
    std = np.sqrt(np.maximum(sq / count - mean ** 2, 0.0))
    log_y = np.log([s["label"] for s in train_dataset.samples])
    return {
        "weather_mean": torch.tensor(mean, dtype=torch.float32),
        "weather_std": torch.tensor(std, dtype=torch.float32).clamp_min(EPS),
        "y_mean": torch.tensor(float(log_y.mean())), "y_std": torch.tensor(max(float(log_y.std()), EPS)),
    }


def model_inputs(batch: dict, stats: dict, device) -> dict:
    weather = (batch["weather"] - stats["weather_mean"]) / stats["weather_std"]
    weather = weather * batch["grid_mask"][:, None, :, None, None].to(weather.dtype)  # padding 保持 0
    return {
        "images": batch["images"].to(device, non_blocking=True),
        "weather": weather.to(device, non_blocking=True),
        "grid_mask": batch["grid_mask"].to(device, non_blocking=True),
        "image_mask": batch["image_mask"].to(device, non_blocking=True),
    }


def target(batch: dict, stats: dict) -> torch.Tensor:
    return (torch.log(batch["labels"]) - stats["y_mean"]) / stats["y_std"]


def stats_to_json(stats: dict) -> dict:
    return {k: v.tolist() for k, v in stats.items()}


def stats_from_json(payload: dict) -> dict:
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in payload.items()}


__all__ = ["TRAIN_YEARS", "TEST_YEARS", "DATA_ROOT", "build_dataset", "build_loader", "collate", "fit_stats",
           "model_inputs", "target", "stats_to_json", "stats_from_json"]
