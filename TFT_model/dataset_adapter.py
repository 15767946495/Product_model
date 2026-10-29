"""TFT 的统一 CropNet Dataset 薄适配器。"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_dataset.collate import collate_county_year
from train_dataset.dataset import (
    YEARS,
    CropNetDataset,
    SOIL_FEATURES,
    WEATHER_FEATURES,
    weather_dates,
)

TRAIN_YEARS = (2017, 2018, 2019, 2020, 2021)
TEST_YEARS = (2022,)
CONSTRUCTED_FEATURES = ("CumGDD", "KDD", "CumPRCP", "CumDeficit")
GDD_BASE = 8.0
KDD_BASE = 30.0
_GSC = 0.0820
_LAMBDA = 2.45


def split_years(test_year: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """逐年留一：test_year 为测试年份，其余年份为训练年份。默认 test_year=2022 即原协议。"""
    if test_year not in YEARS:
        raise ValueError(f"test year {test_year} outside {YEARS}")
    return tuple(y for y in YEARS if y != test_year), (int(test_year),)


def dynamic_feature_names(use_constructed: bool) -> tuple[str, ...]:
    constructed = CONSTRUCTED_FEATURES if use_constructed else ("CumGDD",)
    return tuple(WEATHER_FEATURES) + tuple(constructed)


@dataclass(frozen=True)
class TFTStats:
    dynamic_mean: torch.Tensor
    dynamic_std: torch.Tensor
    soil_mean: torch.Tensor
    soil_std: torch.Tensor
    rs_mean: torch.Tensor | None = None
    rs_std: torch.Tensor | None = None
    # 县均值偏移：fips -> 训练年份对数单产均值。网络只学习相对本县历史水平的年际偏离。
    county_offset: dict | None = None
    # 遥感特征的县内中心：fips -> (6, G, K) 训练年份均值；特征先减去本网格历史均值再标准化。
    rs_center: dict | None = None


class RSFeatureStore:
    """rs_features.py 缓存的逐网格-月份数值遥感特征，按 sample_id 取用。"""

    def __init__(self, path):
        self.path = Path(path)
        raw = np.load(self.path, allow_pickle=False)
        self.names = tuple(str(name) for name in raw["feature_names"].tolist())
        self._features = raw["features"]                      # (N, 6, Gmax, K)
        self._grid_count = raw["grid_count"]
        self._index = {str(sid): i for i, sid in enumerate(raw["sample_id"].tolist())}

    def __len__(self) -> int:
        return len(self._index)

    def get(self, sample_id: str, grids: int) -> np.ndarray:
        """(6, grids, K) float32；网格数与缓存不一致时报错，避免静默错位。"""
        try:
            row = self._index[sample_id]
        except KeyError:
            raise KeyError(f"{sample_id}: not in remote-sensing feature cache {self.path}") from None
        if int(self._grid_count[row]) != grids:
            raise ValueError(f"{sample_id}: cache has {int(self._grid_count[row])} grids, dataset has {grids}")
        return self._features[row, :, :grids]


def load_rs_store(model_config: dict) -> RSFeatureStore | None:
    """按 checkpoint 的 model_config 打开特征缓存并核对 SHA-256；未使用数值特征时返回 None。"""
    if not model_config.get("rs_feature_dim"):
        return None
    from train_dataset import run_output as ro

    path = model_config["rs_features_path"]
    expected = model_config.get("rs_features_sha256")
    if expected and ro.sha256_file(path) != expected:
        raise ValueError(f"remote-sensing feature cache {path} changed since training (SHA-256 mismatch)")
    store = RSFeatureStore(path)
    if len(store.names) != int(model_config["rs_feature_dim"]):
        raise ValueError("remote-sensing feature cache dimension differs from checkpoint")
    return store


def hargreaves_pet(tmean_c: torch.Tensor, tmax_c: torch.Tensor, tmin_c: torch.Tensor,
                   lat_deg: torch.Tensor, doy_start: int = 91) -> torch.Tensor:
    steps = tmean_c.shape[-1]
    doy = torch.arange(steps, dtype=tmean_c.dtype, device=tmean_c.device) + doy_start
    lat = torch.deg2rad(lat_deg).unsqueeze(-1)
    dr = 1.0 + 0.033 * torch.cos(2.0 * math.pi * doy / 365.0)
    delta = 0.409 * torch.sin(2.0 * math.pi * doy / 365.0 - 1.39)
    sin_phi, cos_phi = torch.sin(lat), torch.cos(lat)
    omega = torch.acos(torch.clamp(-torch.tan(lat) * torch.tan(delta), -1.0, 1.0))
    ra = (24.0 * 60.0 / math.pi) * _GSC * dr * (
        omega * sin_phi * torch.sin(delta) + cos_phi * torch.cos(delta) * torch.sin(omega)
    )
    return 0.0023 * (ra / _LAMBDA) * (tmean_c + 17.8) * torch.sqrt(
        torch.clamp(tmax_c - tmin_c, min=0.0)
    )


def append_constructed_features(feats: torch.Tensor, coords: torch.Tensor,
                                names: Sequence[str]) -> torch.Tensor:
    tmean = feats[..., 0] - 273.15
    tmax = feats[..., 1] - 273.15
    tmin = feats[..., 2] - 273.15
    prcp = torch.clamp(feats[..., 3], min=0.0)
    result = [feats]
    for name in names:
        if name == "CumGDD":
            value = torch.cumsum(torch.clamp(tmean - GDD_BASE, min=0.0), dim=-1)
        elif name == "KDD":
            # 热害度日按日最高温计（日均温超过 30℃ 在玉米带几乎不出现，该定义会退化为常数）
            value = torch.cumsum(torch.clamp(tmax - KDD_BASE, min=0.0), dim=-1)
        elif name == "CumPRCP":
            value = torch.cumsum(prcp, dim=-1)
        elif name == "CumDeficit":
            pet = hargreaves_pet(tmean, tmax, tmin, coords[..., 0])
            value = torch.cumsum(prcp, dim=-1) - torch.cumsum(pet, dim=-1)
        else:
            raise ValueError(f"unknown constructed feature: {name}")
        result.append(value.unsqueeze(-1))
    return torch.cat(result, dim=-1)


def construct_weather(weather: torch.Tensor, coords: torch.Tensor,
                      feature_names: Sequence[str], use_constructed: bool) -> torch.Tensor:
    names = list(feature_names)
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate weather feature {duplicates}")
    if weather.shape[-1] != len(names):
        raise ValueError(f"last dimension {weather.shape[-1]} != {len(names)} weather features")
    index = {name: i for i, name in enumerate(names)}
    missing = [name for name in WEATHER_FEATURES if name not in index]
    if missing:
        raise ValueError(f"missing weather feature {missing[0]}")
    ordered = weather[..., [index[name] for name in WEATHER_FEATURES]]
    names = CONSTRUCTED_FEATURES if use_constructed else ("CumGDD",)
    return append_constructed_features(ordered, coords=coords, names=names)


def build_dataset(path: Path, data_root: Path, years,
                  *, image_size: int = 224, load_images: bool = True) -> CropNetDataset:
    return CropNetDataset(
        Path(path), tuple(years), load_images=load_images, data_root=Path(data_root), image_size=image_size
    )


def _stats_from_accumulators(total: torch.Tensor, squares: torch.Tensor,
                             count: int) -> tuple[torch.Tensor, torch.Tensor]:
    if count == 0:
        raise ValueError("dataset has no values for statistics")
    mean = total / count
    variance = (squares / count - mean * mean).clamp_min(0.0)
    return mean.float(), variance.sqrt().clamp_min(1e-6).float()


def fit_stats(dataset: CropNetDataset, *, use_constructed: bool,
              rs_store: RSFeatureStore | None = None, train_years=TRAIN_YEARS,
              county_offset: bool = False, demean_rs: bool = False) -> TFTStats:
    actual_years = {int(sample["year"]) for sample in dataset.samples}
    unexpected = sorted(actual_years - set(train_years))
    if unexpected:
        raise ValueError(f"statistics require training years only; found {unexpected}")
    if demean_rs and rs_store is None:
        raise ValueError("demean_rs requires numeric remote-sensing features")
    rs_center = None
    if demean_rs:
        grouped: dict[str, list] = {}
        for sample in dataset.samples:
            grouped.setdefault(sample["fips"], []).append(
                rs_store.get(sample["sample_id"], len(sample["grid_ids"])).astype(np.float64))
        rs_center = {fips: np.mean(np.stack(values), axis=0) for fips, values in grouped.items()}
    offset = None
    if county_offset:
        logs: dict[str, list] = {}
        for sample in dataset.samples:
            logs.setdefault(sample["fips"], []).append(math.log(float(sample["label"])))
        offset = {fips: float(np.mean(values)) for fips, values in logs.items()}
    dynamic_size = len(WEATHER_FEATURES) + (len(CONSTRUCTED_FEATURES) if use_constructed else 1)
    dynamic_sum = torch.zeros(dynamic_size, dtype=torch.float64)
    dynamic_sumsq = torch.zeros(dynamic_size, dtype=torch.float64)
    soil_sum = torch.zeros(len(SOIL_FEATURES), dtype=torch.float64)
    soil_sumsq = torch.zeros(len(SOIL_FEATURES), dtype=torch.float64)
    dynamic_count = 0
    soil_count = 0
    rs_sum = rs_sumsq = None
    rs_count = 0
    for sample in dataset.samples:
        if rs_store is not None:
            values = torch.from_numpy(rs_store.get(sample["sample_id"], len(sample["grid_ids"])).astype(np.float64))
            if rs_center is not None:
                values = values - torch.from_numpy(rs_center[sample["fips"]])
            values = values.reshape(-1, values.shape[-1])
            if rs_sum is None:
                rs_sum = torch.zeros(values.shape[1], dtype=torch.float64)
                rs_sumsq = torch.zeros(values.shape[1], dtype=torch.float64)
            rs_sum += values.sum(dim=0)
            rs_sumsq += (values * values).sum(dim=0)
            rs_count += values.shape[0]
        weather = torch.from_numpy(sample["weather"])
        coords = torch.from_numpy(sample["coords"])
        features = construct_weather(weather, coords, WEATHER_FEATURES, use_constructed)
        features = features.to(dtype=torch.float64).reshape(-1, dynamic_size)
        dynamic_sum += features.sum(dim=0)
        dynamic_sumsq += (features * features).sum(dim=0)
        dynamic_count += features.shape[0]

        soil = torch.from_numpy(sample["soil"]).to(dtype=torch.float64)
        soil_sum += soil
        soil_sumsq += soil * soil
        soil_count += 1
    dynamic_mean, dynamic_std = _stats_from_accumulators(dynamic_sum, dynamic_sumsq, dynamic_count)
    soil_mean, soil_std = _stats_from_accumulators(soil_sum, soil_sumsq, soil_count)
    rs_mean = rs_std = None
    if rs_sum is not None:
        rs_mean, rs_std = _stats_from_accumulators(rs_sum, rs_sumsq, rs_count)
    return TFTStats(dynamic_mean, dynamic_std, soil_mean, soil_std, rs_mean, rs_std, offset, rs_center)


def stats_to_json(stats: TFTStats) -> dict:
    payload = {
        "dynamic_mean": stats.dynamic_mean.tolist(),
        "dynamic_std": stats.dynamic_std.tolist(),
        "soil_mean": stats.soil_mean.tolist(),
        "soil_std": stats.soil_std.tolist(),
    }
    if stats.rs_mean is not None:
        payload["rs_mean"] = stats.rs_mean.tolist()
        payload["rs_std"] = stats.rs_std.tolist()
    if stats.county_offset is not None:
        payload["county_offset"] = dict(stats.county_offset)
    if stats.rs_center is not None:
        payload["rs_center"] = {fips: value.tolist() for fips, value in stats.rs_center.items()}
    return payload


def stats_from_json(payload: dict) -> TFTStats:
    keys = ("dynamic_mean", "dynamic_std", "soil_mean", "soil_std")
    if "rs_mean" in payload:
        keys += ("rs_mean", "rs_std")
    tensors = {key: torch.tensor(payload[key], dtype=torch.float32) for key in keys}
    offset = {k: float(v) for k, v in payload["county_offset"].items()} if "county_offset" in payload else None
    center = ({k: np.asarray(v, dtype=np.float64) for k, v in payload["rs_center"].items()}
              if "rs_center" in payload else None)
    return TFTStats(**tensors, county_offset=offset, rs_center=center)


def _standardize(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    shape = [1] * (value.ndim - 1) + [-1]
    return (value - mean.view(shape)) / std.clamp_min(1e-6).view(shape)


def _lookup(table: dict, fips: str, what: str):
    """县级统计量只由训练年份估计；测试县必须在训练年份出现过（固定效应要求已见县）。"""
    try:
        return table[fips]
    except KeyError:
        raise KeyError(f"{fips}: no {what} from training years (county unseen in training)") from None


def _tft_collate(items, stats: TFTStats, use_constructed: bool,
                 rs_store: RSFeatureStore | None = None) -> dict:
    raw = collate_county_year(items)
    features = [
        construct_weather(item["weather"], item["coords"], WEATHER_FEATURES, use_constructed)
        for item in items
    ]
    batch_size, max_grids = raw["weather"].shape[:2]
    feature_count = features[0].shape[-1]
    grid_feats = torch.zeros((batch_size, max_grids, features[0].shape[1], feature_count))
    for index, feature in enumerate(features):
        grid_feats[index, :feature.shape[0]] = feature
    grid_feats = _standardize(grid_feats, stats.dynamic_mean, stats.dynamic_std)
    grid_feats = torch.where(raw["grid_mask"].unsqueeze(-1).unsqueeze(-1), grid_feats, torch.zeros_like(grid_feats))
    soil_feats = _standardize(raw["soil"], stats.soil_mean, stats.soil_std)
    dates = weather_dates(items[0]["year"])
    month = torch.tensor([int(date[5:7]) for date in dates], dtype=torch.long)
    day = torch.tensor([int(date[8:10]) for date in dates], dtype=torch.long)
    rs_features = None
    if rs_store is None:
        ag_grid_mask = raw["image_mask"]
        ag_images = raw["images"]
    else:
        if stats.rs_mean is None:
            raise ValueError("stats lack remote-sensing statistics; fit_stats needs the feature store")
        ag_images = None
        rs_features = torch.zeros((batch_size, 6, max_grids, len(rs_store.names)))
        for index, item in enumerate(items):
            values = rs_store.get(item["sample_id"], len(item["grid_ids"])).astype(np.float64)
            if stats.rs_center is not None:
                values = values - _lookup(stats.rs_center, item["fips"], "remote-sensing centre")
            values = torch.from_numpy(values).float()
            rs_features[index, :, :values.shape[1]] = _standardize(values, stats.rs_mean, stats.rs_std)
        ag_grid_mask = raw["grid_mask"][:, None, :].expand(batch_size, 6, max_grids).clone()
        rs_features = rs_features * ag_grid_mask.unsqueeze(-1)
    offset = None
    if stats.county_offset is not None:
        offset = torch.tensor([_lookup(stats.county_offset, item["fips"], "county offset") for item in items],
                              dtype=torch.float32)
    return {
        "county_offset": offset,
        "grid_feats": grid_feats,
        "grid_coords": raw["coords"],
        "grid_mask": raw["grid_mask"],
        "soil_feats": soil_feats,
        "seq_lens": torch.full((batch_size,), grid_feats.shape[2], dtype=torch.long),
        "month_ids": month.unsqueeze(0).expand(batch_size, -1),
        "day_ids": day.unsqueeze(0).expand(batch_size, -1),
        "ag_images": ag_images,
        "rs_features": rs_features,
        "ag_mask": ag_grid_mask.any(dim=2),
        "ag_grid_mask": ag_grid_mask,
        "dynamic_feature_names": dynamic_feature_names(use_constructed),
        "labels": raw["labels"],
        "sample_id": raw["sample_id"],
        "fips": raw["fips"],
        "year": raw["year"],
        "grid_ids": raw["grid_ids"],
    }


def cap_grids(batch: dict, max_grids: int) -> dict:
    """把每个样本的有效网格数限制到 max_grids（等间距抽稀，确定性）。

    仅由基线TFT（PVT 图像编码器 + 网格级 TFT）使用：TFT 的注意力在网格—日 token
    上是二次复杂度，单县 321 个网格（全部县年的 0.13%）会使一次前向占用上百 GiB。
    PA-TFT 走数值植被指数、在县内先聚合网格，不经过这里。
    """
    if max_grids <= 0 or batch["grid_mask"].shape[1] <= max_grids:
        return batch
    B = batch["grid_mask"].shape[0]
    picks = []
    for b in range(B):
        idx = torch.nonzero(batch["grid_mask"][b], as_tuple=False).flatten()
        if idx.numel() <= max_grids:
            picks.append(idx)
            continue
        picks.append(idx[torch.linspace(0, idx.numel() - 1, max_grids).long()])
    G2 = max(int(p.numel()) for p in picks)
    # 真正缩短网格维（只靠 mask 不省显存：注意力的复杂度由填充后的网格数决定）
    keep = torch.zeros(B, G2, dtype=torch.bool)
    for b, p in enumerate(picks):
        keep[b, :p.numel()] = True
    index = torch.zeros(B, G2, dtype=torch.long)
    for b, p in enumerate(picks):
        index[b, :p.numel()] = p
    out = dict(batch)
    out["grid_mask"] = keep
    out["grid_feats"] = torch.gather(batch["grid_feats"], 1, index[:, :, None, None].expand(-1, -1, *batch["grid_feats"].shape[2:]))
    out["grid_coords"] = torch.gather(batch["grid_coords"], 1, index[:, :, None].expand(-1, -1, batch["grid_coords"].shape[2]))
    if batch.get("ag_images") is not None:
        out["ag_images"] = torch.gather(
            batch["ag_images"], 2,
            index[:, None, :, None, None, None].expand(-1, batch["ag_images"].shape[1], -1, *batch["ag_images"].shape[3:]))
    ag_keep = torch.gather(batch["ag_grid_mask"], 2, index[:, None, :].expand(-1, batch["ag_grid_mask"].shape[1], -1)) & keep[:, None, :]
    out["ag_grid_mask"] = ag_keep
    out["ag_mask"] = ag_keep.any(dim=2)
    for key in ("grid_ids",):
        if key in batch and isinstance(batch[key], torch.Tensor):
            out[key] = torch.gather(batch[key], 1, index)
    return out


def model_inputs(batch: dict, device) -> dict[str, torch.Tensor]:
    max_grids = int(os.environ.get("TFT_MAX_GRIDS", "0"))
    if max_grids:
        batch = cap_grids(batch, max_grids)
    keys = (
        "grid_feats", "grid_coords", "grid_mask", "soil_feats", "seq_lens",
        "month_ids", "day_ids", "ag_mask", "ag_grid_mask",
    )
    inputs = {key: batch[key].to(device) for key in keys}
    for key in ("ag_images", "rs_features"):
        if batch.get(key) is not None:
            inputs[key] = batch[key].to(device)
    return inputs


class _TFTCollate:
    def __init__(self, stats: TFTStats, use_constructed: bool, rs_store: RSFeatureStore | None = None):
        self.stats = stats
        self.use_constructed = use_constructed
        self.rs_store = rs_store

    def __call__(self, items):
        return _tft_collate(items, self.stats, self.use_constructed, self.rs_store)


def build_loader(dataset, stats: TFTStats, batch_size: int, shuffle: bool, *,
                 use_constructed: bool, num_workers: int, generator=None,
                 rs_store: RSFeatureStore | None = None) -> DataLoader:
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        generator=generator, collate_fn=_TFTCollate(stats, use_constructed, rs_store),
        persistent_workers=num_workers > 0,
    )
