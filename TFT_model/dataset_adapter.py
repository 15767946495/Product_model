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
    """前向滚动起点：测试年为 test_year，训练年只用它之前的年份。

    默认 test_year=2022 即 2017--2021 训练（与旧协议相同）。2017 没有更早的年份，
    因此要求 test_year > YEARS[0]。空间交叉验证的 ``--all-years`` 不走这里。
    """
    if test_year not in YEARS:
        raise ValueError(f"test year {test_year} outside {YEARS}")
    train_years = tuple(y for y in YEARS if y < test_year)
    if not train_years:
        raise ValueError(f"test year {test_year} has no preceding training years")
    return train_years, (int(test_year),)


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
    # 空间交叉验证用：sample_id -> 该县更早年份的对数单产均值（留一）。留出县只有
    # 目标年之前的信息可用；目标年本身不在 offset_dataset 中，由 county_offset 兜底。
    offset_by_sample: dict | None = None
    # 标准化的统计口径（第3.6节）：train = 训练年份全局统计量；none = 不标准化；
    # per-year = 每年用自己的统计量；all-years = 训练年与测试年合并估计。
    # per_year 非空时按样本年份取用，见 _tft_collate。
    norm_mode: str = "train"
    per_year: dict | None = None
    # 气象特征的县内中心（--demean-weather）：fips -> (6, G, D) 训练年份逐月均值；
    # 每个网格日的动态特征先减去本县该月的历史均值再标准化，与 rs_center 同一口径。
    weather_center: dict | None = None


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


def _weather_month_index(year: int) -> np.ndarray:
    """168 个气象日 -> 月序号 0--5（四月起算）；各年份的日期布局相同。"""
    return np.array([int(date[5:7]) - 4 for date in weather_dates(year)], dtype=np.int64)


def _accumulate_features(samples, *, dynamic_size: int, use_constructed: bool,
                         rs_store: RSFeatureStore | None, rs_center: dict | None,
                         weather_center: dict | None = None, month_index: np.ndarray | None = None):
    """把一批县年样本的天气、土壤与遥感特征累加成 (sum, sumsq, count)。"""
    dynamic_sum = torch.zeros(dynamic_size, dtype=torch.float64)
    dynamic_sumsq = torch.zeros(dynamic_size, dtype=torch.float64)
    soil_sum = torch.zeros(len(SOIL_FEATURES), dtype=torch.float64)
    soil_sumsq = torch.zeros(len(SOIL_FEATURES), dtype=torch.float64)
    dynamic_count = 0
    soil_count = 0
    rs_sum = rs_sumsq = None
    rs_count = 0
    for sample in samples:
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
        if weather_center is not None:
            # (G, T, D) 减去本县该月的训练年均值：center (6, G, D) -> (T, G, D) -> (G, T, D)
            center = _lookup(weather_center, sample["fips"], "weather centre")
            if center.shape[1] != features.shape[0]:
                raise ValueError(
                    f"{sample['fips']}: weather centre has {center.shape[1]} grids, "
                    f"sample has {features.shape[0]}")
            features = features - torch.from_numpy(
                np.transpose(center[month_index], (1, 0, 2))).to(features.dtype)
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
    return {"dynamic_mean": dynamic_mean, "dynamic_std": dynamic_std,
            "soil_mean": soil_mean, "soil_std": soil_std,
            "rs_mean": rs_mean, "rs_std": rs_std}


def fit_stats(dataset: CropNetDataset, *, use_constructed: bool,
              rs_store: RSFeatureStore | None = None, train_years=TRAIN_YEARS,
              county_offset: bool = False, demean_rs: bool = False,
              demean_weather: bool = False,
              offset_dataset: CropNetDataset | None = None,
              leave_one_out_offset: bool = True,
              norm_mode: str = "train",
              extra_dataset: CropNetDataset | None = None) -> TFTStats:
    """标准化统计量只用训练县年；县均值偏移与遥感县内中心可用另一份全样本数据估计。

    ``offset_dataset`` 用于空间交叉验证：留出的县不出现在训练集里，但它的历史单产与
    历史遥感特征在预测时是已知的。前向协议下它只含测试年之前的年份：所有样本都用
    county_offset（该县更早年份的均值），目标尺度与测试行一致。``leave_one_out_offset``
    只用于旧版 ``--all-years`` 空间折：那时偏移面板含目标年，测试行必须排除自己。

    ``norm_mode`` 是第3.6节的标准化口径消融：``train``（默认）只用训练年份的统计量；
    ``none`` 不标准化；``per-year`` 每年用该年自己的统计量；``all-years`` 用训练年与
    测试年合并的统计量。后两种需要 ``extra_dataset`` 提供测试年份的特征（不使用标签）。
    """
    if norm_mode not in ("train", "none", "per-year", "all-years"):
        raise ValueError(f"unknown norm_mode: {norm_mode}")
    actual_years = {int(sample["year"]) for sample in dataset.samples}
    unexpected = sorted(actual_years - set(train_years))
    if unexpected:
        raise ValueError(f"statistics require training years only; found {unexpected}")
    if demean_rs and rs_store is None:
        raise ValueError("demean_rs requires numeric remote-sensing features")
    if norm_mode in ("per-year", "all-years") and extra_dataset is None:
        raise ValueError(f"norm_mode={norm_mode} requires extra_dataset with the test years")
    source = offset_dataset if offset_dataset is not None else dataset
    weather_center = None
    month_index = None
    if demean_weather:
        # 与 demean_rs 同口径：本县每个网格月的训练年均值，从所有动态特征（含累积量）中减去。
        month_index = _weather_month_index(int(source.samples[0]["year"]))
        grouped_weather: dict[str, list] = {}
        for sample in source.samples:
            weather = torch.from_numpy(sample["weather"])
            coords = torch.from_numpy(sample["coords"])
            features = construct_weather(weather, coords, WEATHER_FEATURES, use_constructed)
            grouped_weather.setdefault(sample["fips"], []).append(
                features.numpy().astype(np.float64))
        weather_center = {}
        for fips, values in grouped_weather.items():
            shapes = {value.shape for value in values}
            if len(shapes) != 1:
                raise ValueError(f"{fips}: weather grids differ across years; cannot centre weather")
            stack = np.stack(values)  # (Y, G, T, D)
            center = np.zeros((6, stack.shape[1], stack.shape[3]), dtype=np.float64)
            for month in range(6):
                center[month] = stack[:, :, month_index == month].mean(axis=(0, 2))
            weather_center[fips] = center
    rs_center = None
    if demean_rs:
        grouped: dict[str, list] = {}
        for sample in source.samples:
            grouped.setdefault(sample["fips"], []).append(
                rs_store.get(sample["sample_id"], len(sample["grid_ids"])).astype(np.float64))
        rs_center = {fips: np.mean(np.stack(values), axis=0) for fips, values in grouped.items()}
    offset = None
    offset_by_sample = None
    if county_offset:
        per_county: dict[str, dict[int, float]] = {}
        for sample in source.samples:
            per_county.setdefault(sample["fips"], {})[int(sample["year"])] = math.log(float(sample["label"]))
        offset = {fips: float(np.mean(list(per_year.values()))) for fips, per_year in per_county.items()}
        if offset_dataset is not None:
            fallback = float(np.mean([math.log(float(sample["label"])) for sample in source.samples]))
            if leave_one_out_offset:
                # 旧版 --all-years 空间折：偏移面板含目标年，测试行必须排除自己。
                offset_by_sample = {}
                for fips, per_year in per_county.items():
                    for year, value in per_year.items():
                        others = [v for y, v in per_year.items() if y != year]
                        offset_by_sample[f"{fips}_{year}"] = float(np.mean(others)) if others else fallback
            else:
                # 前向折：所有行用 county_offset（目标尺度与测试行一致）。只有单年前史
                # （2018 行）时 county_offset 等于样本自身标签，训练目标会恒为 0；
                # 这些行改用该年跨县均值，测试行不在表里，仍回退到 county_offset。
                single = {fips: per_year for fips, per_year in per_county.items() if len(per_year) == 1}
                offset_by_sample = {} if single else None
                for fips, per_year in single.items():
                    for year in per_year:
                        offset_by_sample[f"{fips}_{year}"] = fallback
    dynamic_size = len(WEATHER_FEATURES) + (len(CONSTRUCTED_FEATURES) if use_constructed else 1)
    if norm_mode == "none":
        dynamic_mean = torch.zeros(dynamic_size)
        dynamic_std = torch.ones(dynamic_size)
        soil_mean = torch.zeros(len(SOIL_FEATURES))
        soil_std = torch.ones(len(SOIL_FEATURES))
        rs_mean = rs_std = None
        if rs_store is not None:
            rs_mean = torch.zeros(len(rs_store.names))
            rs_std = torch.ones(len(rs_store.names))
        return TFTStats(dynamic_mean, dynamic_std, soil_mean, soil_std, rs_mean, rs_std, offset, rs_center,
                        offset_by_sample, norm_mode=norm_mode, weather_center=weather_center)
    accumulate = dict(dynamic_size=dynamic_size, use_constructed=use_constructed,
                      rs_store=rs_store, rs_center=rs_center,
                      weather_center=weather_center, month_index=month_index)
    if norm_mode == "all-years":
        # 测试年份只贡献特征（无标签），用于检验"不用训练集统计量"的替代口径。
        stats = _accumulate_features(list(dataset.samples) + list(extra_dataset.samples), **accumulate)
    else:
        stats = _accumulate_features(dataset.samples, **accumulate)
    per_year = None
    if norm_mode == "per-year":
        by_year: dict[int, list] = {}
        for sample in list(dataset.samples) + list(extra_dataset.samples):
            by_year.setdefault(int(sample["year"]), []).append(sample)
        per_year = {year: _accumulate_features(samples, **accumulate) for year, samples in sorted(by_year.items())}
    return TFTStats(stats["dynamic_mean"], stats["dynamic_std"], stats["soil_mean"], stats["soil_std"],
                    stats["rs_mean"], stats["rs_std"], offset, rs_center, offset_by_sample,
                    norm_mode=norm_mode, per_year=per_year, weather_center=weather_center)


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
    if stats.weather_center is not None:
        payload["weather_center"] = {fips: value.tolist() for fips, value in stats.weather_center.items()}
    if stats.offset_by_sample is not None:
        payload["offset_by_sample"] = dict(stats.offset_by_sample)
    payload["norm_mode"] = stats.norm_mode
    if stats.per_year is not None:
        payload["per_year"] = {
            str(year): {key: (None if value is None else value.tolist()) for key, value in values.items()}
            for year, values in stats.per_year.items()
        }
    return payload


def stats_from_json(payload: dict) -> TFTStats:
    keys = ("dynamic_mean", "dynamic_std", "soil_mean", "soil_std")
    if "rs_mean" in payload:
        keys += ("rs_mean", "rs_std")
    tensors = {key: torch.tensor(payload[key], dtype=torch.float32) for key in keys}
    offset = {k: float(v) for k, v in payload["county_offset"].items()} if "county_offset" in payload else None
    center = ({k: np.asarray(v, dtype=np.float64) for k, v in payload["rs_center"].items()}
              if "rs_center" in payload else None)
    by_sample = ({k: float(v) for k, v in payload["offset_by_sample"].items()}
                 if "offset_by_sample" in payload else None)
    weather = ({k: np.asarray(v, dtype=np.float64) for k, v in payload["weather_center"].items()}
               if "weather_center" in payload else None)
    per_year = None
    if "per_year" in payload:
        per_year = {
            int(year): {key: (None if value is None else torch.tensor(value, dtype=torch.float32))
                        for key, value in values.items()}
            for year, values in payload["per_year"].items()
        }
    return TFTStats(**tensors, county_offset=offset, rs_center=center, offset_by_sample=by_sample,
                    norm_mode=payload.get("norm_mode", "train"), per_year=per_year,
                    weather_center=weather)


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
    if stats.weather_center is not None:
        # 气象县内去均值（第5章消融）：先减去本县该月的历史均值，再做全局标准化。
        month_index = _weather_month_index(int(items[0]["year"]))
        for index, item in enumerate(items):
            center = _lookup(stats.weather_center, item["fips"], "weather centre")
            grids = features[index].shape[0]
            if center.shape[1] != grids:
                raise ValueError(
                    f"{item['fips']}: weather centre has {center.shape[1]} grids, sample has {grids}")
            centered = np.transpose(center[month_index], (1, 0, 2))
            grid_feats[index, :grids] -= torch.from_numpy(centered).to(grid_feats.dtype)
    if stats.per_year is None:
        grid_feats = _standardize(grid_feats, stats.dynamic_mean, stats.dynamic_std)
    else:
        # 按年标准化的消融：每行样本用自己年份的统计量（第3.6节）。
        years = [int(item["year"]) for item in items]
        dynamic_mean = torch.stack([stats.per_year[year]["dynamic_mean"] for year in years])
        dynamic_std = torch.stack([stats.per_year[year]["dynamic_std"] for year in years]).clamp_min(1e-6)
        grid_feats = (grid_feats - dynamic_mean.view(batch_size, 1, 1, -1)) / dynamic_std.view(batch_size, 1, 1, -1)
    grid_feats = torch.where(raw["grid_mask"].unsqueeze(-1).unsqueeze(-1), grid_feats, torch.zeros_like(grid_feats))
    if stats.per_year is None:
        soil_feats = _standardize(raw["soil"], stats.soil_mean, stats.soil_std)
    else:
        soil_mean = torch.stack([stats.per_year[int(item["year"])]["soil_mean"] for item in items])
        soil_std = torch.stack([stats.per_year[int(item["year"])]["soil_std"] for item in items]).clamp_min(1e-6)
        soil_feats = (raw["soil"] - soil_mean) / soil_std
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
            if stats.per_year is None:
                rs_mean, rs_std = stats.rs_mean, stats.rs_std
            else:
                rs_mean, rs_std = stats.per_year[int(item["year"])]["rs_mean"], stats.per_year[int(item["year"])]["rs_std"]
            rs_features[index, :, :values.shape[1]] = _standardize(values, rs_mean, rs_std)
        ag_grid_mask = raw["grid_mask"][:, None, :].expand(batch_size, 6, max_grids).clone()
        rs_features = rs_features * ag_grid_mask.unsqueeze(-1)
    offset = None
    if stats.offset_by_sample is not None:
        values = []
        for item in items:
            value = stats.offset_by_sample.get(item["sample_id"])
            if value is None:
                # 前向空间折：测试年份不在 offset_dataset 里，回退到该县目标年之前
                # 各年对数单产的均值（county_offset）。
                if stats.county_offset is None:
                    raise KeyError(f"{item['sample_id']}: no county offset for forward test year")
                value = _lookup(stats.county_offset, item["fips"], "county offset")
            values.append(value)
        offset = torch.tensor(values, dtype=torch.float32)
    elif stats.county_offset is not None:
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
