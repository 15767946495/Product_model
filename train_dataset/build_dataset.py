"""CropNet 源数据 -> 唯一 JSONL 的构建入口。

  python train_dataset/build_dataset.py                 # 全量构建并维护项目软链接
  python train_dataset/build_dataset.py --validate-only # 只校验，不写文件
  python train_dataset/build_dataset.py --only-fips 19001 --output /tmp/x.jsonl

第一阶段只审计县完整性，第二阶段只为六年完整县构建样本；不边构建边静默跳过。
"""

from __future__ import annotations

import argparse
import calendar
import csv
import functools
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train_dataset.dataset import (  # noqa: E402
    ADJACENCY_RELPATH, DATA_ROOT, DATASET_PATH, DAYS_PER_MONTH, REMOTE_MONTHS,
    SCHEMA_VERSION, SOIL_FEATURES, SOIL_RELPATH, STATES, WEATHER_DAYS,
    WEATHER_FEATURES, YEARS, check_dataset_file, hrrr_relpath, long_term_years,
    match_grid_coordinates, remote_months, sentinel_relpath, stable_grid_order,
    usda_relpath, validate_sample, weather_dates,
)

# 规范变量名 -> HRRR CSV 列名（顺序与 WEATHER_FEATURES 一致）
HRRR_COLUMNS = (
    "Avg Temperature (K)", "Max Temperature (K)", "Min Temperature (K)",
    "Precipitation (kg m**-2)", "Relative Humidity (%)", "Wind Gust (m s**-1)",
    "Wind Speed (m s**-1)", "U Component of Wind (m s**-1)",
    "V Component of Wind (m s**-1)",
    "Downward Shortwave Radiation Flux (W m**-2)",
    "Vapor Pressure Deficit (kPa)",
)
_HRRR_KEYS = ("Year", "Month", "Day", "Lat (llcrnr)", "Lon (llcrnr)", "Lat (urcrnr)", "Lon (urcrnr)")
YIELD_COLUMN = "YIELD, MEASURED IN BU / ACRE"


@dataclass(frozen=True)
class WeatherCube:
    year: int
    dates: list[str]
    coords: np.ndarray   # (G,2) 稳定排序后的网格中心
    values: np.ndarray   # (G,168,11)


@dataclass(frozen=True)
class MonthlyWeatherCube:
    year: int
    coords: np.ndarray   # (G,2)
    values: np.ndarray   # (G,12,11) 由逐日值求月均值


@dataclass(frozen=True)
class ImageRef:
    hdf5_path: str       # 相对于源数据根目录
    fips_group: str
    date_group: str
    coords: np.ndarray   # (G,2) 遥感网格中心，保持 HDF5 内原始顺序


ImageIndex = dict  # month(int) -> ImageRef
# ---------------------------------------------------------------------------
# HRRR 逐日气象
# ---------------------------------------------------------------------------

def _file_key(path: Path) -> tuple:
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"source file not found: {path}")
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


@functools.lru_cache(maxsize=96)
def _hrrr_table(key: tuple) -> dict[str, dict[str, np.ndarray]]:
    """读取一个州月 HRRR CSV，只保留 Daily 行，按 FIPS 分组为数组。"""
    frame = pd.read_csv(key[0], low_memory=False)
    missing = [c for c in ("Daily/Monthly", "FIPS Code", *_HRRR_KEYS, *HRRR_COLUMNS) if c not in frame.columns]
    if missing:
        raise ValueError(f"{key[0]}: missing columns {missing}")
    frame = frame[frame["Daily/Monthly"] == "Daily"]
    numeric = {c: pd.to_numeric(frame[c], errors="coerce").to_numpy(np.float64)
               for c in ("FIPS Code", *_HRRR_KEYS, *HRRR_COLUMNS)}
    fips = numeric.pop("FIPS Code")
    out = {}
    for code in np.unique(fips[np.isfinite(fips)]):
        mask = fips == code
        out[f"{int(code):05d}"] = {c: v[mask] for c, v in numeric.items()}
    return out


def _county_rows(paths: Sequence[Path], fips: str) -> dict[str, np.ndarray]:
    parts = []
    for path in paths:
        part = _hrrr_table(_file_key(path)).get(fips)
        if part is None:
            continue
        for field in ("Year", "Month", "Day"):
            values = part[field]
            invalid = ~np.isfinite(values) | (values != np.round(values))
            if invalid.any():
                value = values[np.flatnonzero(invalid)[0]]
                raise ValueError(
                    f"{fips}: {path}: HRRR {field} must be a finite integer, got {value!r}"
                )
        parts.append(part)
    if not parts:
        raise ValueError(f"{fips}: no HRRR daily rows")
    return {c: np.concatenate([p[c] for p in parts]) for c in parts[0]}


def _grid_ids(rows: dict[str, np.ndarray], fips: str) -> tuple[np.ndarray, np.ndarray]:
    """按网格中心坐标识别网格（不依赖 Grid Index）；返回 (inverse, coords)。"""
    lat = (rows["Lat (llcrnr)"] + rows["Lat (urcrnr)"]) / 2
    lon = (rows["Lon (llcrnr)"] + rows["Lon (urcrnr)"]) / 2
    centers = np.stack([lat, lon], axis=1)
    if not np.isfinite(centers).all():
        raise ValueError(f"{fips}: non-finite HRRR grid coordinates")
    coords, inverse = np.unique(np.round(centers, 6), axis=0, return_inverse=True)
    stable_grid_order(coords)  # 容差内重复（坐标漂移）直接拒绝
    return inverse.reshape(-1), coords


def _single_year(rows: dict[str, np.ndarray], fips: str) -> int:
    years = np.unique(rows["Year"])
    if len(years) != 1 or not np.isfinite(years[0]):
        raise ValueError(f"{fips}: HRRR rows span years {years.tolist()}")
    return int(years[0])


def _check_days(rows, fips):
    day = rows["Day"]
    if not (np.isfinite(day).all() and (day == np.round(day)).all()):
        raise ValueError(f"{fips}: non-integer HRRR day")


def read_daily_weather(paths: Sequence[Path], fips: str) -> WeatherCube:
    """4--9 月前 28 天的网格级逐日气象，(G,168,11)，不做县均值。"""
    rows = _county_rows(paths, fips)
    _check_days(rows, fips)
    keep = np.isin(rows["Month"], REMOTE_MONTHS) & (rows["Day"] >= 1) & (rows["Day"] <= DAYS_PER_MONTH)
    rows = {c: v[keep] for c, v in rows.items()}
    if not len(rows["Month"]):
        raise ValueError(f"{fips}: no growing-season HRRR rows")
    year = _single_year(rows, fips)
    grid, coords = _grid_ids(rows, fips)
    day_index = ((rows["Month"] - REMOTE_MONTHS[0]) * DAYS_PER_MONTH + rows["Day"] - 1).astype(int)
    counts = np.zeros((len(coords), WEATHER_DAYS), dtype=int)
    np.add.at(counts, (grid, day_index), 1)
    dates = weather_dates(year)
    if (counts > 1).any():
        g, d = np.argwhere(counts > 1)[0]
        raise ValueError(f"{fips}_{year}: duplicate HRRR date {dates[d]} at grid {coords[g].tolist()}")
    if (counts == 0).any():
        g, d = np.argwhere(counts == 0)[0]
        raise ValueError(f"{fips}_{year}: missing HRRR date {dates[d]} at grid {coords[g].tolist()} "
                         f"({int((counts == 0).sum())} grid-days missing)")
    values = np.empty((len(coords), WEATHER_DAYS, len(HRRR_COLUMNS)))
    values[grid, day_index] = np.stack([rows[c] for c in HRRR_COLUMNS], axis=1)
    if not np.isfinite(values).all():
        raise ValueError(f"{fips}_{year}: non-finite HRRR weather values")
    order = stable_grid_order(coords)
    return WeatherCube(year, dates, coords[order], values[order])


def read_monthly_weather(paths: Sequence[Path], fips: str) -> MonthlyWeatherCube:
    """12 个月网格级月均气象 (G,12,11)：要求每个网格每月全部日历日恰好一条。"""
    rows = _county_rows(paths, fips)
    _check_days(rows, fips)
    year = _single_year(rows, fips)
    grid, coords = _grid_ids(rows, fips)
    month, day = rows["Month"].astype(int), rows["Day"].astype(int)
    if ((month < 1) | (month > 12)).any():
        raise ValueError(f"{fips}_{year}: invalid HRRR month")
    lengths = np.array([calendar.monthrange(year, m)[1] for m in range(1, 13)])
    if ((day < 1) | (day > lengths[month - 1])).any():
        raise ValueError(f"{fips}_{year}: invalid HRRR day")
    counts = np.zeros((len(coords), 12, 31), dtype=int)
    np.add.at(counts, (grid, month - 1, day - 1), 1)
    if (counts > 1).any():
        g, m, d = np.argwhere(counts > 1)[0]
        raise ValueError(f"{fips}_{year}: duplicate HRRR day {year}-{m + 1:02d}-{d + 1:02d} at grid {coords[g].tolist()}")
    per_month = counts.sum(axis=2)
    if (per_month != lengths[None, :]).any():
        g, m = np.argwhere(per_month != lengths[None, :])[0]
        raise ValueError(f"{fips}_{year}: incomplete HRRR month {m + 1:02d} at grid {coords[g].tolist()} "
                         f"({per_month[g, m]}/{lengths[m]} days)")
    sums = np.zeros((len(coords), 12, len(HRRR_COLUMNS)))
    np.add.at(sums, (grid, month - 1), np.stack([rows[c] for c in HRRR_COLUMNS], axis=1))
    values = sums / lengths[None, :, None]
    if not np.isfinite(values).all():
        raise ValueError(f"{fips}_{year}: non-finite HRRR monthly values")
    order = stable_grid_order(coords)
    return MonthlyWeatherCube(year, coords[order], values[order])
# ---------------------------------------------------------------------------
# Sentinel-2 HDF5 月初索引（只读元数据，不读像素）
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=256)
def _h5_meta(key: tuple) -> dict[tuple[str, str], tuple]:
    """(fips, date) -> (data dtype, data shape, corners)。"""
    import h5py

    out = {}
    with h5py.File(key[0], "r") as handle:
        for fips in handle:
            for date in handle[fips]:
                group = handle[fips][date]
                if "data" not in group or "coordinates" not in group:
                    continue
                corners = np.asarray(group["coordinates"][()], dtype=np.float64)
                corners.setflags(write=False)
                out[(fips, date)] = (group["data"].dtype, group["data"].shape, corners)
    return out


def _relative(path: Path, root: Path) -> str:
    try:
        return Path(path).absolute().relative_to(Path(root).absolute()).as_posix()
    except ValueError:
        raise ValueError(f"{path} is not under source root {root}") from None


def read_image_index(paths: Sequence[Path], fips: str, year: int, *,
                     source_root: Path = DATA_ROOT) -> ImageIndex:
    """4--9 月每月一幅 MM-01 图像；缺月初直接失败，不使用 MM-15 替代。"""
    index: ImageIndex = {}
    metas = [(Path(p), _h5_meta(_file_key(p))) for p in paths]
    for month in REMOTE_MONTHS:
        date = f"{year}-{month:02d}-01"
        hits = [(p, meta[(fips, date)]) for p, meta in metas if (fips, date) in meta]
        if not hits:
            raise ValueError(f"{fips}_{year}: missing month-start image {date}")
        if len(hits) > 1:
            raise ValueError(f"{fips}_{year}: image {date} found in multiple files")
        path, (dtype, shape, corners) = hits[0]
        if dtype != np.uint8 or len(shape) != 4 or shape[-1] != 3:
            raise ValueError(f"{fips}_{year}: {date}/data must be uint8 (G,H,W,3), got {dtype} {shape}")
        if corners.shape != (shape[0], 2, 2) or not np.isfinite(corners).all():
            raise ValueError(f"{fips}_{year}: {date}/coordinates must be finite ({shape[0]},2,2)")
        index[month] = ImageRef(_relative(path, source_root), fips, date, corners.mean(axis=1))
    reference_month = REMOTE_MONTHS[0]
    reference = index[reference_month].coords
    for month in REMOTE_MONTHS[1:]:
        current = index[month].coords
        if len(reference) != len(current):
            raise ValueError(
                f"{fips}_{year}: remote grid set differs in {year}-{month:02d} from reference "
                f"{year}-{reference_month:02d}: grid count mismatch: {len(reference)} != {len(current)}; "
                f"reference coordinates {np.round(reference, 6).tolist()}, "
                f"current coordinates {np.round(current, 6).tolist()}"
            )
        try:
            match_grid_coordinates(reference, current)
        except ValueError as exc:
            raise ValueError(
                f"{fips}_{year}: remote grid set differs in {year}-{month:02d} from reference "
                f"{year}-{reference_month:02d}: reference coordinates "
                f"{np.round(reference, 6).tolist()}, current coordinates "
                f"{np.round(current, 6).tolist()}; {exc}"
            ) from None
    return index


# ---------------------------------------------------------------------------
# USDA 单产与土壤
# ---------------------------------------------------------------------------

def read_usda_yields(path: Path, year: int) -> dict[str, dict]:
    """九州县级玉米单产：完全相同的重复归并，标签冲突抛 ValueError。"""
    out: dict[str, dict] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for lineno, row in enumerate(csv.DictReader(handle), 2):
            if row.get("commodity_desc") != "CORN" or row.get("reference_period_desc") != "YEAR":
                continue
            if str(row.get("year", "")).strip() != str(year):
                continue
            state, county = row["state_ansi"].strip().zfill(2), row["county_ansi"].strip()
            if state not in STATES or not county.isdigit():
                continue
            fips = state + county.zfill(3)
            text = row[YIELD_COLUMN].replace(",", "").strip()
            try:
                value = float(text)
            except ValueError:
                raise ValueError(f"{path}:{lineno}: {fips} yield {text!r} is not numeric") from None
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{path}:{lineno}: {fips} yield {value} must be finite positive")
            record = {"fips": fips, "state": STATES[state][0], "state_abbr": STATES[state][1],
                      "county": row["county_name"].strip().lower(), "yield_bu_ac": value}
            if fips in out and out[fips] != record:
                raise ValueError(f"{path}:{lineno}: conflicting USDA rows for {fips}_{year}")
            out[fips] = record
    return out


def read_soil(path: Path) -> dict[str, dict[str, float]]:
    """县级七维土壤：字段完整且有限；重复 FIPS 值冲突失败；额外列忽略。"""
    out: dict[str, dict[str, float]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [c for c in ("FIPS", *SOIL_FEATURES) if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: missing soil columns {missing}")
        for lineno, row in enumerate(reader, 2):
            fips = row["FIPS"].strip().zfill(5)
            values = {}
            for name in SOIL_FEATURES:
                try:
                    values[name] = float(row[name])
                except (TypeError, ValueError):
                    raise ValueError(f"{path}:{lineno}: {fips}.{name} is not numeric") from None
                if not math.isfinite(values[name]):
                    raise ValueError(f"{path}:{lineno}: {fips}.{name} is not finite")
            if fips in out and out[fips] != values:
                raise ValueError(f"{path}:{lineno}: conflicting soil rows for {fips}")
            out[fips] = values
    return out
# ---------------------------------------------------------------------------
# 单个县年样本
# ---------------------------------------------------------------------------

def _daily_paths(root: Path, abbr: str, year: int) -> list[Path]:
    return [root / hrrr_relpath(year, abbr, m) for m in REMOTE_MONTHS]


def _monthly_paths(root: Path, abbr: str, year: int) -> list[Path]:
    return [root / hrrr_relpath(year, abbr, m) for m in range(1, 13)]


def _image_paths(root: Path, abbr: str, year: int) -> list[Path]:
    return [root / sentinel_relpath(year, abbr, 4), root / sentinel_relpath(year, abbr, 7)]


@functools.lru_cache(maxsize=16)
def _usda_cached(key: tuple, year: int) -> dict[str, dict]:
    return read_usda_yields(Path(key[0]), year)


def _usda(root: Path, year: int) -> dict[str, dict]:
    return _usda_cached(_file_key(root / usda_relpath(year)), year)


def _align(target: np.ndarray, other: np.ndarray, context: str) -> list[int]:
    """other[mapping[i]] 对应 target[i]；失败时报告 FIPS、年份和坐标。"""
    try:
        return match_grid_coordinates(target, other)
    except ValueError as exc:
        raise ValueError(f"{context}: {exc}; target grids {len(target)} "
                         f"(first {np.round(target[:3], 6).tolist()}), other grids {len(other)}") from None


def build_county_year_sample(source_root: Path, fips: str, year: int, soil: dict,
                             neighbors: list[str]) -> dict:
    """构建并校验一个县年样本；像素不进入 JSONL，只写结构化 HDF5 引用。"""
    root = Path(source_root)
    if fips[:2] not in STATES:
        raise ValueError(f"{fips}: outside the nine study states")
    state, abbr = STATES[fips[:2]]
    sid = f"{fips}_{year}"
    labels = _usda(root, year)
    if fips not in labels:
        raise ValueError(f"{sid}: no USDA corn yield")
    if sorted(soil) != sorted(SOIL_FEATURES):
        raise ValueError(f"{sid}: soil must have exactly {SOIL_FEATURES}")

    weather = read_daily_weather(_daily_paths(root, abbr, year), fips)
    if weather.year != year:
        raise ValueError(f"{sid}: HRRR files contain year {weather.year}")
    images = read_image_index(_image_paths(root, abbr, year), fips, year, source_root=root)
    image_map = {m: _align(weather.coords, images[m].coords, f"{sid} image {year}-{m:02d}")
                 for m in REMOTE_MONTHS}

    history = long_term_years(year)
    long_term = None
    if history is not None:
        per_year = []
        for hist_year in history:
            cube = read_monthly_weather(_monthly_paths(root, abbr, hist_year), fips)
            mapping = _align(weather.coords, cube.coords, f"{sid} history year {hist_year}")
            per_year.append(cube.values[mapping])
        long_term = np.stack(per_year, axis=1)  # (G,4,12,11)

    grids = []
    for i, (lat, lon) in enumerate(weather.coords):
        grids.append({
            "grid_id": f"{fips}_grid_{i:04d}",
            "grid_index": i,
            "coordinates": {"lat": float(lat), "lon": float(lon)},
            "weather": {"values": weather.values[i].tolist()},
            "remote_sensing": [
                {"month": f"{year}-{m:02d}", "hdf5_path": images[m].hdf5_path, "fips_group": fips,
                 "date_group": images[m].date_group, "grid_index": int(image_map[m][i])}
                for m in REMOTE_MONTHS
            ],
            "long_term_weather": None if long_term is None else {
                "years": history, "months": list(range(1, 13)), "features": list(WEATHER_FEATURES),
                "values": np.round(long_term[i], 6).tolist(),
            },
        })
    record = labels[fips]
    sample = {
        "schema_version": SCHEMA_VERSION, "sample_id": sid, "fips": fips, "year": year,
        "state": state, "state_abbr": abbr, "county": record["county"],
        "yield_bu_ac": record["yield_bu_ac"], "neighbor_fips": sorted(neighbors),
        "weather_features": list(WEATHER_FEATURES), "weather_dates": weather.dates,
        "remote_sensing_months": remote_months(year),
        "soil": {name: float(soil[name]) for name in SOIL_FEATURES}, "grids": grids,
    }
    validate_sample(sample)
    return sample
# ---------------------------------------------------------------------------
# 第一阶段：县年审计（不写样本）
# ---------------------------------------------------------------------------

def _count_month_starts(paths, fips, year) -> int:
    found = set()
    for path in paths:
        try:
            meta = _h5_meta(_file_key(path))
        except ValueError:
            continue
        found |= {m for m in REMOTE_MONTHS if (fips, f"{year}-{m:02d}-01") in meta}
    return len(found)


def _audit_county(root: Path, fips: str, soil: dict, labels: dict) -> list[dict]:
    state, abbr = STATES[fips[:2]]
    county = next((labels[y][fips]["county"] for y in YEARS if labels.get(y) and fips in labels[y]), "")
    weather, monthly, rows = {}, {}, []

    def monthly_cube(y):
        if y not in monthly:
            try:
                monthly[y] = read_monthly_weather(_monthly_paths(root, abbr, y), fips)
            except ValueError as exc:
                monthly[y] = exc
        return monthly[y]

    for year in YEARS:
        reasons = []
        row = {"fips": fips, "year": year, "state": state, "county": county, "status": "", "reason": ""}
        if labels.get(year) is None:
            row["usda_status"] = "missing_file"
        else:
            row["usda_status"] = "ok" if fips in labels[year] else "missing_label"
        if row["usda_status"] != "ok":
            reasons.append(f"usda: {row['usda_status']}")
        try:
            cube = read_daily_weather(_daily_paths(root, abbr, year), fips)
            weather[year] = cube
            row["weather_days"], row["weather_grids"] = WEATHER_DAYS, len(cube.coords)
        except ValueError as exc:
            row["weather_days"], row["weather_grids"] = 0, 0
            reasons.append(f"weather: {exc}")
        try:
            images = read_image_index(_image_paths(root, abbr, year), fips, year, source_root=root)
            row["remote_months"] = len(images)
            row["remote_grids"] = len(images[REMOTE_MONTHS[0]].coords)
        except ValueError as exc:
            images = None
            row["remote_months"] = _count_month_starts(_image_paths(root, abbr, year), fips, year)
            row["remote_grids"] = 0
            reasons.append(f"remote: {exc}")
        row["soil_status"] = "ok" if fips in soil else "missing"
        if fips not in soil:
            reasons.append("soil: missing or incomplete gSSURGO record")
        row["grid_match_status"] = "not_checked"
        if year in weather and images is not None:
            try:
                for m in REMOTE_MONTHS:
                    _align(weather[year].coords, images[m].coords, f"{fips}_{year} image {year}-{m:02d}")
                row["grid_match_status"] = "ok"
            except ValueError as exc:
                row["grid_match_status"] = "failed"
                reasons.append(f"grid_match: {exc}")
        history = long_term_years(year)
        row["long_term_status"] = "not_required" if history is None else "not_checked"
        if history is not None and year in weather:
            try:
                for y in history:
                    cube = monthly_cube(y)
                    if isinstance(cube, Exception):
                        raise ValueError(f"history year {y}: {cube}")
                    _align(weather[year].coords, cube.coords, f"{fips}_{year} history year {y}")
                row["long_term_status"] = "ok"
            except ValueError as exc:
                row["long_term_status"] = "failed"
                reasons.append(f"long_term: {exc}")
        row["reason"] = "; ".join(reasons)
        rows.append(row)

    # 固定网格集合必须六年一致：以第一个可读年份为参照逐年比较
    reference = next((weather[y].coords for y in YEARS if y in weather), None)
    for row in rows:
        if row["year"] not in weather:
            row["grid_stability_status"] = "not_checked"
            continue
        coords = weather[row["year"]].coords
        try:
            mapping = _align(reference, coords, f"{fips}_{row['year']} grid stability")
            if mapping != list(range(len(reference))):
                raise ValueError(f"{fips}_{row['year']}: grid order changed across years")
            row["grid_stability_status"] = "ok"
        except ValueError as exc:
            row["grid_stability_status"] = "failed"
            row["reason"] = "; ".join(filter(None, [row["reason"], f"grid_stability: {exc}"]))
    return rows


def audit_sources(source_root: Path, only_fips: set[str] | None = None) -> list[dict]:
    """九州候选县逐年审计；候选县 = 任一年 USDA 玉米记录中出现的县。"""
    root = Path(source_root)
    labels = {}
    for year in YEARS:
        try:
            labels[year] = _usda(root, year)
        except ValueError:
            labels[year] = None
    candidates = sorted({f for y in YEARS if labels[y] for f in labels[y]})
    if only_fips is not None:
        candidates = sorted(set(candidates) | set(only_fips))
        candidates = [f for f in candidates if f in only_fips]
    soil_path = root / SOIL_RELPATH
    soil = read_soil(soil_path) if soil_path.is_file() else {}
    rows = []
    for i, fips in enumerate(candidates, 1):
        if fips[:2] not in STATES:
            raise ValueError(f"{fips}: outside the nine study states")
        rows.extend(_audit_county(root, fips, soil, labels))
        if i % 50 == 0 or i == len(candidates):
            print(f"[audit] {i}/{len(candidates)} counties", file=sys.stderr, flush=True)
    return rows


# ---------------------------------------------------------------------------
# 第二阶段：固定县集合 -> 唯一 JSONL
# ---------------------------------------------------------------------------

@dataclass
class BuildSummary:
    counties: int
    samples: int
    grid_min: int
    grid_max: int
    bytes: int
    sha256: str | None
    output: str | None
    written: bool


def ensure_symlink(link: Path, target: Path) -> None:
    """链接已正确则不改；指向错误或已存在普通文件时拒绝覆盖。"""
    link, target = Path(link), Path(target)
    if link.is_symlink():
        if Path(os.readlink(link)) == target:
            return
        raise FileExistsError(f"{link} points to {os.readlink(link)}, expected {target}")
    if link.exists():
        raise FileExistsError(f"{link} exists and is not a symlink; refusing to overwrite")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)


def build_dataset(source_root: Path, output_path: Path, *, only_fips: set[str] | None = None,
                  validate_only: bool = False, link: Path | None = None) -> BuildSummary:
    from DataSrc.download_cropnet import finalize_audit, parse_county_adjacency

    root, output = Path(source_root), Path(output_path)
    rows = finalize_audit(audit_sources(root, only_fips))
    retained = sorted({r["fips"] for r in rows if r["status"] == "retained"})
    if only_fips is not None:
        missing = sorted(set(only_fips) - set(retained))
        if missing:
            reasons = {r["fips"]: r["reason"] for r in rows if r["fips"] in missing and r["reason"]}
            raise ValueError(f"requested counties are not six-year complete: {missing} {reasons}")
    if not retained:
        raise ValueError("no six-year complete counties")
    adjacency_path = root / ADJACENCY_RELPATH
    if not adjacency_path.is_file():
        raise ValueError(f"county adjacency not found: {adjacency_path}")
    graph = parse_county_adjacency(adjacency_path.read_bytes().decode("latin-1"), set(retained))
    soil = read_soil(root / SOIL_RELPATH)

    hasher = hashlib.sha256()
    count, size, grid_counts = 0, 0, []
    tmp = output.with_name(output.name + ".tmp")
    handle = None
    try:
        if not validate_only:
            output.parent.mkdir(parents=True, exist_ok=True)
            handle = open(tmp, "w", encoding="utf-8")
        for year in YEARS:
            for fips in retained:
                sample = build_county_year_sample(root, fips, year, soil[fips], sorted(graph[fips]))
                validate_sample(sample, check_files=True, data_root=root)
                line = json.dumps(sample, separators=(",", ":"), ensure_ascii=False) + "\n"
                data = line.encode("utf-8")
                hasher.update(data)
                size += len(data)
                count += 1
                grid_counts.append(len(sample["grids"]))
                if handle:
                    handle.write(line)
            print(f"[build] year {year} done ({count} samples)", file=sys.stderr, flush=True)
        if handle:
            handle.close()
            handle = None
            check_dataset_file(tmp)  # 逐行重新读取并校验，通过后才替换正式文件
            os.replace(tmp, output)
    finally:
        if handle:
            handle.close()
        tmp.unlink(missing_ok=True)
    if link is not None and not validate_only:
        ensure_symlink(link, output)
    return BuildSummary(len(retained), count, min(grid_counts), max(grid_counts), size,
                        hasher.hexdigest(), None if validate_only else str(output), not validate_only)


PROJECT_LINK = Path(__file__).resolve().parent / "dataset.jsonl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DATASET_PATH)
    parser.add_argument("--only-fips", nargs="+", default=None)
    parser.add_argument("--validate-only", action="store_true", help="完整试构建与校验，不写任何文件")
    args = parser.parse_args(argv)
    only = {f.zfill(5) for f in args.only_fips} if args.only_fips else None
    link = PROJECT_LINK if args.output == DATASET_PATH and only is None else None
    summary = build_dataset(args.source_root, args.output, only_fips=only,
                            validate_only=args.validate_only, link=link)
    report = asdict(summary)
    if args.validate_only and args.output.is_file():
        report["existing_output"] = check_dataset_file(args.output)
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
