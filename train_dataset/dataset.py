"""CropNet 统一 JSONL 的常量、schema 校验、网格匹配与共享读取接口。

所有模型只通过本模块读取 train_dataset/dataset.jsonl。
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from pathlib import Path, PurePosixPath

import numpy as np


DATA_ROOT = Path("/data/raid0/hqx/DataSrc")
DATASET_PATH = Path("/data/raid0/hqx/train_dataset/dataset.jsonl")
SCHEMA_VERSION = 1

WEATHER_FEATURES = (
    "avg_temperature_k", "max_temperature_k", "min_temperature_k",
    "precipitation_kg_m2", "relative_humidity_pct", "wind_gust_m_s",
    "wind_speed_m_s", "u_wind_m_s", "v_wind_m_s",
    "downward_shortwave_radiation_w_m2", "vapor_pressure_deficit_kpa",
)
SOIL_FEATURES = (
    "clay_pct", "sand_pct", "silt_pct", "om_pct", "ph",
    "bulk_density", "awc",
)
YEARS = tuple(range(2017, 2023))
REMOTE_MONTHS = tuple(range(4, 10))
DAYS_PER_MONTH = 28
WEATHER_DAYS = DAYS_PER_MONTH * len(REMOTE_MONTHS)
LONG_TERM_YEARS = 4
GRID_TOLERANCE = 1e-3

# 州 ANSI 代码 -> (小写州名, USPS 缩写)；候选州为 CropNet USDA 玉米表覆盖的 32 州，
# 其中阿拉巴马（01）除外：HF 上 2017/AL 的 HRRR 目录只有阿肯色文件的副本，缺 2017 气象，
# 无法构成六年平衡面板，故不进入候选。
STATES = {
    "05": ("arkansas", "AR"), "06": ("california", "CA"),
    "08": ("colorado", "CO"), "10": ("delaware", "DE"), "13": ("georgia", "GA"),
    "16": ("idaho", "ID"), "17": ("illinois", "IL"), "18": ("indiana", "IN"),
    "19": ("iowa", "IA"), "20": ("kansas", "KS"), "21": ("kentucky", "KY"),
    "22": ("louisiana", "LA"), "24": ("maryland", "MD"), "26": ("michigan", "MI"),
    "27": ("minnesota", "MN"), "28": ("mississippi", "MS"), "29": ("missouri", "MO"),
    "31": ("nebraska", "NE"), "36": ("new york", "NY"), "37": ("north carolina", "NC"),
    "38": ("north dakota", "ND"), "39": ("ohio", "OH"), "40": ("oklahoma", "OK"),
    "42": ("pennsylvania", "PA"), "45": ("south carolina", "SC"), "46": ("south dakota", "SD"),
    "47": ("tennessee", "TN"), "48": ("texas", "TX"), "51": ("virginia", "VA"),
    "53": ("washington", "WA"), "55": ("wisconsin", "WI"),
}

ABBR_TO_ANSI = {abbr: ansi for ansi, (_, abbr) in STATES.items()}


# ---------------------------------------------------------------------------
# 源数据目录布局（相对于 DATA_ROOT；下载与构建共用）
# ---------------------------------------------------------------------------

def usda_relpath(year: int) -> str:
    return f"USDA Crop Dataset/data/Corn/{year}/USDA_Corn_County_{year}.csv"


def hrrr_relpath(year: int, state_abbr: str, month: int) -> str:
    ansi = ABBR_TO_ANSI[state_abbr]
    return f"WRF-HRRR Computed Dataset/data/{year}/{state_abbr}/HRRR_{ansi}_{state_abbr}_{year}-{month:02d}.csv"


def sentinel_relpath(year: int, state_abbr: str, month: int) -> str:
    """4--6 月位于 Q2 文件，7--9 月位于 Q3 文件。"""
    ansi = ABBR_TO_ANSI[state_abbr]
    start, end = ("04-01", "06-30") if month <= 6 else ("07-01", "09-30")
    return f"Sentinel-2 Imagery/data/{year}/{state_abbr}/Agriculture_{ansi}_{state_abbr}_{year}-{start}_{year}-{end}.h5"


SOIL_RELPATH = "soil/county_soil.csv"
ADJACENCY_RELPATH = "census/county_adjacency.txt"


TOP_LEVEL_FIELDS = (
    "schema_version", "sample_id", "fips", "year", "state", "state_abbr",
    "county", "yield_bu_ac", "neighbor_fips", "weather_features",
    "weather_dates", "remote_sensing_months", "soil", "grids",
)
GRID_FIELDS = (
    "grid_id", "grid_index", "coordinates", "weather", "remote_sensing",
    "long_term_weather",
)
REMOTE_FIELDS = ("month", "hdf5_path", "fips_group", "date_group", "grid_index")
LONG_TERM_FIELDS = ("years", "months", "features", "values")


# ---------------------------------------------------------------------------
# 时间协议
# ---------------------------------------------------------------------------

def weather_dates(year: int) -> list[str]:
    """4--9 月每月前 28 天，共 168 个日期。"""
    return [
        _dt.date(year, month, day).isoformat()
        for month in REMOTE_MONTHS
        for day in range(1, DAYS_PER_MONTH + 1)
    ]


def remote_months(year: int) -> list[str]:
    return [f"{year}-{month:02d}" for month in REMOTE_MONTHS]


def long_term_years(year: int) -> list[int] | None:
    """2017--2020 无长期气象；2021 -> 2017--2020；2022 -> 2018--2021。"""
    if year < YEARS[0] + LONG_TERM_YEARS:
        return None
    return list(range(year - LONG_TERM_YEARS, year))


# ---------------------------------------------------------------------------
# 网格匹配
# ---------------------------------------------------------------------------

def _as_coords(name: str, coords) -> np.ndarray:
    array = np.asarray(coords, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name}: expected shape (G,2), got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{name}: no grids")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: contains non-finite coordinates")
    return array


def match_grid_coordinates(weather_coords: np.ndarray, image_coords: np.ndarray,
                           tolerance: float = GRID_TOLERANCE) -> list[int]:
    """返回 mapping，使 image_coords[mapping[i]] 与 weather_coords[i] 一一对应。

    每个气象网格必须在容差内恰好有一个遥感候选，且目标索引不得重复；
    不使用贪心最近邻掩盖冲突。
    """
    weather = _as_coords("weather_coords", weather_coords)
    image = _as_coords("image_coords", image_coords)
    if len(weather) != len(image):
        raise ValueError(f"grid count mismatch: weather {len(weather)} vs image {len(image)}")
    distance = np.linalg.norm(weather[:, None, :] - image[None, :, :], axis=2)
    mapping = []
    for i, row in enumerate(distance):
        candidates = np.flatnonzero(row <= tolerance)
        lat, lon = weather[i]
        if len(candidates) == 0:
            raise ValueError(
                f"weather grid {i} at ({lat:.6f}, {lon:.6f}) has no image grid within "
                f"{tolerance} deg (nearest {row.min():.6g})"
            )
        if len(candidates) > 1:
            raise ValueError(
                f"weather grid {i} at ({lat:.6f}, {lon:.6f}) matches multiple image grids "
                f"{candidates.tolist()}"
            )
        mapping.append(int(candidates[0]))
    if len(set(mapping)) != len(mapping):
        seen = {}
        for i, j in enumerate(mapping):
            if j in seen:
                raise ValueError(f"weather grids {seen[j]} and {i} both match image grid {j}")
            seen[j] = i
    return mapping


def stable_grid_order(coords: np.ndarray, tolerance: float = GRID_TOLERANCE) -> list[int]:
    """按 (lat, lon) 升序给出网格顺序；容差内重复坐标直接拒绝。"""
    array = _as_coords("coords", coords)
    distance = np.linalg.norm(array[:, None, :] - array[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    if (distance <= tolerance).any():
        i, j = np.argwhere(distance <= tolerance)[0]
        raise ValueError(f"duplicate grid coordinates at indices {int(i)} and {int(j)}: {array[i].tolist()}")
    return [int(i) for i in np.lexsort((array[:, 1], array[:, 0]))]


# ---------------------------------------------------------------------------
# Schema 校验
# ---------------------------------------------------------------------------

def _fail(sample_id: str, path: str, message: str):
    raise ValueError(f"{sample_id}.{path}: {message}")


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _shape_text(shape) -> str:
    return "x".join(str(n) for n in shape) or "scalar"


def _numeric_array(sample_id: str, path: str, value, shape: tuple[int, ...]) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        _fail(sample_id, path, f"expected shape {_shape_text(shape)}, got ragged or non-numeric array")
    if array.shape != shape:
        _fail(sample_id, path, f"expected shape {_shape_text(shape)}, got {_shape_text(array.shape)}")
    if not np.isfinite(array).all():
        _fail(sample_id, path, "contains non-finite values")
    return array


def _exact_fields(sample_id: str, path: str, value, fields: tuple[str, ...]):
    if not isinstance(value, dict):
        _fail(sample_id, path, "expected an object")
    missing = [name for name in fields if name not in value]
    if missing:
        _fail(sample_id, f"{path}.{missing[0]}" if path else missing[0], "missing field")
    extra = sorted(set(value) - set(fields))
    if extra:
        _fail(sample_id, f"{path}.{extra[0]}" if path else extra[0], "unexpected field")


def _is_fips(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"\d{5}", value) is not None


def validate_sample(sample: dict, *, check_files: bool = False, data_root: Path | None = None) -> None:
    """纯内存 schema 校验；check_files=True 时额外核对 HDF5 引用。错误统一抛 ValueError。"""
    sid = sample.get("sample_id", "<unknown>") if isinstance(sample, dict) else "<unknown>"
    sid = sid if isinstance(sid, str) else "<unknown>"
    _exact_fields(sid, "", sample, TOP_LEVEL_FIELDS)

    if sample["schema_version"] != SCHEMA_VERSION or isinstance(sample["schema_version"], bool):
        _fail(sid, "schema_version", f"expected {SCHEMA_VERSION}")
    fips, year = sample["fips"], sample["year"]
    if not _is_fips(fips):
        _fail(sid, "fips", "expected five-digit string")
    if type(year) is not int or year not in YEARS:
        _fail(sid, "year", f"expected integer in {YEARS[0]}..{YEARS[-1]}")
    if sample["sample_id"] != f"{fips}_{year}":
        _fail(sid, "sample_id", f"expected {fips}_{year}")
    if fips[:2] not in STATES:
        _fail(sid, "fips", "state is outside the study states")
    state, state_abbr = STATES[fips[:2]]
    if sample["state"] != state:
        _fail(sid, "state", f"expected {state!r}")
    if sample["state_abbr"] != state_abbr:
        _fail(sid, "state_abbr", f"expected {state_abbr!r}")
    if not isinstance(sample["county"], str) or not sample["county"]:
        _fail(sid, "county", "expected non-empty string")
    if not _is_number(sample["yield_bu_ac"]) or sample["yield_bu_ac"] <= 0:
        _fail(sid, "yield_bu_ac", "expected finite positive number")

    neighbors = sample["neighbor_fips"]
    if not isinstance(neighbors, list) or not all(_is_fips(n) for n in neighbors):
        _fail(sid, "neighbor_fips", "expected list of five-digit strings")
    if len(set(neighbors)) != len(neighbors):
        _fail(sid, "neighbor_fips", "contains duplicates")
    if neighbors != sorted(neighbors):
        _fail(sid, "neighbor_fips", "must be sorted ascending")
    if fips in neighbors:
        _fail(sid, "neighbor_fips", "must not contain the county itself")

    if sample["weather_features"] != list(WEATHER_FEATURES):
        _fail(sid, "weather_features", "must match the 11 protocol features in order")
    if sample["weather_dates"] != weather_dates(year):
        _fail(sid, "weather_dates", f"expected {WEATHER_DAYS} dates Apr-Sep days 1-28")
    months = remote_months(year)
    if sample["remote_sensing_months"] != months:
        _fail(sid, "remote_sensing_months", f"expected {months}")

    _exact_fields(sid, "soil", sample["soil"], SOIL_FEATURES)
    for name in SOIL_FEATURES:
        if not _is_number(sample["soil"][name]):
            _fail(sid, f"soil.{name}", "expected finite number")

    grids = sample["grids"]
    if not isinstance(grids, list) or not grids:
        _fail(sid, "grids", "expected non-empty list")
    history = long_term_years(year)
    coords = []
    for i, grid in enumerate(grids):
        path = f"grids[{i}]"
        _exact_fields(sid, path, grid, GRID_FIELDS)
        if grid["grid_index"] != i or type(grid["grid_index"]) is not int:
            _fail(sid, f"{path}.grid_index", f"expected {i}")
        if grid["grid_id"] != f"{fips}_grid_{i:04d}":
            _fail(sid, f"{path}.grid_id", f"expected {fips}_grid_{i:04d}")
        _exact_fields(sid, f"{path}.coordinates", grid["coordinates"], ("lat", "lon"))
        lat, lon = grid["coordinates"]["lat"], grid["coordinates"]["lon"]
        if not (_is_number(lat) and -90 <= lat <= 90 and _is_number(lon) and -180 <= lon <= 180):
            _fail(sid, f"{path}.coordinates", "expected finite lat/lon")
        coords.append((lat, lon))
        _exact_fields(sid, f"{path}.weather", grid["weather"], ("values",))
        _numeric_array(sid, f"{path}.weather.values", grid["weather"]["values"],
                       (WEATHER_DAYS, len(WEATHER_FEATURES)))

        refs = grid["remote_sensing"]
        if not isinstance(refs, list) or len(refs) != len(months):
            _fail(sid, f"{path}.remote_sensing", f"expected exactly {len(months)} references")
        for k, (ref, month) in enumerate(zip(refs, months)):
            rpath = f"{path}.remote_sensing[{k}]"
            _exact_fields(sid, rpath, ref, REMOTE_FIELDS)
            if ref["month"] != month:
                _fail(sid, f"{rpath}.month", f"expected {month}")
            hdf5_path = ref["hdf5_path"]
            pure = PurePosixPath(hdf5_path) if isinstance(hdf5_path, str) else None
            if pure is None or pure.is_absolute() or ".." in pure.parts or pure.suffix != ".h5":
                _fail(sid, f"{rpath}.hdf5_path", "expected relative .h5 path under the data root")
            if ref["fips_group"] != fips:
                _fail(sid, f"{rpath}.fips_group", f"expected {fips}")
            if ref["date_group"] != f"{month}-01":
                _fail(sid, f"{rpath}.date_group", f"expected {month}-01 (month-start image only)")
            if type(ref["grid_index"]) is not int or ref["grid_index"] < 0:
                _fail(sid, f"{rpath}.grid_index", "expected non-negative integer")

        long_term = grid["long_term_weather"]
        lpath = f"{path}.long_term_weather"
        if history is None:
            if long_term is not None:
                _fail(sid, lpath, f"must be null for {year}")
        else:
            _exact_fields(sid, lpath, long_term, LONG_TERM_FIELDS)
            if long_term["years"] != history:
                _fail(sid, f"{lpath}.years", f"expected {history}")
            if long_term["months"] != list(range(1, 13)):
                _fail(sid, f"{lpath}.months", "expected months 1..12")
            if long_term["features"] != list(WEATHER_FEATURES):
                _fail(sid, f"{lpath}.features", "must match the 11 protocol features in order")
            _numeric_array(sid, f"{lpath}.values", long_term["values"],
                           (LONG_TERM_YEARS, 12, len(WEATHER_FEATURES)))

    try:
        order = stable_grid_order(np.asarray(coords))
    except ValueError as exc:
        _fail(sid, "grids", str(exc))
    if order != list(range(len(coords))):
        _fail(sid, "grids", "grids must be sorted by (lat, lon)")

    if check_files:
        _check_hdf5_refs(sample, Path(data_root) if data_root is not None else DATA_ROOT)


def _check_hdf5_refs(sample: dict, data_root: Path) -> None:
    import h5py

    sid = sample["sample_id"]
    handles = {}
    try:
        for i, grid in enumerate(sample["grids"]):
            center = np.array([grid["coordinates"]["lat"], grid["coordinates"]["lon"]])
            for k, ref in enumerate(grid["remote_sensing"]):
                rpath = f"grids[{i}].remote_sensing[{k}]"
                path = data_root / ref["hdf5_path"]
                if path not in handles:
                    if not path.is_file():
                        _fail(sid, f"{rpath}.hdf5_path", f"file not found: {path}")
                    handles[path] = h5py.File(path, "r")
                key = f"{ref['fips_group']}/{ref['date_group']}"
                if key not in handles[path]:
                    _fail(sid, rpath, f"group {key} not found in {ref['hdf5_path']}")
                group = handles[path][key]
                if "data" not in group or "coordinates" not in group:
                    _fail(sid, rpath, f"group {key} lacks data/coordinates")
                data, corners = group["data"], group["coordinates"]
                if data.dtype != np.uint8 or data.ndim != 4 or data.shape[-1] != 3:
                    _fail(sid, rpath, f"{key}/data must be uint8 (G,H,W,3), got {data.dtype} {data.shape}")
                if corners.shape != (data.shape[0], 2, 2):
                    _fail(sid, rpath, f"{key}/coordinates must be ({data.shape[0]},2,2), got {corners.shape}")
                index = ref["grid_index"]
                if index >= data.shape[0]:
                    _fail(sid, f"{rpath}.grid_index", f"index {index} out of range for {data.shape[0]} grids")
                image_center = np.asarray(corners[index], dtype=np.float64).mean(axis=0)
                if np.linalg.norm(image_center - center) > GRID_TOLERANCE:
                    _fail(sid, f"{rpath}.grid_index",
                          f"image grid center {image_center.tolist()} does not match {center.tolist()}")
    finally:
        for handle in handles.values():
            handle.close()


# ---------------------------------------------------------------------------
# JSONL 读取与全文件不变量
# ---------------------------------------------------------------------------

def iter_jsonl(path: Path, *, validate: bool = True, check_files: bool = False,
               data_root: Path | None = None):
    """逐行读取；错误消息带行号。"""
    import json

    with open(path, encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"{path}:{lineno}: blank line")
            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({exc})") from None
            if validate:
                try:
                    validate_sample(sample, check_files=check_files, data_root=data_root)
                except ValueError as exc:
                    raise ValueError(f"{path}:{lineno}: {exc}") from None
            yield lineno, sample


def check_dataset_invariants(records, years: tuple[int, ...] = YEARS) -> dict:
    """records: 含 sample_id/fips/year/neighbor_fips 的序列（按文件顺序）。

    唯一键、(year,fips) 排序、每县恰好覆盖 years、各年县集合一致、邻接一致且对称。
    """
    seen, last_key = set(), None
    by_county: dict[str, dict[int, tuple]] = {}
    for rec in records:
        sid, fips, year = rec["sample_id"], rec["fips"], rec["year"]
        if sid in seen:
            raise ValueError(f"duplicate sample_id {sid}")
        seen.add(sid)
        key = (year, fips)
        if last_key is not None and key <= last_key:
            raise ValueError(f"{sid}: rows must be sorted by (year, fips)")
        last_key = key
        by_county.setdefault(fips, {})[year] = tuple(rec["neighbor_fips"])
    counties = set(by_county)
    for fips, per_year in by_county.items():
        if sorted(per_year) != sorted(years):
            raise ValueError(f"{fips}: expected years {list(years)}, got {sorted(per_year)}")
        if len(set(per_year.values())) != 1:
            raise ValueError(f"{fips}: neighbor_fips differs across years")
        neighbors = next(iter(per_year.values()))
        outside = sorted(set(neighbors) - counties)
        if outside:
            raise ValueError(f"{fips}: neighbors outside fixed county set {outside}")
        for n in neighbors:
            if fips not in next(iter(by_county[n].values())):
                raise ValueError(f"{fips}: adjacency with {n} is not symmetric")
    return {"samples": len(seen), "counties": len(counties)}


def _record(sample: dict) -> dict:
    return {k: sample[k] for k in ("sample_id", "fips", "year", "neighbor_fips")}


def check_dataset_file(path: Path, *, check_files: bool = False, data_root: Path | None = None,
                       years: tuple[int, ...] = YEARS) -> dict:
    """流式校验整个 JSONL（不把全部样本留在内存）。"""
    seen_keys = set()
    by_county: dict[str, dict[int, tuple]] = {}
    by_year: dict[int, set[str]] = {year: set() for year in years}
    last_key = None
    sample_count = 0
    grid_min = None
    grid_max = None
    for _, sample in iter_jsonl(path, check_files=check_files, data_root=data_root):
        sid, fips, year = sample["sample_id"], sample["fips"], sample["year"]
        key = (year, fips)
        if key in seen_keys:
            raise ValueError(f"duplicate sample_id {sid}")
        seen_keys.add(key)
        if last_key is not None and key <= last_key:
            raise ValueError(f"{sid}: rows must be sorted by (year, fips)")
        last_key = key
        if year in by_year:
            by_year[year].add(fips)
        per_year = by_county.setdefault(fips, {})
        per_year[year] = tuple(sample["neighbor_fips"])
        grid_count = len(sample["grids"])
        grid_min = grid_count if grid_min is None else min(grid_min, grid_count)
        grid_max = grid_count if grid_max is None else max(grid_max, grid_count)
        sample_count += 1

    counties = set(by_county)
    expected_years = sorted(years)
    for fips, per_year in by_county.items():
        if sorted(per_year) != expected_years:
            raise ValueError(f"{fips}: expected years {list(years)}, got {sorted(per_year)}")
        if len(set(per_year.values())) != 1:
            raise ValueError(f"{fips}: neighbor_fips differs across years")
        neighbors = next(iter(per_year.values()))
        outside = sorted(set(neighbors) - counties)
        if outside:
            raise ValueError(f"{fips}: neighbors outside fixed county set {outside}")
        for neighbor in neighbors:
            if fips not in next(iter(by_county[neighbor].values())):
                raise ValueError(f"{fips}: adjacency with {neighbor} is not symmetric")

    reference = by_year[years[0]] if years else set()
    for year in years:
        if by_year[year] != reference:
            raise ValueError(f"year {year}: county set differs from {years[0]}")
    return {
        "samples": sample_count,
        "counties": len(counties),
        "grid_min": grid_min or 0,
        "grid_max": grid_max or 0,
    }


def load_jsonl(path: Path, *, validate: bool = True) -> list[dict]:
    samples = [s for _, s in iter_jsonl(path, validate=validate)]
    if validate:
        check_dataset_invariants([_record(s) for s in samples])
    return samples


# ---------------------------------------------------------------------------
# 年份协议与共享 Dataset
# ---------------------------------------------------------------------------

PRETRAIN_YEARS = (2017, 2018, 2019, 2020)    # PhenoYieldNet 无监督预训练
FINETUNE_YEARS = (2021,)                     # PhenoYieldNet 监督微调
SUPERVISED_YEARS = (2017, 2018, 2019, 2020, 2021)  # TFT / UNet-ConvLSTM 监督训练
TEST_YEARS = (2022,)                         # 所有模型共同测试集
GNN_WINDOWS = {"train": (2017, 2018, 2019, 2020, 2021), "test": (2018, 2019, 2020, 2021, 2022)}
IMAGE_SIZE = 224


def select_years(samples, years) -> list:
    years = set(years)
    unknown = sorted(years - set(YEARS))
    if unknown:
        raise ValueError(f"years outside protocol: {unknown}")
    return sorted((s for s in samples if s["year"] in years), key=lambda s: (s["year"], s["fips"]))


def build_gnn_windows(samples) -> dict[str, list[list]]:
    """每县一个连续五年窗口；缺任一年直接失败，不缩短、不补零。"""
    by_key = {}
    for s in samples:
        key = (s["fips"], s["year"])
        if key in by_key:
            raise ValueError(f"duplicate sample {key[0]}_{key[1]}")
        by_key[key] = s
    counties = sorted({f for f, _ in by_key})
    out = {}
    for split, years in GNN_WINDOWS.items():
        windows = []
        for fips in counties:
            missing = [y for y in years if (fips, y) not in by_key]
            if missing:
                raise ValueError(f"{fips}: GNN {split} window missing years {missing}")
            window = [by_key[(fips, y)] for y in years]
            if len({tuple(s["neighbor_fips"]) for s in window}) != 1:
                raise ValueError(f"{fips}: neighbor_fips differs inside GNN {split} window")
            windows.append(window)
        out[split] = windows
    return out


def _compact(sample: dict) -> dict:
    """把一行 JSON 转成紧凑 numpy 表示，避免全量 Python 列表常驻内存。"""
    grids = sample["grids"]
    long_term = None
    if grids[0]["long_term_weather"] is not None:
        long_term = np.asarray([g["long_term_weather"]["values"] for g in grids], dtype=np.float32)
    return {
        "sample_id": sample["sample_id"], "fips": sample["fips"], "year": sample["year"],
        "state": sample["state"], "county": sample["county"],
        "neighbor_fips": list(sample["neighbor_fips"]),
        "grid_ids": [g["grid_id"] for g in grids],
        "weather": np.asarray([g["weather"]["values"] for g in grids], dtype=np.float32),
        "coords": np.asarray([[g["coordinates"]["lat"], g["coordinates"]["lon"]] for g in grids],
                             dtype=np.float32),
        "soil": np.asarray([sample["soil"][n] for n in SOIL_FEATURES], dtype=np.float32),
        "label": float(sample["yield_bu_ac"]),
        # remote[m][g] = (hdf5_path, fips_group, date_group, grid_index)
        "remote": [[(g["remote_sensing"][m]["hdf5_path"], g["remote_sensing"][m]["fips_group"],
                     g["remote_sensing"][m]["date_group"], g["remote_sensing"][m]["grid_index"])
                    for g in grids] for m in range(len(REMOTE_MONTHS))],
        "long_term": long_term,
    }


class CropNetDataset:
    """所有模型共用的县年 Dataset；遥感像素通过 HDF5 引用惰性加载。

    返回 dict：weather (G,168,11)、coords (G,2)、soil (7,)、label ()、
    long_term (G,4,12,11) 或 None；load_images=True 时额外返回 images (6,G,3,S,S) float32 [0,1]。
    """

    def __init__(self, path: Path, years, *, load_images: bool = False,
                 data_root: Path = DATA_ROOT, image_size: int = IMAGE_SIZE, validate: bool = True):
        self.path = Path(path)
        self.years = tuple(sorted(set(years)))
        unknown = sorted(set(self.years) - set(YEARS))
        if unknown:
            raise ValueError(f"years outside protocol: {unknown}")
        self.load_images = load_images
        self.data_root = Path(data_root)
        self.image_size = image_size
        records, kept = [], []
        for _, sample in iter_jsonl(self.path, validate=validate):
            records.append(_record(sample))
            if sample["year"] in self.years:
                kept.append(_compact(sample))
        check_dataset_invariants(records)
        self.samples = kept

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def sample_ids(self) -> list[str]:
        return [s["sample_id"] for s in self.samples]

    def __getitem__(self, index: int) -> dict:
        import torch

        rec = self.samples[index]
        item = {
            "sample_id": rec["sample_id"], "fips": rec["fips"], "year": rec["year"],
            "neighbor_fips": rec["neighbor_fips"], "grid_ids": rec["grid_ids"],
            "weather": torch.from_numpy(rec["weather"]),
            "coords": torch.from_numpy(rec["coords"]),
            "soil": torch.from_numpy(rec["soil"]),
            "label": torch.tensor(rec["label"], dtype=torch.float32),
            "remote_sensing": rec["remote"],
            "long_term": None if rec["long_term"] is None else torch.from_numpy(rec["long_term"]),
        }
        if self.load_images:
            item["images"] = self._load_images(rec)
        return item

    def _load_images(self, rec: dict):
        import h5py
        import torch
        import torch.nn.functional as F

        size = self.image_size
        out = torch.empty((len(REMOTE_MONTHS), len(rec["grid_ids"]), 3, size, size), dtype=torch.float32)
        handles = {}
        try:
            for m, refs in enumerate(rec["remote"]):
                groups: dict[tuple, list[tuple[int, int]]] = {}
                for g, (path, fips_group, date_group, index) in enumerate(refs):
                    groups.setdefault((path, fips_group, date_group), []).append((g, index))
                for (path, fips_group, date_group), members in groups.items():
                    where = f"{rec['sample_id']}.{rec['grid_ids'][members[0][0]]}.remote_sensing[{m}] {path}"
                    try:
                        if path not in handles:
                            handles[path] = h5py.File(self.data_root / path, "r")
                        data = handles[path][f"{fips_group}/{date_group}"]["data"]
                        wanted = sorted({i for _, i in members})
                        array = data[wanted]
                    except (OSError, KeyError, ValueError, IndexError) as exc:
                        raise ValueError(f"{where}: {exc}") from None
                    tensor = torch.from_numpy(np.ascontiguousarray(array)).permute(0, 3, 1, 2).float().div_(255.0)
                    if tensor.shape[-2:] != (size, size):
                        tensor = F.interpolate(tensor, size=(size, size), mode="bilinear",
                                               align_corners=False, antialias=True)
                    if not torch.isfinite(tensor).all():
                        raise ValueError(f"{where}: non-finite pixels")
                    position = {i: k for k, i in enumerate(wanted)}
                    for g, i in members:
                        out[m, g] = tensor[position[i]]
        finally:
            for handle in handles.values():
                handle.close()
        return out
