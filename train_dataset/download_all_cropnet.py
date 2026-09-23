"""从 HuggingFace CropNet 下载全美县级玉米训练数据。

用法：python -u train_dataset/download_all_cropnet.py

断点：每个州-年处理完写入 checkpoint，重启时跳过。
数据路径：
  AG:   sentinel/source/AG/{year}/{state}/
  Weather: weather/{year}/{state}/
  Label: label/hf/
  Soil:  soil/
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from download_complete_cropnet_hf_with_soil import (
    ANSI_TO_USPS, MONTHS, SOIL_COLS,
    aggregate_gssurgo, query_gssurgo,
)

ROOT = Path("/data/raid0/hqx/Product_model_runtime")
DATASRC = ROOT / "DataSrc"
AG_DIR = DATASRC / "sentinel" / "source" / "AG"
WEATHER_DIR = DATASRC / "weather"
LABEL_DIR = DATASRC / "label" / "hf"
SOIL_DIR = DATASRC / "soil"
ADDED_SOIL = SOIL_DIR / "county_soil_gssurgo_added.json"
CHECKPOINT = DATASRC / ".cropnet_download_ckpt.json"
HF_REPO = "CropNet/CropNet"
YEARS = [2017, 2018, 2019, 2020, 2021, 2022]

_T0 = time.time()

def log(msg: str):
    elapsed = time.time() - _T0
    print(f"[{elapsed:7.1f}s] {msg}", file=sys.stderr, flush=True)


def hf_download(remote_path: str, local_path: Path) -> bool:
    if local_path.is_file():
        return True
    local_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"  <- HF: {remote_path}  ->  {local_path.relative_to(DATASRC)}")
    cached = hf_hub_download(repo_id=HF_REPO, filename=remote_path, repo_type="dataset",
                             cache_dir=str(ROOT / ".hf_cache"))
    shutil.copy2(cached, local_path)
    return True


def parse_hf_index(files):
    ag, weather, usda = defaultdict(dict), defaultdict(dict), {}
    for path in files:
        parts = path.split("/")
        if path.startswith("Sentinel-2 Imagery/data/AG/") and path.endswith(".h5"):
            year, state = int(parts[3]), parts[4].upper()
            end = parts[-1].rsplit("_", 1)[-1].replace(".h5", "")[-5:]
            if end in ("06-30", "09-30"):
                ag[(year, state)][end] = path
        elif path.startswith("WRF-HRRR Computed Dataset/data/") and path.endswith(".csv"):
            year, state = int(parts[2]), parts[3].upper()
            month = int(parts[-1].rsplit("-", 1)[-1].replace(".csv", ""))
            weather[(year, state)][month] = path
        elif path.startswith("USDA Crop Dataset/Corn/") and path.endswith(".csv"):
            usda[int(parts[-2])] = path
    return ag, weather, usda


def weather_path(year: int, state: str, month: int) -> Path:
    """统一 weather 路径: weather/{year}/{state}/HRRR_{fips}_{state}_{year}-{month:02d}.csv"""
    fips = {v: k for k, v in ANSI_TO_USPS.items()}.get(state, "XX")
    return WEATHER_DIR / str(year) / state / f"HRRR_{fips}_{state}_{year}-{month:02d}.csv"


def load_soil() -> dict:
    log("加载本地土壤数据 ...")
    result = {}
    for p in (SOIL_DIR / "county_soil.json", ADDED_SOIL):
        if p.is_file():
            data = json.load(p.open(encoding="utf-8"))
            result.update(data)
            log(f"  {p.name}: {len(data)} 县")
    log(f"  土壤总计: {len(result)} 县")
    return result


def soil_ok(soil: dict, fips: str) -> bool:
    row = soil.get(fips)
    if row is None:
        return False
    try:
        return all(np.isfinite(float(row[name])) for name in SOIL_COLS)
    except (KeyError, TypeError, ValueError):
        return False


def load_checkpoint() -> set:
    if CHECKPOINT.is_file():
        ck = json.load(CHECKPOINT.open(encoding="utf-8"))
        return {tuple(x) for x in ck.get("completed_state_years", [])}
    return set()


def save_checkpoint(state: str, year: int):
    ck = {"completed_state_years": []}
    if CHECKPOINT.is_file():
        ck = json.load(CHECKPOINT.open(encoding="utf-8"))
    existing = {tuple(x) for x in ck.get("completed_state_years", [])}
    existing.add((year, state))
    ck["completed_state_years"] = sorted([[y, s] for y, s in existing])
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(ck, ensure_ascii=False, indent=2))


def scan_disk_state(years: list, ag_idx: dict, weather_idx: dict):
    """枚举所有 (year,state) 组合，检查本地磁盘数据完整性"""
    log("扫描本地磁盘数据状态 ...")
    states = set()
    for (y, s) in set(list(ag_idx.keys()) + list(weather_idx.keys())):
        if y in years:
            states.add((y, s))
    log(f"  候选 (year,state) 组合: {len(states)}")
    
    complete = []   # AG+Weather+USDA 全部在本地
    need_ag = []    # 缺 AG
    need_weather = []  # 缺 Weather
    need_both = []  # 两者都缺
    no_usda = []    # USDA 文件不存在

    for year, state in sorted(states):
        usda_file = LABEL_DIR / f"USDA_Corn_County_{year}.csv"
        if not usda_file.is_file():
            no_usda.append((year, state))
            continue

        # AG 检查
        ag_exist = True
        if (year, state) in ag_idx:
            for end in ("06-30", "09-30"):
                start = "04-01" if end == "06-30" else "07-01"
                fips_key = {v: k for k, v in ANSI_TO_USPS.items()}.get(state, "XX")
                local = AG_DIR / str(year) / state / f"Agriculture_{fips_key}_{state}_{year}-{start}_{year}-{end}.h5"
                if not local.is_file():
                    ag_exist = False
                    break
        else:
            ag_exist = False

        # Weather 检查
        weather_exist = True
        if (year, state) in weather_idx:
            for month in MONTHS:
                if not weather_path(year, state, month).is_file():
                    weather_exist = False
                    break
        else:
            weather_exist = False

        if ag_exist and weather_exist:
            complete.append((year, state))
        elif ag_exist and not weather_exist:
            need_weather.append((year, state))
        elif not ag_exist and weather_exist:
            need_ag.append((year, state))
        else:
            need_both.append((year, state))

    log(f"  完整: {len(complete)} 州年, 缺Weather:{len(need_weather)}, 缺AG:{len(need_ag)}, 都缺:{len(need_both)}")
    return complete, need_ag, need_weather, need_both, no_usda


def process_state_year(year: int, state: str, ag_idx: dict, weather_idx: dict, soil: dict,
                       ag_exist: bool, weather_exist: bool):
    """处理一个 (year,state) 所有玉米县的下载"""
    usda_file = LABEL_DIR / f"USDA_Corn_County_{year}.csv"
    frame = pd.read_csv(usda_file)
    frame = frame[(frame["commodity_desc"] == "CORN") & (frame["reference_period_desc"] == "YEAR")]

    fips_map = {}
    for _, row in frame.iterrows():
        try:
            fips = str(int(row["state_ansi"])).zfill(2) + str(int(row["county_ansi"])).zfill(3)
            yield_val = float(str(row["YIELD, MEASURED IN BU / ACRE"]).replace(",", ""))
            if ANSI_TO_USPS.get(fips[:2]) == state:
                fips_map[fips] = yield_val
        except (TypeError, ValueError, KeyError):
            continue

    if not fips_map:
        return 0, 0, 0

    log(f"  [{year} {state}] {len(fips_map)} 个玉米县")

    # 1. 下载 AG
    if not ag_exist and (year, state) in ag_idx:
        for end in ("06-30", "09-30"):
            remote = ag_idx[(year, state)][end]
            start = "04-01" if end == "06-30" else "07-01"
            fips_key = {v: k for k, v in ANSI_TO_USPS.items()}.get(state, "XX")
            local = AG_DIR / str(year) / state / f"Agriculture_{fips_key}_{state}_{year}-{start}_{year}-{end}.h5"
            hf_download(remote, local)

    # 2. 下载 Weather
    if not weather_exist and (year, state) in weather_idx:
        for month in MONTHS:
            remote = weather_idx[(year, state)][month]
            local = weather_path(year, state, month)
            hf_download(remote, local)

    # 3. 逐县处理土壤
    downloaded = 0
    soil_failed = 0
    soil_existing = 0

    for fips, yield_val in sorted(fips_map.items()):
        pending_soil = None
        if soil_ok(soil, fips):
            soil_existing += 1
            downloaded += 1
            continue

        try:
            s = aggregate_gssurgo(query_gssurgo(fips))
            if s is not None:
                soil[fips] = s
                pending_soil = s
            else:
                soil_failed += 1
                continue
        except Exception:
            soil_failed += 1
            continue

        if pending_soil is not None:
            added = {}
            if ADDED_SOIL.is_file():
                added = json.load(ADDED_SOIL.open(encoding="utf-8"))
            added[fips] = pending_soil
            ADDED_SOIL.parent.mkdir(parents=True, exist_ok=True)
            ADDED_SOIL.write_text(json.dumps(added, ensure_ascii=False, indent=2))

        downloaded += 1

    if soil_failed > 0:
        log(f"    [土壤] 已有:{soil_existing} 新查询:{downloaded - soil_existing} 失败:{soil_failed}")
    return downloaded, soil_existing, soil_failed


def main():
    log(f"===== CropNet 全美数据下载 =====")
    log(f"年份: {YEARS}")

    completed = load_checkpoint()
    if completed:
        log(f"断点: 已完成 {len(completed)} 个州年")

    log("从 HF 获取文件列表 ...")
    hf_files = HfApi().list_repo_files(repo_id=HF_REPO, repo_type="dataset")
    log(f"  HF 共 {len(hf_files)} 文件")

    ag_idx, weather_idx, usda_idx = parse_hf_index(hf_files)
    log(f"  AG 遥感: {len(ag_idx)} (year,state) × {len({s for _,s in ag_idx})} 州")
    log(f"  Weather:  {len(weather_idx)} (year,state) × {len({s for _,s in weather_idx})} 州")

    # 下载 USDA 产量文件
    for year in YEARS:
        if year in usda_idx:
            hf_download(usda_idx[year], LABEL_DIR / f"USDA_Corn_County_{year}.csv")

    soil = load_soil()

    # 扫描磁盘状态
    complete, need_ag, need_weather, need_both, no_usda = scan_disk_state(YEARS, ag_idx, weather_idx)

    # 构建待处理列表
    all_states = set()
    for y in YEARS:
        for (yy, ss) in ag_idx:
            if yy == y:
                all_states.add((y, ss))
        for (yy, ss) in weather_idx:
            if yy == y:
                all_states.add((y, ss))

    need_download = set(need_weather + need_ag + need_both)
    to_process = [(y, s) for (y, s) in sorted(all_states) if (y, s) in need_download and (y, s) not in completed]

    if not to_process:
        log("所有州年数据已就绪！")
        return

    log(f"本次需处理: {len(to_process)} 州年")

    for year_idx, (year, state) in enumerate(to_process, 1):
        if (year, state) in completed:
            continue

        ag_exist = (year, state) not in {x for x in need_ag + need_both}
        weather_exist = (year, state) not in {x for x in need_weather + need_both}
        need_what = "AG" if not ag_exist and not weather_exist else ("AG" if not ag_exist else "Weather")
        log(f"[{year_idx}/{len(to_process)}] {year} {state} (缺: {need_what})")

        dl, ex, fail = process_state_year(year, state, ag_idx, weather_idx, soil, ag_exist, weather_exist)
        save_checkpoint(state, year)
        log(f"  -> 完成: {dl} 县 (其中土壤已有:{ex} 失败:{fail})")

    log("===== 全部完成 =====")


if __name__ == "__main__":
    main()