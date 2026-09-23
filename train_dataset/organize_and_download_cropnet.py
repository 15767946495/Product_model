"""统一整理并增量下载 CropNet 数据。

目录：
  /data/raid0/hqx/Product_model_runtime/DataSrc/sentinel
  /data/raid0/hqx/Product_model_runtime/DataSrc/weather
  /data/raid0/hqx/Product_model_runtime/DataSrc/label
  /data/raid0/hqx/Product_model_runtime/DataSrc/soil

来源优先级：统一目录已有 -> 旧本地缓存 -> Hugging Face -> USDA SDA（土壤）。
任何县年缺少产量、完整气象、q2/q3 遥感、完整 gSSURGO 或网格不匹配，均跳过该县年。

先整理现有数据：
  conda run -n product python train_dataset/organize_and_download_cropnet.py --migrate-only

审计并下载：
  conda run -n product python train_dataset/organize_and_download_cropnet.py --download
"""

from __future__ import annotations

import argparse
import json
import os
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
    AG_DATES, ANSI_TO_USPS, MONTHS, SOIL_COLS,
    WEATHER_COLS, aggregate_gssurgo, query_gssurgo,
)

os.environ.setdefault("PYTHONUNBUFFERED", "1")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, "reconfigure") else None

ROOT = Path("/data/raid0/hqx/Product_model_runtime")
DATASRC = ROOT / "DataSrc"
SENTINEL = DATASRC / "sentinel"
SENTINEL_SOURCE = SENTINEL / "source" / "AG"
WEATHER = DATASRC / "weather"
LABEL = DATASRC / "label"
SOIL = DATASRC / "soil"
OLD_SENTINEL = DATASRC / "mmst_vit" / "county" / "AG"
OLD_WEATHER = DATASRC / "cropnet_dataset" / "data" / "weather"
OLD_LABEL = DATASRC / "cropnet_dataset" / "data" / "usda_corn"
OLD_SOIL = DATASRC / "soil_dataset" / "county_soil.json"
ADDED_SOIL = SOIL / "county_soil_gssurgo_added.json"
OSS_MANIFEST = Path("/data/hqx/myself/Product_model/manifests/sentinel_urls.jsonl")
HF_REPO = "CropNet/CropNet"
REPORT = ROOT / "cropnet_unified_audit.jsonl"

_T0 = time.time()


def log(msg: str):
    elapsed = time.time() - _T0
    print(f"[{elapsed:7.1f}s] {msg}", flush=True)


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return "exists"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def migrate_existing():
    log("步骤1: 开始整理旧本地缓存...")
    counts = Counter()
    for src_dir, dst_dir, pattern_name in [
        (OLD_SENTINEL, SENTINEL, "遥感"),
        (OLD_WEATHER, WEATHER, "气象"),
    ]:
        if not src_dir.exists():
            log(f"  跳过 {pattern_name}（源目录不存在: {src_dir}）")
            continue
        files = list(src_dir.rglob("*.h5")) if pattern_name == "遥感" else list(src_dir.rglob("*.csv"))
        log(f"  扫描 {pattern_name} 旧缓存: {len(files)} 文件")
        for source in files:
            rel = source.relative_to(src_dir)
            counts[link_or_copy(source, dst_dir / rel)] += 1
    for source in OLD_LABEL.glob("USDA_Corn_County_*.csv"):
        counts[link_or_copy(source, LABEL / source.name)] += 1
    if OLD_SOIL.is_file():
        counts[link_or_copy(OLD_SOIL, SOIL / "county_soil.json")] += 1
    log(f"步骤1 完成: {dict(counts)}")


def load_oss():
    log("步骤2: 加载 OSS manifest...")
    result = defaultdict(dict)
    if not OSS_MANIFEST.is_file():
        log("  未找到 OSS manifest，跳过")
        return result
    lines = [line for line in OSS_MANIFEST.open(encoding="utf-8") if line.strip()]
    for line in lines:
        item = json.loads(line)
        state = str(item["state"]).upper(); year = int(item["year"])
        end = item["path"].rstrip(".h5").split("_")[-1][-5:]
        if end in ("06-30", "09-30"):
            for fips in item.get("fips", []):
                result[(year, state, str(fips).zfill(5))][end] = item
    log(f"步骤2 完成: OSS 条目 {len(lines)} 条")
    return result


def parse_hf(files):
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


def soil_ok(soil, fips):
    row = soil.get(fips)
    if row is None:
        return False
    try:
        return all(np.isfinite(float(row[name])) for name in SOIL_COLS)
    except (KeyError, TypeError, ValueError):
        return False


def local_ag_pairs():
    log("步骤5: 扫描本地已下载的遥感数据 (local_ag_pairs)...")
    files = defaultdict(set)
    for root in (SENTINEL_SOURCE, SENTINEL, OLD_SENTINEL):
        if not root.exists():
            continue
        h5_files = list(root.rglob("*.h5"))
        log(f"  扫描 {root}: {len(h5_files)} 个 .h5 文件")
        for path in h5_files:
            year = next((int(x) for x in path.parts if x.isdigit() and len(x) == 4), None)
            if year is None:
                continue
            end = path.name.rsplit("_", 1)[-1].replace(".h5", "")[-5:]
            if end in ("06-30", "09-30"):
                try:
                    with h5py.File(path, "r") as handle:
                        fips_list = list(handle.keys())
                except OSError:
                    continue
                for fips in fips_list:
                    if len(str(fips)) == 5 and str(fips).isdigit():
                        files[(year, str(fips))].add(end)
    complete = {key for key, value in files.items() if value == {"06-30", "09-30"}}
    log(f"步骤5 完成: 完整(含q2+q3)县年数 = {len(complete)}")
    return complete


def load_unified_soil():
    log("步骤4: 加载本地土壤数据...")
    result = {}
    for path in (SOIL / "county_soil.json", SOIL / "county_soil_gssurgo_added.json"):
        if path.is_file():
            data = json.load(path.open(encoding="utf-8"))
            result.update(data)
            log(f"  加载 {path.name}: {len(data)} 个县")
    log(f"步骤4 完成: 土壤覆盖 {len(result)} 个县")
    return result


def hf_download(path: str, target: Path, download: bool):
    if target.is_file():
        return True
    if not download:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    log(f"    [下载HF] {target.relative_to(DATASRC)}")
    cached = hf_hub_download(
        repo_id=HF_REPO, filename=path, repo_type="dataset",
        cache_dir=str(ROOT / ".hf_cache"),
    )
    shutil.copy2(cached, target)
    return True


def oss_download(item, target: Path, download: bool):
    if target.is_file():
        return True
    if not download:
        return False
    import urllib.request
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(item["url"], timeout=180) as response, target.open("wb") as output:
        shutil.copyfileobj(response, output)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2017,2018,2019,2020,2021,2022")
    parser.add_argument("--states", default="all")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--migrate-only", action="store_true")
    parser.add_argument("--report", default=str(REPORT))
    args = parser.parse_args()

    log(f"启动: years={args.years}, states={args.states}, download={args.download}")

    migrate_existing()
    if args.migrate_only:
        log("--migrate-only 模式，完成退出")
        return

    years = {int(x) for x in args.years.split(",") if x.strip()}
    requested = None if args.states.lower() == "all" else {x.strip().upper() for x in args.states.split(",")}

    oss = load_oss()

    log("步骤3: 列取 HF CropNet 文件列表 (HfApi.list_repo_files)...")
    hf_files = HfApi().list_repo_files(repo_id=HF_REPO, repo_type="dataset")
    log(f"步骤3 完成: HF 共 {len(hf_files)} 个文件")
    log("步骤3b: 解析 HF 文件索引...")
    hf_ag, hf_weather, hf_usda = parse_hf(hf_files)
    log(f"  HF AG 遥感 (year,state) 覆盖率: {len(hf_ag)} 个 (year,state), "
        f"{len({s for _,s in hf_ag})} 个州")
    log(f"  HF Weather 气象: {len(hf_weather)} 个 (year,state), "
        f"{len({s for _,s in hf_weather})} 个州")
    log(f"  HF USDA 产量: {len(hf_usda)} 个年份")

    soil = load_unified_soil()
    local = local_ag_pairs()

    skipped_stats = Counter()
    eligible_count = 0
    download_count = 0
    report = []

    for year_idx, year in enumerate(sorted(years), 1):
        if year not in hf_usda:
            log(f"[{year}] HF 无 USDA 数据，跳过")
            continue

        label = LABEL / "hf" / f"USDA_Corn_County_{year}.csv"
        hf_download(hf_usda[year], label, True)
        log(f"[{year}] 读取 USDA 产量文件: {label.name}")
        frame = pd.read_csv(label)
        frame = frame[(frame["commodity_desc"] == "CORN") & (frame["reference_period_desc"] == "YEAR")]
        rows = list(frame.iterrows())
        year_total = len(rows)
        log(f"[{year}] 共 {year_total} 条 CORN+YEAR 记录")

        year_skipped = Counter()
        year_downloaded = 0
        for row_index, (_, row) in enumerate(rows, 1):
            try:
                fips = str(int(row["state_ansi"])).zfill(2) + str(int(row["county_ansi"])).zfill(3)
                yield_value = float(str(row["YIELD, MEASURED IN BU / ACRE"]).replace(",", ""))
            except (TypeError, ValueError):
                continue
            state = ANSI_TO_USPS.get(fips[:2])
            if requested and state not in requested:
                continue
            key = (year, fips); reasons = []

            if row_index == 1 or row_index % 500 == 0:
                log(f"  [{year}] 进度 {row_index}/{year_total} fips={fips} state={state}"
                    f" (跳过{sum(year_skipped.values())}/下载{year_downloaded})")

            if key in local:
                reasons.append("already_downloaded")

            pending_soil = None
            if not soil_ok(soil, fips):
                if args.download:
                    try:
                        added = aggregate_gssurgo(query_gssurgo(fips))
                    except Exception as exc:
                        log(f"    [土壤查询失败] fips={fips}: {exc}")
                        added = None
                    if added is not None:
                        soil[fips] = added
                        pending_soil = added
                    else:
                        reasons.append("missing_gssurgo_soil")
                else:
                    reasons.append("missing_gssurgo_soil")

            source_key = (year, state, fips)
            if source_key not in oss and (year, state) not in hf_ag:
                reasons.append("missing_ag_source")
            if (year, state) not in hf_weather:
                reasons.append("missing_weather_source")

            if reasons:
                for r in reasons:
                    year_skipped[r] += 1
                report.append({"year": year, "fips": fips, "status": "skipped", "reasons": reasons})
                continue

            downloaded = []
            weather_ok = True
            for month in MONTHS:
                rel = hf_weather[(year, state)][month]
                if not hf_download(rel, WEATHER / rel, args.download):
                    year_skipped["weather_not_downloaded"] += 1
                    weather_ok = False
                else:
                    downloaded.append(WEATHER / rel)

            ag_ok = True
            for end in ("06-30", "09-30"):
                target = SENTINEL_SOURCE / str(year) / state / f"Agriculture_{fips[:2]}_{state}_{year}-{'04-01' if end == '06-30' else '07-01'}_{year}-{end}.h5"
                if end in oss.get(source_key, {}):
                    ok = oss_download(oss[source_key][end], target, args.download)
                elif end in hf_ag.get((year, state), {}):
                    ok = hf_download(hf_ag[(year, state)][end], target, args.download)
                else:
                    ok = False
                if not ok:
                    year_skipped[f"ag_{end}_not_downloaded"] += 1
                    ag_ok = False
                else:
                    downloaded.append(target)

            if not weather_ok or not ag_ok:
                reasons_diff = [r for r in ["weather_not_downloaded", "ag_06-30_not_downloaded", "ag_09-30_not_downloaded"]
                                if year_skipped.get(r) and not reasons]
                report.append({"year": year, "fips": fips, "status": "skipped",
                               "reasons": reasons + reasons_diff[-1:]})
                continue

            if pending_soil is not None:
                existing_added = {}
                if ADDED_SOIL.is_file():
                    existing_added = json.load(ADDED_SOIL.open(encoding="utf-8"))
                existing_added[fips] = pending_soil
                ADDED_SOIL.parent.mkdir(parents=True, exist_ok=True)
                ADDED_SOIL.write_text(json.dumps(existing_added, ensure_ascii=False, indent=2))

            local.add(key)
            year_downloaded += 1
            download_count += 1
            status = "downloaded" if args.download else "eligible"
            report.append({"year": year, "fips": fips, "yield": yield_value, "status": status})

        log(f"[{year}] 完成: 跳过={dict(year_skipped)}, 下载={year_downloaded}")

    args.report = Path(args.report); args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w") as handle:
        for row in report:
            handle.write(json.dumps(row) + "\n")

    status_counts = Counter(x["status"] for x in report)
    log(f"==================== 审计完成 ====================")
    log(f"总计候选: {len(report)}")
    log(f"状态分布: {dict(status_counts)}")
    log(f"报告: {args.report}")


if __name__ == "__main__":
    main()