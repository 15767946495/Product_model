"""增量下载完整 CropNet 县年数据，并补充 USDA gSSURGO 土壤。

唯一入口，合并：
  - Hugging Face CropNet/CropNet 的 USDA Corn、WRF-HRRR、Sentinel-2 AG；
  - USDA NRCS Soil Data Access 的 gSSURGO 县级土壤。

严格规则：
1. 先扫描 /data/raid0/hqx/Product_model_runtime 中已有的县年文件；
2. 已有完整县年直接跳过；
3. 新县年必须同时具备 USDA、WRF、AG、gSSURGO；
4. 气象网格必须全部能按坐标匹配到遥感网格；
5. 任一源缺失或网格不匹配，该县年不下载任何文件；
6. 土壤是县级数据，一个 FIPS 的 gSSURGO 记录供该县所有年份复用。

默认只审计。下载使用：
  conda run -n product python train_dataset/download_complete_cropnet_hf_with_soil.py --download
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download


REPO_ID = "CropNet/CropNet"
REPO_TYPE = "dataset"
ROOT = Path("/data/raid0/hqx/Product_model_runtime")
HF_ROOT = ROOT / "DataSrc" / "cropnet_hf"
EXISTING_AG_ROOT = ROOT / "DataSrc" / "mmst_vit" / "county" / "AG"
SOIL_ROOT = ROOT / "DataSrc" / "soil_dataset"
EXISTING_SOIL = SOIL_ROOT / "county_soil.json"
ADDED_SOIL = SOIL_ROOT / "county_soil_gssurgo_added.json"
REPORT = ROOT / "cropnet_hf_with_soil_audit.jsonl"
MONTHS = range(4, 10)
AG_DATES = {"04-01", "05-01", "06-01", "07-01", "08-01", "09-01"}
SOIL_COLS = ["clay_pct", "sand_pct", "silt_pct", "om_pct", "ph", "bulk_density", "awc"]
WEATHER_COLS = [
    "Avg Temperature (K)", "Max Temperature (K)", "Min Temperature (K)",
    "Precipitation (kg m**-2)", "Relative Humidity (%)", "Wind Gust (m s**-1)",
    "Wind Speed (m s**-1)", "U Component of Wind (m s**-1)",
    "V Component of Wind (m s**-1)",
    "Downward Shortwave Radiation Flux (W m**-2)",
    "Vapor Pressure Deficit (kPa)",
]
ANSI_TO_USPS = {
    "01": "AL", "05": "AR", "06": "CA", "08": "CO", "10": "DE",
    "13": "GA", "16": "ID", "17": "IL", "18": "IN", "19": "IA",
    "20": "KS", "21": "KY", "22": "LA", "24": "MD", "26": "MI",
    "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "42": "PA", "45": "SC", "46": "SD",
    "47": "TN", "48": "TX", "51": "VA", "53": "WA", "54": "WV",
    "55": "WI", "56": "WY",
}
SOIL_SDA = "https://SDMDataAccess.sc.egov.usda.gov/Tabular/post.rest"
CENSUS = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/State_County/MapServer/1/query"
WEATHER_CACHE = {}


def http_json(url, data=None, headers=None, timeout=180):
    request = urllib.request.Request(url, data=data, headers=headers or {"User-Agent": "CropNet-HF-gSSURGO/1.0"}, method="POST" if data else "GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def hf_files():
    return HfApi().list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE)


def parse_files(files):
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


def local_h5_pairs(root=HF_ROOT):
    files = defaultdict(set)
    patterns = ["Sentinel-2 Imagery/data/AG/*/*/*.h5", "*/*/*.h5"]
    paths = []
    for pattern in patterns:
        paths.extend(root.glob(pattern))
    for path in set(paths):
        parts = path.parts
        try:
            year = int(parts[-3]); fips = parts[-2]
        except ValueError:
            continue
        end = path.name.rsplit("_", 1)[-1].replace(".h5", "")[-5:]
        if end in ("06-30", "09-30"):
            files[(year, fips)].add(end)
    return {key for key, ends in files.items() if ends == {"06-30", "09-30"}}


def all_local_h5_pairs():
    return local_h5_pairs(HF_ROOT) | local_h5_pairs(EXISTING_AG_ROOT)


def existing_soil():
    result = {}
    for path in (EXISTING_SOIL, ADDED_SOIL):
        if path.is_file():
            result.update(json.load(path.open(encoding="utf-8")))
    return result


def soil_geometry_wkt(fips):
    payload = http_json(CENSUS + "?" + urllib.parse.urlencode({
        "where": f"GEOID='{fips}'", "outFields": "GEOID",
        "returnGeometry": "true", "f": "geojson",
    }))
    geometry = payload["features"][0]["geometry"]
    def ring(values):
        return "(" + ",".join(f"{x} {y}" for x, y in values) + ")"
    if geometry["type"] == "Polygon":
        return "POLYGON(" + ",".join(ring(r) for r in geometry["coordinates"]) + ")"
    return "MULTIPOLYGON(" + ",".join("(" + ",".join(ring(r) for r in p) + ")" for p in geometry["coordinates"]) + ")"


def query_gssurgo(fips):
    sql = f"""SELECT co.comppct_r,ch.hzdept_r,ch.hzdepb_r,ch.claytotal_r,ch.sandtotal_r,ch.silttotal_r,ch.om_r,ch.ph1to1h2o_r,ch.dbthirdbar_r,ch.awc_r FROM mapunit mu JOIN component co ON co.mukey=mu.mukey JOIN chorizon ch ON ch.cokey=co.cokey WHERE mu.mukey IN (SELECT mukey FROM SDA_Get_Mukey_from_intersection_with_Wkt('{soil_geometry_wkt(fips)}')) AND ch.hzdept_r < 30 AND ch.hzdepb_r > 0"""
    body = urllib.parse.urlencode({"query": sql, "format": "JSON+COLUMNNAME"}).encode()
    request = urllib.request.Request(
        SOIL_SDA, data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded",
                 "User-Agent": "CropNet-HF-gSSURGO/1.0"}, method="POST"
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = json.loads(response.read().decode())
    return payload.get("Table", payload) if isinstance(payload, dict) else payload


def aggregate_gssurgo(rows):
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    numeric = ["comppct_r", "hzdept_r", "hzdepb_r", "claytotal_r", "sandtotal_r",
               "silttotal_r", "om_r", "ph1to1h2o_r", "dbthirdbar_r", "awc_r"]
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["comppct_r", "hzdept_r", "hzdepb_r"])
    frame["thickness"] = (frame["hzdepb_r"].clip(upper=30) - frame["hzdept_r"].clip(upper=30)).clip(lower=0)
    frame = frame[frame["thickness"] > 0]
    if frame.empty:
        return None
    frame["weight"] = frame["comppct_r"].clip(lower=0) * frame["thickness"]
    mapping = {"clay_pct": "claytotal_r", "sand_pct": "sandtotal_r", "silt_pct": "silttotal_r",
               "om_pct": "om_r", "ph": "ph1to1h2o_r", "bulk_density": "dbthirdbar_r", "awc": "awc_r"}
    result = {}
    for target, source in mapping.items():
        valid = frame.dropna(subset=[source, "weight"])
        if valid.empty or float(valid["weight"].sum()) <= 0:
            return None
        result[target] = float((valid[source] * valid["weight"]).sum() / valid["weight"].sum())
    return result


def soil_complete(soil, fips):
    row = soil.get(fips)
    if row is None:
        return False
    for name in SOIL_COLS:
        try:
            if not np.isfinite(float(row.get(name))):
                return False
        except (TypeError, ValueError):
            return False
    return True


def hf_download(path):
    target = HF_ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        hf_hub_download(repo_id=REPO_ID, filename=path, repo_type="dataset",
                        cache_dir=str(ROOT / ".hf_cache"), local_dir=str(target.parent),
                        local_dir_use_symlinks=False)
    return target


def usda_rows(path, year):
    frame = pd.read_csv(path)
    frame = frame[(frame["commodity_desc"] == "CORN") & (frame["reference_period_desc"] == "YEAR")]
    out = []
    for _, row in frame.iterrows():
        try:
            value = float(str(row["YIELD, MEASURED IN BU / ACRE"]).replace(",", ""))
            fips = str(int(row["state_ansi"])).zfill(2) + str(int(row["county_ansi"])).zfill(3)
        except (TypeError, ValueError):
            continue
        out.append((fips, ANSI_TO_USPS.get(fips[:2]), value, year))
    return out


def weather_coords(paths, fips):
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frames.append(frame[frame["Daily/Monthly"] == "Daily"])
    frame = pd.concat(frames, ignore_index=True)
    frame["FIPS Code"] = pd.to_numeric(frame["FIPS Code"], errors="coerce").map(lambda x: "" if pd.isna(x) else str(int(x)).zfill(5))
    frame = frame[(frame["FIPS Code"] == fips) & frame["Month"].between(4, 9) & frame["Day"].between(1, 28)]
    if len(frame[["Year", "Month", "Day"]].drop_duplicates()) != 168 or any(x not in frame.columns for x in WEATHER_COLS):
        return None
    unique = frame.sort_values(["Grid Index", "Month", "Day"]).drop_duplicates("Grid Index")
    return np.stack([(unique["Lat (llcrnr)"] + unique["Lat (urcrnr)"]) / 2, (unique["Lon (llcrnr)"] + unique["Lon (urcrnr)"]) / 2], axis=1)


def geometry_wkt(fips):
    payload = http_json(CENSUS + "?" + urllib.parse.urlencode({"where": f"GEOID='{fips}'", "outFields": "GEOID", "returnGeometry": "true", "f": "geojson"}))
    geometry = payload["features"][0]["geometry"]
    def ring(values): return "(" + ",".join(f"{x} {y}" for x, y in values) + ")"
    if geometry["type"] == "Polygon": return "POLYGON(" + ",".join(ring(r) for r in geometry["coordinates"]) + ")"
    return "MULTIPOLYGON(" + ",".join("(" + ",".join(ring(r) for r in p) + ")" for p in geometry["coordinates"]) + ")"


def query_soil(fips):
    sql = f"""SELECT co.comppct_r,ch.hzdept_r,ch.hzdepb_r,ch.claytotal_r,ch.sandtotal_r,ch.silttotal_r,ch.om_r,ch.ph1to1h2o_r,ch.dbthirdbar_r,ch.awc_r FROM mapunit mu JOIN component co ON co.mukey=mu.mukey JOIN chorizon ch ON ch.cokey=co.cokey WHERE mu.mukey IN (SELECT mukey FROM SDA_Get_Mukey_from_intersection_with_Wkt('{geometry_wkt(fips)}')) AND ch.hzdept_r < 30 AND ch.hzdepb_r > 0"""
    body = urllib.parse.urlencode({"query": sql, "format": "JSON+COLUMNNAME"}).encode()
    payload = http_json(SOIL_SDA, data=body, headers={"Content-Type": "application/x-www-form-urlencoded", "User-Agent": "CropNet-HF-gSSURGO/1.0"})
    return payload.get("Table", payload) if isinstance(payload, dict) else payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", default="2017,2018,2019,2020,2021,2022")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--report", default=str(REPORT))
    args = parser.parse_args()
    years = {int(x) for x in args.years.split(",") if x.strip()}
    ag, weather, usda = parse_files(hf_files())
    soil = existing_soil()
    local = all_local_h5_pairs()
    added_soil = {}
    report = []
    for year in sorted(years):
        usda_path = hf_download(usda[year]) if args.download else HF_ROOT / usda[year]
        if not usda_path.is_file(): continue
        for fips, state, yield_value, _ in usda_rows(usda_path, year):
            key = (year, fips)
            reasons=[]
            if key in local: reasons.append("already_downloaded")
            if not soil_complete(soil, fips):
                if not args.download:
                    reasons.append("missing_gssurgo_soil")
                else:
                    try:
                        added = aggregate_gssurgo(query_soil(fips))
                    except Exception as error:  # noqa: BLE001
                        added = None
                        reasons.append(f"gssurgo_error:{type(error).__name__}")
                    if added is None:
                        reasons.append("missing_or_incomplete_gssurgo_soil")
                    else:
                        soil[fips] = added
                        added_soil[fips] = added
            if state not in {s for y,s in ag if y==year} or set(ag[(year,state)]) != {"06-30","09-30"}: reasons.append("missing_hf_ag")
            wk=(year,state)
            if wk not in weather or any(m not in weather[wk] for m in MONTHS): reasons.append("missing_hf_weather")
            if reasons:
                report.append({"year":year,"fips":fips,"status":"skipped","reasons":reasons})
                continue
            if not args.download:
                report.append({"year":year,"fips":fips,"status":"ready"}); continue
            weather_paths=[hf_download(weather[wk][m]) for m in MONTHS]
            wc=weather_coords(weather_paths,fips)
            if wc is None: report.append({"year":year,"fips":fips,"status":"skipped","reasons":["incomplete_weather"]}); continue
            ag_paths=[hf_download(ag[(year,state)][x]) for x in ("06-30","09-30")]
            with h5py.File(ag_paths[0],"r") as h:
                if fips not in h: report.append({"year":year,"fips":fips,"status":"skipped","reasons":["missing_ag_fips"]}); continue
                ac=np.asarray(h[fips]["2022-04-01" if year==2022 else next(iter(h[fips].keys()))]["coordinates"][:]).mean(1)
            if np.any(np.sqrt(((wc[:,None]-ac[None,:])**2).sum(-1)).min(1)>1e-3): report.append({"year":year,"fips":fips,"status":"skipped","reasons":["grid_mismatch"]}); continue
            local.add(key)
            report.append({"year":year,"fips":fips,"status":"downloaded"})
    if args.download and added_soil:
        ADDED_SOIL.parent.mkdir(parents=True, exist_ok=True)
        existing_added = {}
        if ADDED_SOIL.is_file():
            existing_added = json.load(ADDED_SOIL.open(encoding="utf-8"))
        existing_added.update(added_soil)
        with ADDED_SOIL.open("w", encoding="utf-8") as handle:
            json.dump(existing_added, handle, ensure_ascii=False, indent=2)
    Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    with open(args.report,"w") as h:
        for row in report: h.write(json.dumps(row)+"\n")
    print('候选:',len(report),'状态:',Counter(x['status'] for x in report),'报告:',args.report)

if __name__ == '__main__': main()
