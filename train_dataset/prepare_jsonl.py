'''
将 cropnet_dataset (USDA 县级玉米数据 + WRF-HRRR 气象数据)
处理成 JSONL 格式，县级粒度，与 short_seq_nonorm.ipynb 的输出格式对齐。

处理逻辑：
  1. 遍历 USDA 每个 county（年份 × 县）
  2. 对每个 county，从 WRF-HRRR 加载对应 FIPS 的气象数据
  3. 按原始日粒度输出（不重采样）
  4. 输出 JSONL：每行一个样本

用法：
  conda activate product
  python prepare_jsonl.py

输出：
  cropnet_dataset/dataset.jsonl
'''
import os
import json
import sys
import pandas as pd
import numpy as np
from glob import glob

# ============================================================
# 1. 路径配置（基于脚本所在目录，Windows/WSL 通用）
# ============================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)          # cropnet_model/
# 源数据已迁移到 DataSrc/ 下(2026-08 重构)
DATA_DIR = os.path.join(PROJECT_DIR, "DataSrc", "cropnet_dataset", "data")
USDA_DIR = os.path.join(DATA_DIR, "usda_corn")
WEATHER_DIR = os.path.join(DATA_DIR, "weather")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "dataset.jsonl")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# ============================================================
# 2. 特征配置 — 使用 WRF-HRRR 原始列名
# ============================================================
WRF_COLS = [
    "Avg Temperature (K)",
    "Max Temperature (K)",
    "Min Temperature (K)",
    "Precipitation (kg m**-2)",
    "Relative Humidity (%)",
    "Wind Gust (m s**-1)",
    "Wind Speed (m s**-1)",
    "U Component of Wind (m s**-1)",
    "V Component of Wind (m s**-1)",
    "Downward Shortwave Radiation Flux (W m**-2)",
    "Vapor Pressure Deficit (kPa)",
]

# 州缩写 → 全称小写+下划线（用于输出）
USPS_MAP = {
    "AL": "alabama", "AR": "arkansas", "CA": "california", "CO": "colorado",
    "DE": "delaware", "GA": "georgia", "IA": "iowa", "ID": "idaho",
    "IL": "illinois", "IN": "indiana", "KS": "kansas", "KY": "kentucky",
    "LA": "louisiana", "MD": "maryland", "MI": "michigan", "MN": "minnesota",
    "MO": "missouri", "MS": "mississippi", "MT": "montana", "NC": "north_carolina",
    "ND": "north_dakota", "NE": "nebraska", "NJ": "new_jersey", "NM": "new_mexico",
    "NY": "new_york", "OH": "ohio", "OK": "oklahoma", "PA": "pennsylvania",
    "SC": "south_carolina", "SD": "south_dakota", "TN": "tennessee",
    "TX": "texas", "VA": "virginia", "WA": "washington",
    "WI": "wisconsin", "WV": "west_virginia", "WY": "wyoming",
}
USPS_TO_FULL = {k: v for k, v in USPS_MAP.items()}

# USDA state_name → 缩写（反向查找用）
STATE_TO_USPS = {v.upper(): k for k, v in USPS_MAP.items()}


def _load_state_weather(year, state_abbr):
    """
    加载某年某州全部气象 CSV，返回所有县逐日数据。

    返回 DataFrame，含 [date, FIPS Code, Grid Index, ...气象列]
    """
    state_path = os.path.join(WEATHER_DIR, str(year), state_abbr)
    if not os.path.isdir(state_path):
        return None

    csv_files = sorted(glob(os.path.join(state_path, "*.csv")))
    if not csv_files:
        return None

    monthly = []
    for f in csv_files:
        try:
            monthly.append(pd.read_csv(f))
        except Exception as e:
            print(f"    [警告] 读取失败 {os.path.basename(f)}: {e}")

    if not monthly:
        return None

    df = pd.concat(monthly, ignore_index=True)

    # 检查是否有 Daily/Monthly 列（有些空文件没有该列）
    if "Daily/Monthly" not in df.columns:
        print(f"    [警告] {state_abbr} {year} 数据格式异常，无 Daily/Monthly 列")
        return None

    df = df[df["Daily/Monthly"] == "Daily"].copy()
    if df.empty:
        return None

    # 检查是否有 Month 列
    if "Month" not in df.columns:
        print(f"    [警告] {state_abbr} {year} 数据无 Month 列")
        return None

    # 只保留 3月-11月（玉米生长季）
    df = df[df["Month"].isin(range(3, 12))].copy()
    if df.empty:
        return None

    # 统一 FIPS 为 5位字符串
    df["FIPS Code"] = df["FIPS Code"].astype(str).str.zfill(5)
    # 构建日期
    df["date"] = pd.to_datetime(
        df[["Year", "Month", "Day"]].astype(int).rename(
            columns={"Year": "year", "Month": "month", "Day": "day"}
        )
    )
    return df


def _county_daily_series(county_df, src_cols):
    """
    对一个 county 的逐日数据（多 Grid 平均后），按日期排序返回各列时间序列和月份。

    county_df : DataFrame with [date, Grid Index, ...气象列]
    src_cols  : list of 气象列名

    返回 dict，含各气象列 list + "month" list。
    """
    daily_avg = county_df.groupby("date")[src_cols].mean().reset_index()
    daily_avg = daily_avg.sort_values("date")

    # 按行剔除 NaN（任一列为 NaN 则整行删除），确保各列长度一致
    daily_avg = daily_avg.dropna(subset=src_cols).reset_index(drop=True)

    result = {}
    for col in src_cols:
        result[col] = daily_avg[col].tolist()
    result["month"] = daily_avg["date"].dt.month.tolist()
    return result


def process_all():
    print("=" * 60)
    print("cropnet_dataset → JSONL (县级粒度)")
    print("=" * 60)

    # ---- 1. 读取 USDA 数据 ----
    print("\n[1/3] 读取 USDA 县级产量数据...")

    # state_ansi 数字 → state 缩写（USDA用ANSI码，WRF用USPS缩写）
    # USDA CSV 里有 state_name，我们用它来映射成 usps 缩写
    usda_rows = []
    csv_files = sorted(glob(os.path.join(USDA_DIR, "USDA_Corn_County_*.csv")))
    for fpath in csv_files:
        year = int(os.path.basename(fpath).split("_")[-1].replace(".csv", ""))
        df = pd.read_csv(fpath)
        df = df[(df["commodity_desc"] == "CORN") & (df["reference_period_desc"] == "YEAR")].copy()
        df["Year"] = year
        df["FIPS"] = df["state_ansi"].astype(str).str.zfill(2) + \
                     df["county_ansi"].astype(str).str.zfill(3)
        # state_name → 小写_下划线
        df["state"] = df["state_name"].str.strip().str.lower().str.replace(" ", "_")
        # state_name → USPS 缩写
        df["state_abbr"] = df["state_name"].str.strip().str.upper().map(STATE_TO_USPS)
        # 产量转数值
        df["prod"] = pd.to_numeric(
            df["PRODUCTION, MEASURED IN BU"].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )
        df["yield_"] = pd.to_numeric(
            df["YIELD, MEASURED IN BU / ACRE"].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )
        # 县名
        df["county"] = df["county_name"].str.strip().str.lower().str.replace(" ", "_")
        usda_rows.append(df)

    usda_all = pd.concat(usda_rows, ignore_index=True)
    # 移除无缩写或无效产量的行
    usda_all = usda_all.dropna(subset=["state_abbr", "prod", "yield_"])
    print(f"  USDA 总行数: {len(usda_all)}")
    print(f"  年份: {sorted(usda_all['Year'].unique())}")
    print(f"  Counties: {usda_all['FIPS'].nunique()}")

    # ---- 2. 逐 county 处理气象 ----
    print("\n[2/3] 逐 county 加载 WRF-HRRR 气象...")

    # 按 (year, state_abbr) 分组，一次加载全州气象再按 FIPS 过滤（避免重复IO）
    usda_grouped = usda_all.groupby(["Year", "state_abbr"])

    all_samples = []
    total = len(usda_grouped)
    processed = 0

    for (year, state_abbr), group in usda_grouped:
        processed += 1
        if processed % 20 == 0 or processed == total:
            print(f"  进度: {processed}/{total} (year-state 组合), 已生成 {len(all_samples)} 样本")

        # 加载该州该年全部气象
        state_weather = _load_state_weather(year, state_abbr)
        if state_weather is None:
            continue

        fips_list = group["FIPS"].unique()
        # 过滤出该州所有目标 county 的气象
        weather_filtered = state_weather[state_weather["FIPS Code"].isin(fips_list)]

        for _, row in group.iterrows():
            fips = row["FIPS"]
            county_df = weather_filtered[weather_filtered["FIPS Code"] == fips]
            if county_df.empty:
                continue

            # 原始日数据
            resampled = _county_daily_series(county_df, WRF_COLS)

            # 检查是否有有效数据
            if not resampled or len(resampled[WRF_COLS[0]]) == 0:
                continue

            # 构建样本(目标字段仅保留单产 yield_per_acre,不再输出总产量 label)
            sample = {
                "Year": int(row["Year"]),
                "State": row["state"],
                "FIPS": fips,
                "County": row["county"],
                "yield_per_acre": float(row["yield_"]),
            }
            # 各特征长度对齐（含 month）
            lengths = [len(v) for v in resampled.values()]
            min_len = min(lengths)
            for col in WRF_COLS:
                sample[col] = resampled[col][:min_len]
            sample["month"] = resampled["month"][:min_len]
            sample["l_enc"] = min_len

            all_samples.append(sample)

    # ---- 3. Merge 静态土壤特征 ----
    print("\n[3/4] Merge 土壤静态特征...")
    soil_df = pd.read_csv(os.path.join(SCRIPT_DIR, "us_state_soil.csv"))
    soil_map = soil_df.set_index("State")[["carbon_bucket", "ph_bucket"]].to_dict("index")

    matched = 0
    missing_states = set()
    for sample in all_samples:
        state = sample["State"]
        if state in soil_map:
            sample["carbon_bucket"] = soil_map[state]["carbon_bucket"]
            sample["ph_bucket"] = soil_map[state]["ph_bucket"]
            matched += 1
        else:
            missing_states.add(state)

    if missing_states:
        print(f"  [警告] {len(missing_states)} 个州无土壤数据: {sorted(missing_states)}")

    print(f"\n[4/4] 写入 JSONL...")
    print(f"  总样本数: {len(all_samples)}, 含土壤特征: {matched}")

    if not all_samples:
        print("[错误] 无样本生成，退出")
        sys.exit(1)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # 统计
    years = sorted(set(s["Year"] for s in all_samples))
    states = sorted(set(s["State"] for s in all_samples))
    fips_count = len(set(s["FIPS"] for s in all_samples))
    print(f"  年份: {years}")
    print(f"  州数: {len(states)}")
    print(f"  Counties: {fips_count}")
    lens = [s["l_enc"] for s in all_samples]
    print(f"  序列长度: min={min(lens)}, max={max(lens)}, mean={np.mean(lens):.1f}")
    print(f"  输出: {OUTPUT_PATH}")

    # 示例
    print(f"\n  示例:")
    ex = all_samples[0]
    ex_show = {k: (v if not isinstance(v, list) or len(v) <= 4 else v[:4] + [f"...({len(v)}步)"])
               for k, v in ex.items()}
    print(f"  {json.dumps(ex_show, ensure_ascii=False, indent=2)[:600]}")

    print("\n完成!")


if __name__ == "__main__":
    process_all()
