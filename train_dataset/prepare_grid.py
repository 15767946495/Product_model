"""
将 WRF-HRRR 气象数据按「网格级」处理成二进制缓存 grid_cache.pt。

与 prepare_jsonl.py 的区别:
  - prepare_jsonl.py 把每县多个 9x9km 网格 `groupby("date").mean()` 平均成一条县序列。
  - 本脚本保留每个县的 G 个网格,每条 (县,年) 存 `(G,T,F)` 三维气象 + 网格经纬度,
    供模型做「逐特征网格注意力合并」后,再进入时序 VSN 特征筛选。

输出:
  - train_dataset/grid_cache.pt   (~1.3GB): {"version":2, "feat_names": WRF_COLS,
      "entries": [ None | {"feats":(G,T,F) f32, "coords":(G,2) f32, "month":(T,) long, "l_enc":int} ]}
    entries[i] 与 dataset.jsonl 第 i 行按索引对齐(顺序即缓存索引)。
  - train_dataset/grid_cache_meta.json: 版本、G 分布、与 jsonl l_enc 不一致数等。

用法:
  python prepare_grid.py
"""
import os
import json
import numpy as np
import pandas as pd
import torch

import prepare_jsonl
from prepare_jsonl import WRF_COLS, _load_state_weather, STATE_TO_USPS

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSONL_PATH = os.path.join(SCRIPT_DIR, "dataset.jsonl")
OUT_PATH = os.path.join(SCRIPT_DIR, "grid_cache.pt")
META_PATH = os.path.join(SCRIPT_DIR, "grid_cache_meta.json")


def _median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def build_entry(county_df, feats):
    """对一个县(单 FIPS)的全部网格,构建 (G,T,F) 网格级样本。"""
    cdf = county_df.copy()
    cdf["Grid Index"] = pd.to_numeric(
        cdf["Grid Index"].astype(str).str.strip().replace("N/A", None),
        errors="coerce",
    )
    cdf = cdf.dropna(subset=["Grid Index"])
    if cdf.empty:
        return None
    cdf["Grid Index"] = cdf["Grid Index"].astype(int)
    # 同一网格同一天只保留一行
    cdf = cdf.drop_duplicates(subset=["Grid Index", "date"])
    grids = sorted(cdf["Grid Index"].unique())
    G = len(grids)

    grid_tables = {}
    for g in grids:
        sub = cdf[cdf["Grid Index"] == g].set_index("date")[feats]
        grid_tables[g] = sub

    # 公共日期:所有网格、所有特征都有限(保证各网格 T 一致,便于密实张量)
    common = None
    for g in grids:
        valid = set(grid_tables[g].dropna().index)
        common = valid if common is None else (common & valid)
    common = sorted(common)
    T = len(common)
    if T == 0:
        return None

    F = len(feats)
    feats_arr = np.empty((G, T, F), dtype=np.float32)
    coords = np.empty((G, 2), dtype=np.float32)
    for gi, g in enumerate(grids):
        vals = grid_tables[g].loc[common].values.astype(np.float32)  # (T,F)
        feats_arr[gi] = vals
        row0 = cdf[cdf["Grid Index"] == g].iloc[0]
        coords[gi] = [float(row0["Lat (llcrnr)"]), float(row0["Lon (llcrnr)"])]

    month = np.array([d.month for d in common], dtype=np.int64)
    return {
        "feats": torch.from_numpy(feats_arr),
        "coords": torch.from_numpy(coords),
        "month": torch.from_numpy(month),
        "l_enc": int(T),
        "G": int(G),
    }


def process():
    print("=" * 60)
    print("WRF-HRRR 气象 → 网格级 grid_cache.pt")
    print("=" * 60)

    # ---- 1. 读取 dataset.jsonl 作为元数据/顺序基准 ----
    print("\n[1/3] 读取 dataset.jsonl 行(元数据基准)...")
    meta_lines = []
    with open(JSONL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                meta_lines.append(json.loads(line))
    print(f"  jsonl 行数: {len(meta_lines)}")

    # ---- 2. 按 (year, state_abbr) 分组,逐组加载气象 ----
    print("\n[2/3] 逐 (year, state) 加载 WRF-HRRR,构建网格级样本...")
    groups = {}
    for i, m in enumerate(meta_lines):
        abbr = STATE_TO_USPS.get(str(m["State"]).upper())
        if abbr is None:
            print(f"  [警告] State={m['State']} 无 USPS 映射, line {i}")
            continue
        groups.setdefault((int(m["Year"]), abbr), []).append(i)

    entries = [None] * len(meta_lines)
    n_done = 0
    for (year, abbr), idxs in groups.items():
        df = _load_state_weather(year, abbr)
        if df is None:
            continue
        for i in idxs:
            m = meta_lines[i]
            fips = str(m["FIPS"]).zfill(5)
            cdf = df[df["FIPS Code"].astype(str).str.strip() == fips]
            if cdf.empty:
                continue
            entries[i] = build_entry(cdf, WRF_COLS)
            n_done += 1
            if n_done % 500 == 0:
                print(f"  {n_done}/{len(meta_lines)} ...")

    # ---- 3. 校验 + 保存 ----
    print("\n[3/3] 校验与保存")
    n_ok = sum(1 for e in entries if e is not None)
    gs = [e["G"] for e in entries if e is not None]
    ts = [e["l_enc"] for e in entries if e is not None]
    mismatch = sum(
        1 for i, e in enumerate(entries)
        if e is not None and e["l_enc"] != int(meta_lines[i]["l_enc"])
    )
    print(f"  有效样本: {n_ok}/{len(meta_lines)}")
    print(f"  G: min={min(gs)} median={_median(gs)} max={max(gs)}")
    print(f"  T: min={min(ts)} median={_median(ts)} max={max(ts)}")
    print(f"  与 jsonl l_enc 不一致条数: {mismatch}")

    payload = {"version": 2, "feat_names": list(WRF_COLS), "entries": entries}
    torch.save(payload, OUT_PATH)
    print(f"  已保存: {OUT_PATH} ({round(os.path.getsize(OUT_PATH)/1e9, 2)} GB)")

    meta = {
        "version": 2,
        "n_lines": len(meta_lines),
        "n_ok": n_ok,
        "feat_names": list(WRF_COLS),
        "G_min": min(gs), "G_median": _median(gs), "G_max": max(gs),
        "T_min": min(ts), "T_median": _median(ts), "T_max": max(ts),
        "l_enc_mismatch": mismatch,
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  元数据: {META_PATH}")
    print("\n完成!")


if __name__ == "__main__":
    process()
