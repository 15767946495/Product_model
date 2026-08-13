"""
BaseLine_Model 共享数据模块。

从 TFT 数据管线(grid_cache.pt + dataset.jsonl + DataSrc 土壤)构建
DeepCropNet 9 玉米带州的 weather / soil / yield 张量,供
CNNRNN / ConvLSTM / GNNRNN 三个基线共用。

要点:
  - 逐日 275 步(与 TFT 同粒度),网格不截断(逐样本列表存储,训练时按 batch 内最大 G 动态填充)
  - 天气/土壤/目标 z-score(仅训练集统计),报告时还原到原始 bu/ac
  - 训练 = 年份 < val_year,验证 = val_year(默认 2021)
  - 结果缓存到 output/baselines_data.pt,用 force=True 重建
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent          # BaseLine_Model/
_TFT = _ROOT.parent / "TFT_model"
if str(_TFT) not in sys.path:
    sys.path.insert(0, str(_TFT))

from data import (  # noqa: E402   (TFT_model 数据管线)
    load_jsonl,
    load_grid_cache,
    load_county_soil,
    SOIL_FEATURES,
    DEFAULT_DYNAMIC_FEATURE_NAMES,
    DEFAULT_DATA_JSONL,
    DEFAULT_GRID_CACHE,
    DEFAULT_COUNTY_SOIL,
)

OUT_DIR = _ROOT / "output"

# DeepCropNet(Lin 2020)论文的 9 玉米带州:MN, WI, MI, IA, IL, IN, OH, MO, KY
STATES = ["minnesota", "wisconsin", "michigan", "iowa", "illinois",
          "indiana", "ohio", "missouri", "kentucky"]
N_STEPS = 275          # 逐日(3-11月),与 TFT 输入粒度一致
N_FEATS = len(DEFAULT_DYNAMIC_FEATURE_NAMES)            # 11
SOIL_DIM = len(SOIL_FEATURES)                           # 7


# ---------------------------------------------------------------- 特征工程

def pad_daily(daily: np.ndarray, n_steps: int = N_STEPS) -> np.ndarray:
    """daily: (..., T, F) -> 对齐到 (..., n_steps, F)。
    T 不足时用最后一天的值补齐(全州数据各县长短不一)。"""
    d = daily[..., :n_steps, :]
    if d.shape[-2] < n_steps:
        pad = np.repeat(d[..., -1:, :], n_steps - d.shape[-2], axis=-2)
        d = np.concatenate([d, pad], axis=-2)
    return d


def build_dataset(cache_entries, meta_lines, soil_dict, states=STATES):
    """返回逐样本 dict 列表。states=None 表示全部州。"""
    samples = []
    for m, e in zip(meta_lines, cache_entries):
        state = str(m.get("State", "")).lower()
        if states is not None and state not in states:
            continue
        if e is None:
            continue
        feats = e["feats"].numpy()                 # (G, T, 11)
        coords = e["coords"].numpy()               # (G, 2)
        county_daily = feats.mean(axis=0)          # (T, 11) 县均值
        weather = pad_daily(county_daily)          # (275, 11)
        # 网格按经纬度排序(ConvLSTM 的 1D 空间轴)
        order = np.lexsort((coords[:, 1], coords[:, 0]))
        grid_daily = pad_daily(feats[order])       # (G, 275, 11)
        soil = np.array([float(soil_dict[str(m["FIPS"])][f]) for f in SOIL_FEATURES],
                        dtype=np.float32)          # (7,)
        samples.append({
            "fips": str(m["FIPS"]), "county": str(m.get("County", "")),
            "state": state, "year": int(m["Year"]),
            "y": float(m["yield_per_acre"]),
            "weather": weather.astype(np.float32),        # (275, 11)
            "grid_weather": grid_daily.astype(np.float32),  # (G, 275, 11)
            "grid_coords": coords[order].astype(np.float32),  # (G, 2)
            "soil": soil,
        })
    return samples


def kNN_adjacency(centroids, k=5, self_loop=True):
    """按质心欧氏距离建 kNN 邻接(归一化 coords),返回 D^-1/2 A D^-1/2 对称归一化矩阵。"""
    c = np.asarray(centroids, dtype=np.float32)
    c = (c - c.mean(axis=0)) / (c.std(axis=0) + 1e-6)
    n = len(c)
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        idx = np.argsort(d[i])[:k + 1]
        A[i, idx] = 1.0
        A[idx, i] = 1.0
    if self_loop:
        A = np.maximum(A, np.eye(n, dtype=np.float32))
    deg = A.sum(axis=1)
    dinv = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    A_norm = (A * dinv[:, None]) * dinv[None, :]
    return A_norm.astype(np.float32)


# ---------------------------------------------------------------- 数据准备

def split_years(samples, val_year: int, test_year: int):
    if test_year <= val_year:
        raise ValueError("test_year must be greater than val_year")
    train = [s for s in samples if s["year"] < val_year]
    val = [s for s in samples if s["year"] == val_year]
    test = [s for s in samples if s["year"] == test_year]
    return train, val, test


def prepare(val_year: int = 2021, test_year: int = 2022, out_dir=None,
            force: bool = False, gnn_k: int = 5):
    """构建/加载 9 州数据(缓存到 output/baselines_data.pt)。"""
    out_dir = Path(out_dir) if out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / f"baselines_data_val{val_year}_test{test_year}.pt"
    if cache_path.exists() and not force:
        print(f"[数据] 使用缓存 {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    meta = load_jsonl(DEFAULT_DATA_JSONL)
    cache = load_grid_cache(DEFAULT_GRID_CACHE)
    soil_dict = load_county_soil(DEFAULT_COUNTY_SOIL)
    all_samples = build_dataset(cache["entries"], meta, soil_dict, STATES)

    tr, va, te = split_years(all_samples, val_year, test_year)
    print(f"[数据] 训练 {len(tr)} / 验证 {len(va)} / 测试 {len(te)} (9 玉米带州)")

    def pack(samples):
        W = np.stack([s["weather"] for s in samples])
        S = np.stack([s["soil"] for s in samples])
        Y = np.array([s["y"] for s in samples], dtype=np.float32)
        meta_list = [{k: s[k] for k in ("fips", "county", "state", "year")} for s in samples]
        # 网格逐样本列表存储(不截断,训练时按 batch 动态填充,避免稠密 >10GB)
        grid_weather_list = [s["grid_weather"] for s in samples]
        grid_coords_list = [s["grid_coords"] for s in samples]
        centroids = [s["grid_coords"].mean(axis=0) for s in samples]
        A = kNN_adjacency(centroids, k=gnn_k)
        return {"weather": W, "soil": S, "y": Y, "grid_weather": grid_weather_list,
                "grid_coords": grid_coords_list, "adjacency": A, "meta": meta_list}

    tr_p, va_p, te_p = pack(tr), pack(va), pack(te)

    wm = tr_p["weather"].reshape(-1, N_FEATS).mean(0)
    ws = tr_p["weather"].reshape(-1, N_FEATS).std(0) + 1e-6
    sm = tr_p["soil"].mean(0)
    ss = tr_p["soil"].std(0) + 1e-6
    ym = float(tr_p["y"].mean())
    ys = float(tr_p["y"].std()) + 1e-6

    for split in (tr_p, va_p, te_p):
        split["weather"] = (split["weather"] - wm) / ws
        split["soil"] = (split["soil"] - sm) / ss
        split["y_std"] = (split["y"] - ym) / ys
        split["grid_weather"] = [(gw - wm) / ws for gw in split["grid_weather"]]

    gmax = max(max(g.shape[0] for g in tr_p["grid_weather"]),
               max(g.shape[0] for g in va_p["grid_weather"]),
               max(g.shape[0] for g in te_p["grid_weather"]))
    data = {"train": tr_p, "val": va_p, "test": te_p, "Gmax": gmax,
            "stats": {"wmean": wm, "wstd": ws, "smean": sm, "sstd": ss,
                      "ymean": ym, "ystd": ys}}
    torch.save(data, cache_path)
    print(f"[数据] 已保存 {cache_path}")
    return data


# ---------------------------------------------------------------- 工具

def to_tensor(arr, device):
    return torch.tensor(np.asarray(arr, dtype=np.float32), device=device)


def de_std(ymean, ystd):
    """标准化预测 -> 原始 bu/ac。"""
    def f(pred_std):
        return pred_std * ystd + ymean
    return f


def metrics(pred_raw, label_raw):
    p = np.asarray(pred_raw, dtype=np.float64)
    l = np.asarray(label_raw, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((p - l) ** 2)))
    r2 = 1.0 - np.sum((p - l) ** 2) / max(np.sum((l - l.mean()) ** 2), 1e-12)
    pc, lc = p - p.mean(), l - l.mean()
    corr = float(np.sum(pc * lc) / max(np.sqrt(np.sum(pc ** 2) * np.sum(lc ** 2)), 1e-12))
    return {"rmse": rmse, "r2": float(r2), "corr": corr, "n": int(len(p))}


def per_state_report(meta, pred_raw, y_raw):
    out = {}
    for st in sorted(set(s["state"] for s in meta)):
        mask = np.array([s["state"] == st for s in meta])
        out[st] = metrics(pred_raw[mask], y_raw[mask])
    return out


def grid_batch(split, idx, device):
    """ConvLSTM: 按 batch 内最大 G 动态填充逐样本网格列表(保留全部网格,不截断)。"""
    gw_list = [split["grid_weather"][i] for i in idx]
    B = len(gw_list)
    Gm = max(g.shape[0] for g in gw_list)
    gw = torch.zeros(B, Gm, N_STEPS, N_FEATS, device=device)
    gm = torch.zeros(B, Gm, device=device)
    for j, g in enumerate(gw_list):
        g = torch.tensor(np.asarray(g, dtype=np.float32), device=device)
        gw[j, :g.shape[0]] = g
        gm[j, :g.shape[0]] = 1.0
    return gw, gm


def eval_convlstm(model, split, soil_t, device, bs=128):
    """ConvLSTM 分批评估(避免全量 pad 到验证集最大 G 造成 >3GB 张量)。"""
    preds = []
    n = len(split["y"])
    for i in range(0, n, bs):
        idx = list(range(i, min(i + bs, n)))
        gwb, gmb = grid_batch(split, idx, device)
        with torch.no_grad():
            pv = model(gwb, gmb, soil_t[idx])
        preds.append(pv.squeeze(1).cpu().numpy())
    return np.concatenate(preds)
