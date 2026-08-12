#!/usr/bin/env python3
"""
分析四个网格搜索 checkpoint 在 2021 验证集上的逐时间步曲线,
找出 RMSE / R² / Corr 的最优点并映射到日历日期,判断最优点是否在最后一个时间步。
"""
import json
import math
import os
import sys
from datetime import date as _date
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from models import TFTEncoderForYieldPrediction
from data import (
    load_jsonl,
    load_grid_cache,
    load_county_soil,
    build_grid_samples,
    GridTimeSeriesDataset,
    make_grid_collate_fn,
    SOIL_DIM,
    DEFAULT_DATA_JSONL,
    DEFAULT_GRID_CACHE,
    DEFAULT_COUNTY_SOIL,
    dynamic_names_from_hparams,
)

DIRS = [
    ("hs32_h1_lstm1", "val_2021_grid_hs32_h1_lstm1"),
    ("hs32_h1_lstm2", "val_2021_grid_hs32_h1_lstm2"),
    ("hs32_h2_lstm1", "val_2021_grid_hs32_h2_lstm1"),
    ("hs32_h2_lstm2", "val_2021_grid_hs32_h2_lstm2"),
]
VAL_YEAR = 2021
STATES = "minnesota,wisconsin,michigan,iowa,illinois,indiana,ohio,missouri,kentucky"


def compute_metrics(p: np.ndarray, l: np.ndarray):
    n = len(p)
    if n < 2:
        return None
    resid = p - l
    ss_res = float((resid ** 2).sum())
    rmse = math.sqrt(ss_res / n)
    ss_tot = float(((l - l.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    p_c, l_c = p - p.mean(), l - l.mean()
    cov = float((p_c * l_c).sum())
    std_p, std_l = float(np.sqrt((p_c ** 2).sum())), float(np.sqrt((l_c ** 2).sum()))
    corr = cov / max(std_p * std_l, 1e-12)
    corr = max(-1.0, min(1.0, corr))
    return rmse, r2, corr


def load_common_data():
    jsonl_path = DEFAULT_DATA_JSONL
    grid_cache_path = DEFAULT_GRID_CACHE
    print(f"加载数据:\n    jsonl: {jsonl_path}\n    grid_cache: {grid_cache_path}")
    meta_lines = load_jsonl(jsonl_path)
    cache = load_grid_cache(grid_cache_path)
    cache_entries = cache["entries"]
    assert len(cache_entries) == len(meta_lines), "grid_cache 行数与 jsonl 不一致"
    soil_dict = load_county_soil(DEFAULT_COUNTY_SOIL)
    pairs = list(zip(meta_lines, cache_entries))
    state_set = {s.strip().lower() for s in STATES.split(",")}
    pairs = [p for p in pairs if str(p[0].get("State", "")).lower() in state_set]
    val_pairs = [(m, e) for m, e in pairs if int(m["Year"]) == VAL_YEAR]
    print(f"    {VAL_YEAR} 验证样本: {len(val_pairs)}")
    return val_pairs, soil_dict


def run_curve(ckpt_dir: str, val_pairs, soil_dict, device="cpu"):
    out_dir = Path(ckpt_dir)
    with open(out_dir / "model_hparams.json", "r", encoding="utf-8") as f:
        hp = json.load(f)
    hidden_size = int(hp.get("hidden_size", 32))
    num_heads = int(hp.get("num_heads", 4))
    num_lstm_layers = int(hp.get("num_lstm_layers", 1))
    dropout = float(hp.get("dropout", 0.2))
    spatial_mode = str(hp.get("spatial_mode", "attention"))

    with open(out_dir / "feature_norm.json", "r", encoding="utf-8") as f:
        fn_raw = json.load(f)
    global_stats = {k: (torch.tensor(v["mean"], dtype=torch.float32),
                        torch.tensor(v["std"], dtype=torch.float32)) for k, v in fn_raw.items()}

    dyn_names = dynamic_names_from_hparams(hp)
    collate_fn = make_grid_collate_fn(global_stats, dyn_names)
    samples = build_grid_samples(val_pairs, soil_dict=soil_dict, dynamic_feature_names=dyn_names)
    loader = DataLoader(GridTimeSeriesDataset(samples), batch_size=32, shuffle=False, collate_fn=collate_fn)

    model = TFTEncoderForYieldPrediction(
        soil_dim=SOIL_DIM, dynamic_feature_names=dyn_names,
        hidden_size=hidden_size, num_lstm_layers=num_lstm_layers,
        dropout=dropout, output_size=1, num_heads=num_heads, spatial_mode=spatial_mode,
    )
    model.load_state_dict(torch.load(out_dir / "best_model.pth", map_location=device))
    model.to(device).eval()

    step_preds = {}
    step_labels = {}
    full_len = 0  # 全序列长度(日期映射用)
    with torch.no_grad():
        for batch in tqdm(loader, desc="infer"):
            grid_feats, grid_coords, grid_mask, month_ids, soil_feats, labels, seq_lens, _, years, _, _ = batch
            grid_feats = grid_feats.to(device); grid_coords = grid_coords.to(device)
            grid_mask = grid_mask.to(device); month_ids = month_ids.to(device)
            soil_feats = soil_feats.to(device); labels = labels.to(device)
            seq_lens = seq_lens.to(device)
            pred_all, _, _ = model(grid_feats, grid_coords, grid_mask, soil_feats, seq_lens)
            B, T, _ = pred_all.shape
            pred_raw = pred_all.squeeze(-1)
            label_raw = labels.expand(-1, T)
            pad_mask = torch.arange(T, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
            valid_mask = pad_mask & (month_ids >= 8)
            full_len = max(full_len, int(seq_lens.max().item()))
            for t in range(T):
                t_mask = valid_mask[:, t]
                if t_mask.any():
                    step_preds.setdefault(t, []).extend(pred_raw[t_mask, t].detach().cpu().tolist())
                    step_labels.setdefault(t, []).extend(label_raw[t_mask, t].detach().cpu().tolist())

    # 每个时间步的日期(序列锚定 11-30;以最长序列映射)
    rows = []
    for t in sorted(step_preds.keys()):
        d = _date(VAL_YEAR, 11, 30) - timedelta(days=(full_len - 1 - t))
        m = compute_metrics(np.array(step_preds[t]), np.array(step_labels[t]))
        if m is None:
            continue
        rows.append({"step": t, "date": d.isoformat(), "month": d.month,
                     "rmse": round(m[0], 4), "r2": round(m[1], 4), "corr": round(m[2], 4),
                     "n": len(step_preds[t])})
    last = rows[-1]
    i_rmse = min(range(len(rows)), key=lambda i: rows[i]["rmse"])
    i_r2 = max(range(len(rows)), key=lambda i: rows[i]["r2"])
    i_corr = max(range(len(rows)), key=lambda i: rows[i]["corr"])
    summary = {
        "config": {"hidden": hidden_size, "heads": num_heads, "lstm": num_lstm_layers},
        "full_len": full_len,
        "last_step": last,
        "argmin_rmse": rows[i_rmse],
        "argmax_r2": rows[i_r2],
        "argmax_corr": rows[i_corr],
        "min_rmse_at_last": rows[i_rmse]["step"] == last["step"],
        "curve": rows,
    }
    return summary


def main():
    device = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ("cpu", "cuda") else (
        "cuda" if torch.cuda.is_available() else "cpu")
    val_pairs, soil_dict = load_common_data()
    results = {}
    for name, d in DIRS:
        p = _THIS_DIR / "train_output" / d
        print(f"\n{'='*60}\n  {name}  ({p.name})\n{'='*60}")
        r = run_curve(p, val_pairs, soil_dict, device)
        results[name] = r
        print(f"  最后一步: {r['last_step']}")
        print(f"  RMSE最小: {r['argmin_rmse']}  -> 最优点是否=最后一步: {r['min_rmse_at_last']}")
        print(f"  R2最大  : {r['argmax_r2']}")
        print(f"  Corr最大: {r['argmax_corr']}")

    print(f"\n{'='*70}")
    print("  汇总 (2021 验证集,逐时间步)")
    print(f"{'='*70}")
    print(f"  {'config':<14} {'argmin RMSE':>28} {'最后一步RMSE':>20} {'最优在最后?':>12}")
    for name, r in results.items():
        a = r["argmin_rmse"]
        last = r["last_step"]
        print(f"  {name:<14} {a['date']} rmse={a['rmse']:>8}  {last['rmse']:>12.4f}   {'否' if not r['min_rmse_at_last'] else '是':>8}")

    with open(_THIS_DIR / "analyze_curves_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {_THIS_DIR}/analyze_curves_results.json")


if __name__ == "__main__":
    main()
