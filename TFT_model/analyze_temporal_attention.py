#!/usr/bin/env python3
"""
时序注意力权重分析（直接证据）
==============================
在 2022 测试集上,对最终模型取每个样本"最后一个有效时间步(query)"
的因果注意力分布 attn_avg (B,T,T) 中该 query 对应的一行,
把 key 位置映射回日历日期并按日期聚合(跨县求平均),
得到"末步 query -> 各日期平均注意力权重"曲线。

输出:
  train_output/test_2022_grid_<cfg>/attn_weights_analysis.json
  train_output/test_2022_grid_<cfg>/attn_weights_curve.png

用法:
  python analyze_temporal_attention.py [--cfg hs36_h1_lstm2] [--device cpu|cuda]
"""

import argparse
import json
import sys
from datetime import date as _date
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    DEFAULT_DYNAMIC_FEATURE_NAMES,
    DEFAULT_DATA_JSONL,
    DEFAULT_GRID_CACHE,
    DEFAULT_COUNTY_SOIL,
    dynamic_names_from_hparams,
)

DEFAULT_STATES = "minnesota,wisconsin,michigan,iowa,illinois,indiana,ohio,missouri,kentucky"


def analyze_temporal_attention(cfg: str, device: str):
    out_dir = _THIS_DIR / "train_output" / f"test_2022_grid_{cfg}"
    if not out_dir.exists():
        print(f"[错误] 输出目录不存在: {out_dir}")
        sys.exit(1)

    # ========== 1. 数据(与 infer.py 一致) ==========
    jsonl_path = DEFAULT_DATA_JSONL
    grid_cache_path = DEFAULT_GRID_CACHE
    print(f"[1] 加载数据:\n    jsonl: {jsonl_path}\n    网格缓存: {grid_cache_path}")
    meta_lines = load_jsonl(jsonl_path)
    cache = load_grid_cache(grid_cache_path)
    cache_entries = cache["entries"]
    assert len(cache_entries) == len(meta_lines), "grid_cache 行数与 jsonl 不一致"

    pairs = list(zip(meta_lines, cache_entries))
    state_set = {s.strip().lower() for s in DEFAULT_STATES.split(",")}
    pairs = [p for p in pairs if str(p[0].get("State", "")).lower() in state_set]
    val_pairs = [(m, e) for m, e in pairs if int(m["Year"]) == 2022]
    print(f"    2022 测试样本: {len(val_pairs)}")

    soil_dict = load_county_soil(DEFAULT_COUNTY_SOIL)

    with open(out_dir / "feature_norm.json", "r", encoding="utf-8") as f:
        fn_raw = json.load(f)
    global_stats = {}
    for k, v in fn_raw.items():
        global_stats[k] = (
            torch.tensor(v["mean"], dtype=torch.float32),
            torch.tensor(v["std"], dtype=torch.float32),
        )

    # 特征名与训练一致(use_constructed/use_gdd),从 model_hparams.json 读取
    hparams_path = out_dir / "model_hparams.json"
    hp = {}
    if hparams_path.exists():
        with open(hparams_path, "r", encoding="utf-8") as f:
            hp = json.load(f)
    dyn_names = dynamic_names_from_hparams(hp)
    collate_fn = make_grid_collate_fn(global_stats, dyn_names)
    samples = build_grid_samples(val_pairs, soil_dict=soil_dict, dynamic_feature_names=dyn_names)
    loader = DataLoader(GridTimeSeriesDataset(samples), batch_size=32, shuffle=False, collate_fn=collate_fn)

    # ========== 2. 模型(最终配置) ==========
    # 权重存于 val_2021_grid_<cfg>/best_model.pth;test_2022 目录仅含推理输出
    ckpt_path = out_dir / "best_model.pth"
    if not ckpt_path.exists():
        ckpt_path = _THIS_DIR / "train_output" / f"val_2021_grid_{cfg}" / "best_model.pth"
    print(f"[2] checkpoint: {ckpt_path}")

    with open(out_dir / "model_hparams.json", "r", encoding="utf-8") as f:
        hp = json.load(f)
    hidden_size = int(hp.get("hidden_size", 36))
    num_heads = int(hp.get("num_heads", 1))
    num_lstm_layers = int(hp.get("num_lstm_layers", 2))
    dropout = float(hp.get("dropout", 0.2))
    spatial_mode = str(hp.get("spatial_mode", "attention"))
    use_grid_rope = bool(hp.get("grid_rope", False))
    use_time_rope = bool(hp.get("time_rope", False))
    print(f"[2] 超参: hidden={hidden_size}, heads={num_heads}, lstm={num_lstm_layers}, "
          f"spatial={spatial_mode}, grid_rope={use_grid_rope}, time_rope={use_time_rope}")

    model = TFTEncoderForYieldPrediction(
        soil_dim=SOIL_DIM,
        dynamic_feature_names=dyn_names,
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        dropout=dropout,
        output_size=1,
        num_heads=num_heads,
        spatial_mode=spatial_mode,
        use_grid_rope=use_grid_rope,
        use_time_rope=use_time_rope,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device).eval()
    print(f"    参数: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 3. 收集末步 query 的注意力分布 ==========
    # date_weight[date] = 该日期作为 key 被末步 query 注意到的权重之和
    # date_count[date]  = 该日期出现的样本数
    date_weight: dict = {}
    date_count: dict = {}
    full_len = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="attn"):
            grid_feats, grid_coords, grid_mask, month_ids, soil_feats, labels, seq_lens, _, years, _, _ = batch
            grid_feats = grid_feats.to(device)
            grid_coords = grid_coords.to(device)
            grid_mask = grid_mask.to(device)
            month_ids = month_ids.to(device)
            soil_feats = soil_feats.to(device)
            seq_lens = seq_lens.to(device)

            _, attn_out, _ = model(
                grid_feats=grid_feats,
                grid_coords=grid_coords,
                grid_mask=grid_mask,
                soil_feats=soil_feats,
                seq_lens=seq_lens,
            )  # attn_out: (B, T, T), 因果注意力, 头维平均

            B, T, _ = attn_out.shape
            for b in range(B):
                sl = int(seq_lens[b].item())
                yr = int(years[b])
                end_d = _date(yr, 11, 30)
                full_len = max(full_len, sl)
                # 最后一个有效步 = sl-1 (序列锚定 11-30)
                q = sl - 1
                # 该 query 对 key 位置 k 的注意力 (k<sl;因果掩码已屏蔽 k>=sl)
                row = attn_out[b, q, :sl].cpu().numpy()
                for k in range(sl):
                    d = end_d - timedelta(days=(sl - 1 - k))
                    dkey = d.isoformat()
                    date_weight[dkey] = date_weight.get(dkey, 0.0) + float(row[k])
                    date_count[dkey] = date_count.get(dkey, 0) + 1

    # ========== 4. 按日期聚合 -> 平均注意力 ==========
    dates = sorted(date_weight.keys())
    mean_w = np.array([date_weight[d] / date_count[d] for d in dates])
    curve = [
        {"date": d, "mean_attn": round(date_weight[d] / date_count[d], 6), "n": date_count[d]}
        for d in dates
    ]

    i_peak = int(np.argmax(mean_w))
    peak_date = dates[i_peak]
    conc = float(mean_w.max() / mean_w.mean())

    print(f"\n[4] 末步 query 注意力曲线 (共 {len(dates)} 个日期):")
    print(f"    峰值日期: {peak_date}  (平均注意力 {mean_w.max():.5f})")
    print(f"    注意力集中度(峰值/均值): {conc:.2f}")
    top5 = sorted(curve, key=lambda x: x["mean_attn"], reverse=True)[:5]
    print(f"    注意力最高的 5 个日期: {[(c['date'], c['mean_attn']) for c in top5]}")

    # ========== 5. 绘图 ==========
    print("[5] 绘图")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates, mean_w, color="#1f77b4", linewidth=1.6, marker=".")
    ax.set_xlabel("日期")
    ax.set_ylabel("末步 query 平均注意力权重")
    ax.set_title(f"时序注意力: 末步 query 对各日期的注意力 (2022 测试集, 配置 {cfg})")
    ax.grid(True, alpha=0.3)
    ax.annotate(
        f"峰值 {peak_date}\n权重 {mean_w.max():.4f}",
        xy=(peak_date, mean_w[i_peak]),
        xytext=(dates[max(0, i_peak - len(dates) // 5)], mean_w[i_peak] * 0.6),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    plot_path = out_dir / "attn_weights_curve.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    曲线已保存: {plot_path}")

    # ========== 6. 保存 JSON ===========
    result = {
        "cfg": cfg,
        "val_year": 2022,
        "n_samples": len(val_pairs),
        "full_len": full_len,
        "peak_date": peak_date,
        "peak_mean_attn": round(float(mean_w.max()), 6),
        "concentration_peak_over_mean": round(conc, 3),
        "top5": top5,
        "curve": curve,
    }
    result_path = out_dir / "attn_weights_analysis.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"    结果已保存: {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="时序注意力权重分析")
    parser.add_argument("--cfg", type=str, default="hs36_h1_lstm2")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    analyze_temporal_attention(args.cfg, device)
