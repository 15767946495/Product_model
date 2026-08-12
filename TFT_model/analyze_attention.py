"""
分析验证集上「预测误差较大的州」的网格注意力分布。

对每个县-年样本:
  - 取最后一个有效时间步(month>=8, t<seq_len)的预测与标签 → 误差 |pred-label|
  - 对每个气象特征,用 model.spatial_agg.forward_weights 取该时刻 CLS 对各网格的
    注意力 w[b,t,0,1:1+G] → (G,) 权重分布
计算每个样本/每个特征的注意力分布统计量:
  - entropy_norm = H(p)/log(G)          归一化熵(0 完全集中,1 完全均匀)
  - peak         = max(p)               峰值权重
  - hhi          = sum(p^2)             Herfindahl 集中度(1/G 均匀,1 单峰)
  - effG         = exp(H)               有效网格数
按州聚合:平均误差、平均注意力统计量,并对比高/低误差州。

输出:
  - attention_analysis.json   逐样本与逐州聚合
  - attn_figs.png             4 图:州误差条形图、误差~注意力熵散点、
                              最差州最差县注意力热力图(特征×网格)、
                              最差州最差县网格空间分布(经度×纬度,按注意力着色)
用法:
  python analyze_attention.py                 # val_2021
  python analyze_attention.py --val_year 2022
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
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
    dynamic_names_from_hparams,
)


def attn_stats(p: np.ndarray) -> Dict[str, float]:
    """从归一化注意力分布 p:(G,) 计算统计量。"""
    G = len(p)
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum() if p.sum() > 0 else np.ones(G) / G
    h = -np.sum(p * np.log(np.clip(p, 1e-12, None)))
    return {
        "G": int(G),
        "entropy_norm": float(h / math.log(G)) if G > 1 else 0.0,  # 0 集中,1 均匀
        "peak": float(p.max()),
        "hhi": float(np.sum(p ** 2)),
        "effG": float(np.exp(h)),
    }


def main():
    parser = argparse.ArgumentParser(description="网格注意力分布分析")
    parser.add_argument("--val_year", type=str, default="2021")
    parser.add_argument("--states", type=str, default="minnesota,wisconsin,michigan,iowa,illinois,indiana,ohio,missouri,kentucky",
                        help="按州过滤(逗号分隔的小写全称),默认 DeepCropNet 9 玉米带州")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    val_tag = args.val_year.replace(",", "_")
    output_dir = args.output_dir or os.path.join(_THIS_DIR, "train_output", f"val_{val_tag}")
    os.makedirs(output_dir, exist_ok=True)

    ckpt_path = os.path.join(output_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"[错误] checkpoint 不存在: {ckpt_path}")
        sys.exit(1)
    feature_norm_path = os.path.join(output_dir, "feature_norm.json")
    hparams_path = os.path.join(output_dir, "model_hparams.json")

    # ========== 1. 数据 ==========
    from data import (
        DEFAULT_DATA_JSONL, DEFAULT_GRID_CACHE, DEFAULT_COUNTY_SOIL,
    )
    meta_lines = load_jsonl(DEFAULT_DATA_JSONL)
    cache = load_grid_cache(DEFAULT_GRID_CACHE)
    cache_entries = cache["entries"]
    if len(cache_entries) != len(meta_lines):
        print("[错误] grid_cache 行数与 jsonl 行数不一致")
        sys.exit(1)

    val_years = [int(y) for y in args.val_year.split(",")]
    val_set = set(val_years)
    pairs = list(zip(meta_lines, cache_entries))
    if args.states:
        state_set = {s.strip().lower() for s in args.states.split(",") if s.strip()}
        pairs = [p for p in pairs if str(p[0].get("State", "")).lower() in state_set]
    val_pairs = [(m, e) for m, e in pairs if int(m["Year"]) in val_set]
    print(f"验证样本: {len(val_pairs)}")

    if not val_pairs:
        print(f"[错误] 无验证样本 (val_year={args.val_year})")
        sys.exit(1)

    soil_dict = load_county_soil(DEFAULT_COUNTY_SOIL)

    # ========== 2. 标准化参数 ==========
    with open(feature_norm_path, "r", encoding="utf-8") as f:
        fn_raw = json.load(f)
    global_stats = {}
    for k, v in fn_raw.items():
        global_stats[k] = (
            torch.tensor(v["mean"], dtype=torch.float32),
            torch.tensor(v["std"], dtype=torch.float32),
        )

    # ========== 3. DataLoader ==========
    with open(hparams_path, "r", encoding="utf-8") as f:
        _hp = json.load(f)
    dynamic_feature_names = dynamic_names_from_hparams(_hp)
    collate_fn = make_grid_collate_fn(global_stats, dynamic_feature_names)
    val_samples = build_grid_samples(val_pairs, soil_dict=soil_dict,
                                     dynamic_feature_names=dynamic_feature_names)
    val_dataset = GridTimeSeriesDataset(val_samples)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn)
    print(f"样本数: {len(val_dataset)}")

    # ========== 4. 模型 ==========
    with open(hparams_path, "r", encoding="utf-8") as f:
        hp = json.load(f)
    hidden_size = int(hp.get("hidden_size", 32))
    num_heads = int(hp.get("num_heads", 4))
    num_lstm_layers = int(hp.get("num_lstm_layers", 1))
    dropout = float(hp.get("dropout", 0.2))
    model = TFTEncoderForYieldPrediction(
        soil_dim=SOIL_DIM,
        dynamic_feature_names=dynamic_feature_names,
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        dropout=dropout,
        output_size=1,
        num_heads=num_heads,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"超参: hidden_size={hidden_size}, num_heads={num_heads}, "
          f"num_lstm_layers={num_lstm_layers}, dropout={dropout}")

    # ========== 5. 推理 + 提取注意力 ==========
    records: List[Dict] = []
    feature_names = model.dynamic_feature_names

    class _AttnRecorder(nn.Module):
        """替换 model.spatial_agg:模型自身 forward 里每特征调用一次,顺带记录 CLS 注意力。

        模型 forward 内按 dynamic_feature_names 顺序逐特征调用 spatial_agg,
        记录器按调用序号打标签,并把权重立刻搬到 CPU 释放显存。
        """
        def __init__(self, agg, names):
            super().__init__()
            self.agg = agg
            self.names = list(names)
            self.calls = 0
            self.current = {}

        def forward(self, tokens, coords, grid_mask):
            out, w = self.agg.forward_weights(tokens, coords, grid_mask)
            name = self.names[self.calls % len(self.names)]
            self.current[name] = w.detach().cpu()
            self.calls += 1
            return out

    recorder = _AttnRecorder(model.spatial_agg, feature_names)
    model.spatial_agg = recorder

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Infer+Attn"):
            (grid_feats, grid_coords, grid_mask, month_ids, soil_feats,
             labels, seq_lens, states, years, fips_list, county_list) = batch
            grid_feats = grid_feats.to(device)
            grid_coords = grid_coords.to(device)
            grid_mask = grid_mask.to(device)
            month_ids = month_ids.to(device)
            soil_feats = soil_feats.to(device)
            labels = labels.to(device)
            seq_lens = seq_lens.to(device)

            pred_all, _, _ = model(
                grid_feats=grid_feats, grid_coords=grid_coords,
                grid_mask=grid_mask, soil_feats=soil_feats,
                seq_lens=seq_lens,
            )  # (B,T,1)

            # 模型 forward 结束后,记录器已捕获本 batch 各特征的 CLS 权重(CPU)
            attn_w = recorder.current  # name -> (B, T, G+1, G+1) cpu
            recorder.current = {}
            recorder.calls = 0

            B, T, _ = pred_all.shape
            pad_mask = torch.arange(T, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
            valid_mask = pad_mask & (month_ids >= 8)  # (B,T)
            last_valid_t = valid_mask.sum(dim=1) - 1  # (B,)

            for b in range(B):
                t = int(last_valid_t[b].item())
                if t < 0:
                    continue
                pred = float(pred_all[b, t, 0].item())
                label = float(labels[b, 0].item())
                G = int(grid_mask[b].sum().item())
                if G < 1:
                    continue

                attn_by_feat = {}
                stats_by_feat = {}
                for name in feature_names:
                    wa = attn_w[name][b, t, 0, 1:1 + G].detach().cpu().numpy()
                    wa = wa / wa.sum() if wa.sum() > 0 else np.ones(G) / G
                    attn_by_feat[name] = wa.tolist()
                    stats_by_feat[name] = attn_stats(wa)

                # 跨特征平均统计量
                mean_stats = {}
                for k in ("entropy_norm", "peak", "hhi", "effG"):
                    vals = [s[k] for s in stats_by_feat.values()]
                    mean_stats[k] = float(np.mean(vals))

                records.append({
                    "fips": fips_list[b],
                    "county": county_list[b],
                    "state": str(states[b]).lower(),
                    "year": int(years[b]),
                    "pred": round(pred, 3),
                    "label": round(label, 3),
                    "err": abs(pred - label),
                    "G": G,
                    "last_t": t,
                    "mean_stats": mean_stats,
                    "feat_stats": {name: {k: round(v, 4) for k, v in st.items()}
                                   for name, st in stats_by_feat.items()},
                    "feat_attn": attn_by_feat,
                    "coords": grid_coords[b, :G].detach().cpu().numpy().tolist(),
                })

            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    # ========== 6. 按州聚合 ==========
    state_agg: Dict[str, Dict] = {}
    for r in records:
        s = r["state"]
        a = state_agg.setdefault(s, {"n": 0, "errs": [], "stats": []})
        a["n"] += 1
        a["errs"].append(r["err"])
        a["stats"].append(r["mean_stats"])
    for s, a in state_agg.items():
        a["mean_err"] = float(np.mean(a["errs"]))
        a["std_err"] = float(np.std(a["errs"]))
        a["median_err"] = float(np.median(a["errs"]))
        a["mean_stats"] = {k: float(np.mean([x[k] for x in a["stats"]]))
                           for k in ("entropy_norm", "peak", "hhi", "effG")}

    state_order = sorted(state_agg, key=lambda s: state_agg[s]["mean_err"], reverse=True)
    print("\n===== 按州预测误差(最后一个有效步) =====")
    for s in state_order:
        a = state_agg[s]
        st = a["mean_stats"]
        print(f"  {s:12s} n={a['n']:3d}  mean|err|={a['mean_err']:6.2f}  "
              f"std={a['std_err']:5.2f}  ent={st['entropy_norm']:.3f}  "
              f"peak={st['peak']:.3f}  hhi={st['hhi']:.3f}  effG={st['effG']:.1f}")

    # 高/低误差州注意力对比(以州平均误差中位数为界)
    med = np.median([state_agg[s]["mean_err"] for s in state_order])
    high = [s for s in state_order if state_agg[s]["mean_err"] >= med]
    low = [s for s in state_order if state_agg[s]["mean_err"] < med]

    def pool(states):
        stats = [r["mean_stats"] for r in records if r["state"] in states]
        if not stats:
            return None
        return {k: float(np.mean([x[k] for x in stats]))
                for k in ("entropy_norm", "peak", "hhi", "effG")}

    ph, pl = pool(high), pool(low)
    print("\n===== 高误差州 vs 低误差州 注意力对比 =====")
    if ph and pl:
        for k in ("entropy_norm", "peak", "hhi", "effG"):
            print(f"  {k:12s} 高误差 {ph[k]:.4f}  |  低误差 {pl[k]:.4f}")

    # ========== 7. 绘图 ==========
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # (a) 州误差条形图
    ax = axes[0, 0]
    xs = np.arange(len(state_order))
    ax.bar(xs, [state_agg[s]["mean_err"] for s in state_order],
           color=["#d62728" if state_agg[s]["mean_err"] >= med else "#1f77b4"
                  for s in state_order])
    ax.set_xticks(xs)
    ax.set_xticklabels(state_order, rotation=30, ha="right")
    ax.set_ylabel("mean |err| (bu/ac)")
    ax.set_title(f"Per-state mean |err| (val {args.val_year})")
    ax.grid(True, alpha=0.3)

    # (b) 州误差 vs 注意力熵 散点
    ax = axes[0, 1]
    for s in state_order:
        a = state_agg[s]
        ax.scatter(a["mean_err"], a["mean_stats"]["entropy_norm"], s=60)
        ax.annotate(s, (a["mean_err"], a["mean_stats"]["entropy_norm"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("mean |err| (bu/ac)")
    ax.set_ylabel("mean attention entropy (normalized)")
    ax.set_title("Error vs spatial-attention concentration (higher entropy = more uniform)")
    ax.grid(True, alpha=0.3)

    # (c) 最差州最差县:特征×网格注意力热力图
    worst_state = state_order[0]
    worst_county = max((r for r in records if r["state"] == worst_state),
                       key=lambda r: r["err"])
    ax = axes[1, 0]
    attn_mat = np.stack([worst_county["feat_attn"][f] for f in feature_names])  # (F,G)
    im = ax.imshow(attn_mat, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([f[:14] for f in feature_names], fontsize=7)
    ax.set_xlabel(f"grid index (G={worst_county['G']})")
    ax.set_title(f"{worst_state} worst county {worst_county['county']} "
                 f"err={worst_county['err']:.1f} (features × grids)")
    fig.colorbar(im, ax=ax)

    # (d) 最差州最差县:网格空间分布(经度×纬度)按平均注意力着色
    ax = axes[1, 1]
    coords = np.array(worst_county["coords"])
    mean_attn = np.mean([worst_county["feat_attn"][f] for f in feature_names], axis=0)
    sc = ax.scatter(coords[:, 1], coords[:, 0], c=mean_attn,
                    cmap="hot_r", s=120 * mean_attn / mean_attn.max() + 10)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title(f"{worst_state} {worst_county['county']} grid attention (geo)")
    fig.colorbar(sc, ax=ax)

    fig.tight_layout()
    fig_path = os.path.join(output_dir, "attn_analysis.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n图已保存: {fig_path}")

    # ========== 8. JSON ==========
    summary = {
        "val_year": args.val_year,
        "n_samples": len(records),
        "state_agg": {s: {k: v for k, v in state_agg[s].items() if k != "stats"}
                      for s in state_agg},
        "high_error_states": high,
        "low_error_states": low,
        "high_vs_low": {"high": ph, "low": pl},
    }
    # 逐样本记录仅保留统计量(丢弃大数组),避免 JSON 过大
    per_sample = []
    for r in records:
        r2 = {k: r[k] for k in ("fips", "county", "state", "year", "pred", "label",
                                "err", "G", "mean_stats", "feat_stats")}
        per_sample.append(r2)
    out = {"summary": summary, "per_sample": per_sample}
    out_path = os.path.join(output_dir, "attention_analysis.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"JSON 已保存: {out_path}")


if __name__ == "__main__":
    main()
