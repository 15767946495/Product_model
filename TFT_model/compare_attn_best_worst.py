#!/usr/bin/env python3
"""
比较预测最好与最差样本的时序注意力矩阵。
Pass1: 全 2022 测试集推理,记录每样本 pred/label/|err|,定位最好与最差样本。
Pass2: 仅对这两个样本重跑,保存完整因果注意力矩阵热力图。

输出(存 train_output/test_2022_grid_hs36_h1_lstm2/):
  attn_best_worst.json
  attn_matrix_best.png / attn_matrix_worst.png
"""
import json, sys
from datetime import date as _date, timedelta
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
from models import TFTEncoderForYieldPrediction
from data import (
    load_jsonl, load_grid_cache, load_county_soil, build_grid_samples,
    GridTimeSeriesDataset, make_grid_collate_fn, SOIL_DIM,
    DEFAULT_DATA_JSONL, DEFAULT_GRID_CACHE, DEFAULT_COUNTY_SOIL,
    dynamic_names_from_hparams,
)

CFG = "hs36_h1_lstm2"
OUT = _THIS_DIR / "train_output" / f"test_2022_grid_{CFG}"
CKPT = _THIS_DIR / "train_output" / f"val_2021_grid_{CFG}" / "best_model.pth"


def load_model(device):
    hp = json.load(open(OUT / "model_hparams.json", encoding="utf-8"))
    model = TFTEncoderForYieldPrediction(
        soil_dim=SOIL_DIM, dynamic_feature_names=dynamic_names_from_hparams(hp),
        hidden_size=int(hp["hidden_size"]), num_lstm_layers=int(hp["num_lstm_layers"]),
        dropout=float(hp["dropout"]), output_size=1, num_heads=int(hp["num_heads"]),
        spatial_mode="attention",
    )
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.to(device).eval()
    return model


def build_loader(device, indices=None):
    meta_lines = load_jsonl(DEFAULT_DATA_JSONL)
    cache = load_grid_cache(DEFAULT_GRID_CACHE)
    pairs = [(m, e) for m, e in zip(meta_lines, cache["entries"]) if int(m["Year"]) == 2022]
    soil_dict = load_county_soil(DEFAULT_COUNTY_SOIL)
    fn = json.load(open(OUT / "feature_norm.json", encoding="utf-8"))
    gs = {k: (torch.tensor(v["mean"]), torch.tensor(v["std"])) for k, v in fn.items()}
    hp = json.load(open(OUT / "model_hparams.json", encoding="utf-8"))
    dyn = dynamic_names_from_hparams(hp)
    collate = make_grid_collate_fn(gs, dyn)
    samples = build_grid_samples(pairs, soil_dict=soil_dict, dynamic_feature_names=dyn)
    ds = GridTimeSeriesDataset(samples)
    if indices is not None:
        ds = Subset(ds, indices)
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate)
    return loader, pairs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")
    model = load_model(device)
    loader, pairs = build_loader(device)

    # ---- Pass 1: 全测试集,记录误差 ----
    rec = []
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            gf, gc, gm, mo, sf, lab, sl, st, yr, fips, cnty = batch
            pred, _, _ = model(gf.to(device), gc.to(device), gm.to(device),
                               sf.to(device), sl.to(device))
            B = len(sl)
            for b in range(B):
                sl_b = int(sl[b].item())
                q = sl_b - 1
                p = float(pred[b, q, 0].item())
                l = float(lab[b, 0].item())
                rec.append({
                    "global_i": bi * 32 + b,
                    "fips": str(fips[b]), "county": str(cnty[b]), "state": str(st[b]),
                    "year": int(yr[b]), "pred": p, "label": l, "err": abs(p - l),
                })
    assert len(rec) == len(pairs), f"{len(rec)} != {len(pairs)}"
    print(f"共 {len(rec)} 个测试样本")

    rec.sort(key=lambda r: r["err"])
    best, worst = rec[0], rec[-1]
    print(f"\n最好样本: {best['state']} {best['county']} (FIPS {best['fips']}) "
          f"pred={best['pred']:.2f} label={best['label']:.2f} err={best['err']:.2f}")
    print(f"最差样本: {worst['state']} {worst['county']} (FIPS {worst['fips']}) "
          f"pred={worst['pred']:.2f} label={worst['label']:.2f} err={worst['err']:.2f}")

    # ---- Pass 2: 只跑这两个样本,抓完整注意力矩阵 ----
    sel = [(best, "best"), (worst, "worst")]
    summary = {"cfg": CFG, "best": best, "worst": worst}
    for sample, tag in sel:
        i = sample["global_i"]
        sub_loader, _ = build_loader(device, indices=[i])
        with torch.no_grad():
            for batch in sub_loader:
                gf, gc, gm, mo, sf, lab, sl, st, yr, fips, cnty = batch
                _, attn, _ = model(gf.to(device), gc.to(device), gm.to(device),
                                   sf.to(device), sl.to(device))
                sl_b = int(sl[0].item())
                yr_b = int(yr[0])
                end_d = _date(yr_b, 11, 30)
                A = attn[0, :sl_b, :sl_b].cpu().numpy()  # (sl, sl) 因果上三角
                break

        # 坐标轴: 日期
        dates = [end_d - timedelta(days=(sl_b - 1 - k)) for k in range(sl_b)]
        months = [d.strftime("%Y-%m") for d in dates]
        uniq_months = []
        for m in months:
            if not uniq_months or uniq_months[-1] != m:
                uniq_months.append(m)
        tick_pos = []
        tick_lab = []
        for i2, m in enumerate(months):
            if i2 == 0 or months[i2] != months[i2 - 1]:
                tick_pos.append(i2)
                tick_lab.append(m)

        # 末步 query 行的月度分布
        row = A[-1, :]
        bym = {}
        for k in range(sl_b):
            d = dates[k]
            bym[d.month] = bym.get(d.month, 0.0) + float(row[k])
        month_share = {m: round(v, 3) for m, v in sorted(bym.items())}
        peak_i = int(np.argmax(row))
        summary[f"{tag}_month_share"] = month_share
        summary[f"{tag}_peak"] = {"idx": peak_i, "date": dates[peak_i].isoformat(),
                                  "weight": round(float(row[peak_i]), 4)}
        print(f"\n[{tag}] 末步行 月度占比 {month_share}")
        print(f"[{tag}] 峰值 idx={peak_i} date={dates[peak_i]} weight={row[peak_i]:.4f}")

        # ---- 绘图: 完整注意力矩阵热力图 + 末步行曲线 ----
        fig, axes = plt.subplots(1, 2, figsize=(18, 6.5),
                                 gridspec_kw={"width_ratios": [1.35, 1]})

        ax = axes[0]
        im = ax.imshow(A, aspect="auto", cmap="viridis", origin="lower",
                       vmin=0.0, vmax=float(A.max()))
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, rotation=45, fontsize=7)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(tick_lab, fontsize=7)
        ax.set_xlabel("Key position (date)")
        ax.set_ylabel("Query position (date)")
        ax.set_title(f"{tag.upper()}  |err|={sample['err']:.2f} bu/ac\n"
                     f"causal attention matrix (rows sum to 1)")
        fig.colorbar(im, ax=ax, fraction=0.03)

        ax = axes[1]
        ax.plot([d.strftime("%m-%d") for d in dates], row, color="#1f77b4", lw=1.0)
        ax.set_title(f"last-step query row\npeak={dates[peak_i].strftime('%Y-%m-%d')} "
                     f"({row[peak_i]:.3f})")
        ax.set_ylabel("attention weight")
        ticks2 = [tick_pos[i2] for i2 in range(0, len(tick_pos), 2)]
        labs2 = [tick_lab[i2] for i2 in range(0, len(tick_pos), 2)]
        ax.set_xticks(ticks2)
        ax.set_xticklabels(labs2, rotation=45, fontsize=7)
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        out = OUT / f"attn_matrix_{tag}.png"
        fig.savefig(out, dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"图已保存: {out}")

    with open(OUT / "attn_best_worst.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 已保存: {OUT / 'attn_best_worst.json'}")


if __name__ == "__main__":
    main()
