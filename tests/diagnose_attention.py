"""
模型推理时消融诊断脚本。

加载训练好的 checkpoint，在验证集上运行四组推理：
  1. 完整模型（baseline）
  2. 去掉遥感分支（diagnose_mode={"disable_rs": True}）
  3. 空间注意力改为均值池化（diagnose_mode={"force_mean_spatial": True}）
  4. 同时去掉两者

对比各组 RMSE 差异，定位各模块的实际贡献。
同时可视化注意力分布诊断退化问题。

用法:
  python tests/diagnose_attention.py
  python tests/diagnose_attention.py --val_year 2022 --num_samples 0
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
_PROJ_ROOT = _THIS_DIR.parent
_TFT_DIR = _PROJ_ROOT / "TFT_model"
sys.path.insert(0, str(_TFT_DIR))
sys.path.insert(0, str(_PROJ_ROOT))

from models import TFTEncoderForYieldPrediction
from data import (
    load_jsonl, load_grid_cache, load_county_soil,
    build_grid_samples, GridTimeSeriesDataset, make_grid_collate_fn,
    SOIL_DIM, DEFAULT_DYNAMIC_FEATURE_NAMES,
    DEFAULT_DATA_JSONL, DEFAULT_GRID_CACHE, DEFAULT_COUNTY_SOIL,
    GDD_FEATURE_NAME, CONSTRUCTED_FEATURES,
)
from cropnet_protocol import CROPNET_FIVE_STATES

AG_DATES = [
    "04-01", "04-15", "05-01", "05-15", "06-01", "06-15",
    "07-01", "07-15", "08-01", "08-15", "09-01", "09-15",
]


def _attention_entropy(weights: np.ndarray, axis: int = -1) -> np.ndarray:
    eps = 1e-12
    log_w = np.log(weights + eps)
    entropy = -np.sum(weights * log_w, axis=axis)
    n = weights.shape[axis]
    return entropy / max(np.log(n), 1e-12)


def _shrink_name(name: str, max_len: int = 14) -> str:
    parts = name.split(".")
    if len(parts) > 1:
        name = parts[-1]
    if len(name) > max_len:
        name = name[:max_len - 3] + "..."
    return name


def last_valid_index(seq_lens: torch.Tensor) -> torch.Tensor:
    return (seq_lens - 1).clamp_min(0)


def load_model_and_data(output_dir: str, val_year: str, device: str) -> tuple:
    val_years = [int(y) for y in val_year.split(",")]

    hp_path = os.path.join(output_dir, "model_hparams.json")
    with open(hp_path, "r") as f:
        hp = json.load(f)
    hidden_size = int(hp["hidden_size"])
    num_heads = int(hp["num_heads"])
    num_lstm_layers = int(hp["num_lstm_layers"])
    dropout = float(hp["dropout"])
    spatial_mode = str(hp["spatial_mode"])
    variable_selection_stage = str(hp.get("variable_selection_stage", "grid"))
    use_gdd = bool(hp.get("use_gdd", False))
    use_constructed = bool(hp.get("use_constructed", False))
    use_rs = bool(hp.get("use_remote_sensing", False))
    print(f"[模型] hidden={hidden_size} heads={num_heads} sp={spatial_mode} "
          f"vsn={variable_selection_stage} rs={use_rs}")

    norm_path = os.path.join(output_dir, "feature_norm.json")
    with open(norm_path, "r") as f:
        fn = json.load(f)
    global_stats = {}
    for k, v in fn.items():
        global_stats[k] = (torch.tensor(v["mean"], dtype=torch.float32),
                            torch.tensor(v["std"], dtype=torch.float32))

    jsonl_path, grid_cache_path = DEFAULT_DATA_JSONL, DEFAULT_GRID_CACHE
    meta_lines = load_jsonl(jsonl_path)
    cache = load_grid_cache(grid_cache_path)
    if len(cache["entries"]) != len(meta_lines):
        raise ValueError("grid_cache 与 jsonl 行数不一致")

    dynamic_feature_names = list(DEFAULT_DYNAMIC_FEATURE_NAMES)
    if use_constructed:
        dynamic_feature_names += CONSTRUCTED_FEATURES
    elif use_gdd:
        dynamic_feature_names.append(GDD_FEATURE_NAME)

    soil_dict = load_county_soil(DEFAULT_COUNTY_SOIL)
    pairs = list(zip(meta_lines, cache["entries"]))
    pairs = [p for p in pairs
             if str(p[0].get("State", "")).lower() in CROPNET_FIVE_STATES]
    val_set = set(val_years)
    val_pairs = [(m, e) for m, e in pairs if int(m["Year"]) in val_set]

    kept = []
    for m, e in val_pairs:
        try:
            v = float(m.get("yield_per_acre", np.nan))
            if np.isfinite(v):
                kept.append((m, e))
        except (TypeError, ValueError):
            pass
    val_pairs = kept

    val_ag = None
    if use_rs:
        from data import (AG_STATE_ABBR, load_ag_manifest, DEFAULT_AG_MANIFEST,
                          AgricultureImageDataset)
        raw_ag = load_ag_manifest(DEFAULT_AG_MANIFEST)
        ag_available = set()
        for (abbr, yr, fips_str), entries in raw_ag.items():
            qs = {e["path"].rstrip(".h5").split("_")[-1][-5:] for e in entries}
            if "06-30" in qs and "09-30" in qs:
                ag_available.add((abbr, int(yr), str(fips_str).zfill(5)))
        val_pairs_ag = []
        for m, e in val_pairs:
            abbr = AG_STATE_ABBR.get(str(m.get("State", "")).strip().lower())
            if abbr is None:
                continue
            if (abbr, int(m["Year"]), str(m.get("FIPS", "")).zfill(5)) in ag_available:
                val_pairs_ag.append((m, e))
        print(f"  AG 过滤: {len(val_pairs_ag)}/{len(val_pairs)}")
        val_pairs = val_pairs_ag
        val_ag = AgricultureImageDataset([m for m, _ in val_pairs], train=False, seed=42)
    print(f"  验证样本: {len(val_pairs)}")

    collate_fn = make_grid_collate_fn(global_stats, dynamic_feature_names)
    val_samples = build_grid_samples(val_pairs, soil_dict=soil_dict,
                                       dynamic_feature_names=dynamic_feature_names)
    val_dataset = GridTimeSeriesDataset(val_samples, ag_dataset=val_ag)
    loader = DataLoader(val_dataset, batch_size=8, shuffle=False,
                        collate_fn=collate_fn)

    ckpt = os.path.join(output_dir, "best_model.pth")
    print(f"[加载] {ckpt}")
    model = TFTEncoderForYieldPrediction(
        soil_dim=SOIL_DIM, dynamic_feature_names=dynamic_feature_names,
        hidden_size=hidden_size, num_lstm_layers=num_lstm_layers,
        dropout=dropout, output_size=1, num_heads=num_heads,
        spatial_mode=spatial_mode,
        variable_selection_stage=variable_selection_stage,
        use_remote_sensing=use_rs,
    )
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device)
    model.eval()

    meta = {
        "dynamic_feature_names": dynamic_feature_names,
        "hidden_size": hidden_size,
        "spatial_mode": spatial_mode,
        "variable_selection_stage": variable_selection_stage,
        "use_rs": use_rs,
    }
    return model, loader, meta


def run_ablation(
    model, loader, device, diagnose_mode: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """运行一轮推理，返回 (preds, labels) numpy 数组。"""
    all_preds, all_labels = [], []
    desc = "完整模型" if diagnose_mode is None else str(diagnose_mode)
    with torch.no_grad():
        for batch in loader:
            grid_feats, grid_coords, grid_mask, _, _, \
                soil_feats, labels, seq_lens, *_rest = batch
            ag_images = None
            if model.use_remote_sensing and len(_rest) >= 5 and _rest[4] is not None:
                ag_images = _rest[4].to(device)

            grid_feats = grid_feats.to(device)
            grid_coords = grid_coords.to(device)
            grid_mask = grid_mask.to(device)
            soil_feats = soil_feats.to(device)
            labels = labels.to(device)
            seq_lens = seq_lens.to(device)

            pred_all, _, _ = model(
                grid_feats=grid_feats, grid_coords=grid_coords,
                grid_mask=grid_mask, soil_feats=soil_feats,
                seq_lens=seq_lens, ag_images=ag_images,
                diagnose_mode=diagnose_mode,
            )
            B = pred_all.shape[0]
            idx = last_valid_index(seq_lens)
            batch_idx = torch.arange(B, device=device)
            preds = pred_all[batch_idx, idx].squeeze(-1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.squeeze(-1).cpu().numpy().tolist())
    return np.array(all_preds), np.array(all_labels)


def collect_attentions(model, loader, device, num_samples: int) -> Dict:
    collected: Dict[str, List] = {
        "spatial": [], "rs_spatial": [], "vsn": [], "temporal": [],
    }
    count = 0
    with torch.no_grad():
        for batch in loader:
            grid_feats, grid_coords, grid_mask, _, _, \
                soil_feats, labels, seq_lens, *_rest = batch
            ag_images = None
            if model.use_remote_sensing and len(_rest) >= 5 and _rest[4] is not None:
                ag_images = _rest[4].to(device)
            grid_feats = grid_feats.to(device)
            grid_coords = grid_coords.to(device)
            grid_mask = grid_mask.to(device)
            soil_feats = soil_feats.to(device)
            seq_lens = seq_lens.to(device)

            _, attn_temporal, aux = model(
                grid_feats=grid_feats, grid_coords=grid_coords,
                grid_mask=grid_mask, soil_feats=soil_feats,
                seq_lens=seq_lens, ag_images=ag_images,
            )
            B = grid_feats.shape[0]
            for i in range(B):
                sl_i = int(seq_lens[i].item())
                g_i = int(grid_mask[i].sum().item())
                if sl_i == 0 or g_i == 0:
                    continue
                if aux.get("spatial_weights") is not None:
                    collected["spatial"].append(
                        aux["spatial_weights"][i, :sl_i, :g_i].cpu().numpy())
                if aux.get("rs_spatial_weights") is not None:
                    collected["rs_spatial"].append(
                        aux["rs_spatial_weights"][i, :, :g_i].cpu().numpy())
                vsn_w = aux.get("grid_vsn_weights")
                if vsn_w is None:
                    vsn_w = aux.get("county_vsn_weights")
                if vsn_w is not None:
                    if vsn_w.dim() == 4:
                        vsn_w = vsn_w.mean(dim=2)
                    collected["vsn"].append(vsn_w[i, :sl_i, :].cpu().numpy())
                collected["temporal"].append(
                    attn_temporal[i, :sl_i, :sl_i].cpu().numpy())
                count += 1
                if count >= num_samples:
                    break
            if count >= num_samples:
                break
    print(f"  收集 {count} 个样本注意力")
    return collected


def plot_attention_diag(collected: Dict, feat_names: List[str], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    short_names = [_shrink_name(n) for n in feat_names]
    F = len(feat_names)

    # --- 空间注意力 ---
    if collected["spatial"]:
        print("\n--- 气象空间注意力 ---")
        ents = np.array([_attention_entropy(s, axis=-1).mean()
                         for s in collected["spatial"]])
        print(f"  熵: μ={ents.mean():.3f} σ={ents.std():.3f}  "
              f"{'⚠ 均匀' if ents.mean() > 0.85 else '✓ 有选择'}")
        mid = int(np.argsort(np.abs(ents - np.median(ents)))[0])
        sw = collected["spatial"][mid]
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(sw.T, aspect="auto", cmap="YlOrRd")
        ax.set_xlabel("时间步"); ax.set_ylabel(f"网格 (G={sw.shape[1]})")
        ax.set_title(f"气象空间注意力 (熵={ents[mid]:.3f})")
        plt.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(os.path.join(output_dir, "01_spatial.png"), dpi=120)
        plt.close(fig)

    # --- VSN ---
    if collected["vsn"]:
        print("\n--- VSN 变量选择 ---")
        all_w = np.concatenate([v.mean(axis=0) for v in collected["vsn"]], axis=0)
        if all_w.ndim == 2:
            mw = all_w.mean(axis=0)
            ent = _attention_entropy(mw.reshape(1, -1), axis=-1)[0]
            print(f"  熵: {ent:.3f}  {'⚠ 均匀' if ent > 0.9 else '✓ 有选择'}")
            top_k = min(5, F)
            top = np.argsort(mw)[-top_k:][::-1]
            for idx in top:
                print(f"    {feat_names[int(idx)]:20s}: {mw[int(idx)]:.4f}")
            fig, ax = plt.subplots(figsize=(max(10, F * 0.6), 5))
            ax.bar(range(F), mw, color="#1f77b4")
            ax.set_xticks(range(F))
            ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
            ax.axhline(1.0 / F, color="gray", linestyle="--")
            ax.set_ylabel("平均 VSN 权重")
            ax.set_title(f"VSN 变量权重 (熵={ent:.3f})")
            fig.tight_layout(); fig.savefig(os.path.join(output_dir, "02_vsn.png"), dpi=120)
            plt.close(fig)

    # --- 时序注意力 ---
    if collected["temporal"]:
        print("\n--- 时序因果注意力 ---")
        tw = collected["temporal"][len(collected["temporal"]) // 2]
        T_dim = tw.shape[0]
        diag = np.trace(tw) / T_dim
        print(f"  对角线均值: {diag:.4f}")
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(tw, aspect="auto", cmap="Blues", origin="lower")
        ax.set_xlabel("Key 时间步"); ax.set_ylabel("Query 时间步")
        ax.set_title(f"时序因果注意力 (diag={diag:.3f})")
        plt.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(os.path.join(output_dir, "03_temporal.png"), dpi=120)
        plt.close(fig)

    # --- RS 空间注意力 ---
    if collected["rs_spatial"]:
        print("\n--- RS 空间注意力 ---")
        ents_rs = np.array([_attention_entropy(w, axis=-1).mean()
                            for w in collected["rs_spatial"]])
        print(f"  熵: μ={ents_rs.mean():.3f} σ={ents_rs.std():.3f}  "
              f"{'⚠ 均匀' if ents_rs.mean() > 0.85 else '✓ 有选择'}")
        rsw = collected["rs_spatial"][0]
        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.imshow(rsw.T, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(rsw.shape[0]))
        ax.set_xticklabels(AG_DATES, rotation=45, fontsize=7)
        ax.set_xlabel("遥感时相"); ax.set_ylabel(f"网格 (G={rsw.shape[1]})")
        ax.set_title("RS 空间注意力")
        plt.colorbar(im, ax=ax)
        fig.tight_layout(); fig.savefig(os.path.join(output_dir, "04_rs_spatial.png"), dpi=120)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="模型推理时消融诊断")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--val_year", type=str, default="2022")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=8,
                        help="可视化样本数，0 则跳过注意力分析")
    parser.add_argument("--diagnose_dir", type=str, default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    val_tag = args.val_year.replace(",", "_")
    output_dir = args.output_dir or os.path.join(
        "/data/raid0/hqx", "TFT_train", f"val_{val_tag}")
    diagnose_dir = args.diagnose_dir or os.path.join(output_dir, "attention_diagnose")

    model, loader, meta = load_model_and_data(output_dir, args.val_year, device)

    # ========== 推理时消融 ==========
    print("\n" + "=" * 60)
    print("推理时消融对比")
    print("=" * 60)

    results = {}
    configs = [
        ("完整模型", None),
        ("— 遥感 (disable_rs)", {"disable_rs": True}),
        ("— 空间注意力 (force_mean)", {"force_mean_spatial": True}),
        ("— 遥感 + — 空间注意力", {"disable_rs": True, "force_mean_spatial": True}),
    ]

    for name, dm in configs:
        preds, labels = run_ablation(model, loader, device, dm)
        resid = preds - labels
        rmse = math.sqrt(float((resid ** 2).mean()))
        ss_tot = float(((labels - labels.mean()) ** 2).sum())
        r2 = 1.0 - float((resid ** 2).sum()) / max(ss_tot, 1e-12)
        results[name] = {"RMSE": rmse, "R²": r2, "preds": preds, "labels": labels}
        print(f"\n  {name}:")
        print(f"    RMSE = {rmse:.3f} bu/ac, R² = {r2:.4f}")

    # ========== 贡献分析 ==========
    print("\n" + "=" * 60)
    print("各模块贡献分析")
    print("=" * 60)

    base = results["完整模型"]
    base_rmse = base["RMSE"]

    for ablate_name in ["— 遥感 (disable_rs)", "— 空间注意力 (force_mean)"]:
        ablate = results[ablate_name]
        delta = ablate["RMSE"] - base_rmse
        print(f"\n  {ablate_name}:")
        print(f"    ΔRMSE = {delta:+.3f} bu/ac "
              f"({'═ 无影响' if abs(delta) < 0.3 else '↑ 变差' if delta > 0 else '↓ 变好'})")

        # 逐样本预测差异分析
        pred_diff = ablate["preds"] - base["preds"]
        changed_frac = (np.abs(pred_diff) > 1.0).mean()
        print(f"    预测变化 >1 bu/ac 的样本比例: {changed_frac:.1%}")
        print(f"    预测差异 MAE: {np.abs(pred_diff).mean():.2f} bu/ac")

    both = results["— 遥感 + — 空间注意力"]
    delta_both = both["RMSE"] - base_rmse
    print(f"\n  同时去掉两者: ΔRMSE = {delta_both:+.3f} bu/ac")

    # 判断瓶颈
    print("\n" + "=" * 60)
    print("诊断结论")
    print("=" * 60)
    d_rs = results["— 遥感 (disable_rs)"]["RMSE"] - base_rmse
    d_sp = results["— 空间注意力 (force_mean)"]["RMSE"] - base_rmse
    d_both = both["RMSE"] - base_rmse

    if abs(d_rs) < 0.5 and abs(d_sp) < 0.5 and abs(d_both) < 0.5:
        print("  所有模块对最终 RMSE 影响均 < 0.5 bu/ac")
        print("  ⇒ 瓶颈不在遥感融合和空间注意力，检査 LSTM/注意力/VSN/特征体系")
    elif d_rs > 1 and d_rs > d_sp:
        print(f"  遥感模块贡献最大 (ΔRMSE={d_rs:+.2f})")
        print("  ⇒ RS 融合可能有问题，检查 ViT 编码质量/Cross-Attention")
    elif d_sp > 1 and d_sp > d_rs:
        print(f"  空间注意力贡献最大 (ΔRMSE={d_sp:+.2f})")
        print("  ⇒ 空间注意力未学到有意义的聚合")
    else:
        print(f"  遥感 Δ={d_rs:+.2f}, 空间 Δ={d_sp:+.2f}, 联合 Δ={d_both:+.2f}")
        print("  ⇒ 逐模块影响不大，可能整体特征/架构/训练不足")

    # ========== 注意力可视化（可选） ==========
    if args.num_samples > 0:
        print("\n" + "=" * 60)
        print("注意力分布可视化")
        print("=" * 60)
        collected = collect_attentions(model, loader, device, args.num_samples)
        plot_attention_diag(collected, meta["dynamic_feature_names"], diagnose_dir)
        print(f"图表保存至: {diagnose_dir}")

    print("\n完成")


if __name__ == "__main__":
    main()