"""
cropnet 推理脚本。

加载训练好的 checkpoint，在验证集上逐时间步计算 RMSE、Corr、R²，
绘制变化曲线，并输出最后一个时间步的指标。

用法：
  python infer.py                          # val_year=2022
  python infer.py --val_year 2021
  python infer.py --val_year 2022 --ckpt custom_path.pth
"""

import argparse
import json
import math
import os
import sys
from datetime import date as _date
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from models import MODEL_CONTRACT_VERSION, TFTEncoderForYieldPrediction
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
    GDD_FEATURE_NAME,
    CONSTRUCTED_FEATURES,
)


def infer():
    parser = argparse.ArgumentParser(description="cropnet 推理")
    parser.add_argument("--val_year", type=str, default="2022",
                        help="验证年份，默认 2022")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="checkpoint 路径，默认 train_output/val_<val_year>/best_model.pth")
    parser.add_argument("--states", type=str,
                        default="minnesota,wisconsin,michigan,iowa,illinois,indiana,ohio,missouri,kentucky",
                        help="按州过滤(逗号分隔的小写全称)。默认 DeepCropNet 9 玉米带州,需与训练一致")
    parser.add_argument("--grid_cache", type=str, default=None,
                        help="网格级气象缓存路径,默认 train_dataset/grid_cache.pt")
    parser.add_argument("--soil", type=str, default=None,
                        help="县级土壤数据路径,默认 soil_dataset/county_soil.json")
    parser.add_argument("--data_jsonl", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录，默认与 checkpoint 同目录")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cutoffs", type=str, default="08-01,08-16,08-31,09-15,09-30,10-15,10-30,11-14,11-30",
                        help="提前预报节点(MM-DD,逗号分隔)。默认自8月1日起每半个月一个节点直至11月30日")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    val_tag = args.val_year.replace(",", "_")
    output_dir = args.output_dir or os.path.join(_THIS_DIR, "train_output", f"val_{val_tag}")
    os.makedirs(output_dir, exist_ok=True)

    # 提前预报节点(MM-DD) -> [(month, day), ...]
    cutoff_list = []
    for s in args.cutoffs.split(","):
        s = s.strip()
        if s:
            mm, dd = (int(x) for x in s.split("-"))
            cutoff_list.append((mm, dd))

    ckpt_path = args.ckpt or os.path.join(output_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        # 尝试 MSE 版
        ckpt_path = os.path.join(output_dir, "best_model.pth")
    if not os.path.exists(ckpt_path):
        print(f"[错误] checkpoint 不存在: {ckpt_path}")
        sys.exit(1)

    feature_norm_path = os.path.join(output_dir, "feature_norm.json")

    # ========== 1. 加载数据(网格级) ==========
    jsonl_path = args.data_jsonl or DEFAULT_DATA_JSONL
    grid_cache_path = args.grid_cache or DEFAULT_GRID_CACHE
    print(f"[1] 加载数据:\n    jsonl: {jsonl_path}\n    网格缓存: {grid_cache_path}")
    meta_lines = load_jsonl(jsonl_path)
    cache = load_grid_cache(grid_cache_path)
    cache_entries = cache["entries"]
    if len(cache_entries) != len(meta_lines):
        print(f"[错误] grid_cache 行数 {len(cache_entries)} 与 jsonl 行数 {len(meta_lines)} 不一致")
        sys.exit(1)

    # 按年份划分:验证集 = val_year
    val_years = [int(y) for y in args.val_year.split(",")]
    val_set = set(val_years)
    pairs = list(zip(meta_lines, cache_entries))
    if args.states:
        state_set = {s.strip().lower() for s in args.states.split(",") if s.strip()}
        pairs = [p for p in pairs if str(p[0].get("State", "")).lower() in state_set]
    val_pairs = [(m, e) for m, e in pairs if int(m["Year"]) in val_set]
    print(f"    验证样本: {len(val_pairs)}")

    if not val_pairs:
        print(f"[错误] 无验证样本 (val_year={args.val_year})")
        sys.exit(1)

    # 县级连续土壤静态特征
    soil_path = args.soil or DEFAULT_COUNTY_SOIL
    soil_dict = load_county_soil(soil_path)

    # ========== 2. 加载特征标准化参数 ==========
    print(f"[2] 加载特征标准化参数")
    # 特征 norm
    with open(feature_norm_path, "r", encoding="utf-8") as f:
        fn_raw = json.load(f)
    global_stats = {}
    for k, v in fn_raw.items():
        global_stats[k] = (
            torch.tensor(v["mean"], dtype=torch.float32),
            torch.tensor(v["std"], dtype=torch.float32),
        )

    # ========== 3. DataLoader ==========
    print(f"[3] 构建 DataLoader")
    # 特征名与训练一致(use_gdd 时含 CumGDD 通道),从 hparams 读取
    hparams_path0 = os.path.join(output_dir, "model_hparams.json")
    use_gdd_hp = False
    use_constructed_hp = False
    if os.path.exists(hparams_path0):
        with open(hparams_path0, "r", encoding="utf-8") as f:
            hp0 = json.load(f)
            use_gdd_hp = bool(hp0.get("use_gdd", False))
            use_constructed_hp = bool(hp0.get("use_constructed", False))
    dynamic_feature_names = list(DEFAULT_DYNAMIC_FEATURE_NAMES)
    if use_constructed_hp:
        dynamic_feature_names += CONSTRUCTED_FEATURES   # 11+4=15 维
    elif use_gdd_hp:
        dynamic_feature_names.append(GDD_FEATURE_NAME)  # 旧口径 12 维
    collate_fn = make_grid_collate_fn(global_stats, dynamic_feature_names)

    val_samples = build_grid_samples(val_pairs, soil_dict=soil_dict, dynamic_feature_names=dynamic_feature_names)
    val_dataset = GridTimeSeriesDataset(val_samples)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    print(f"    样本数: {len(val_dataset)}, batch: {len(val_loader)}")

    # ========== 4. 创建模型 + 加载权重 ==========
    print(f"[4] 创建模型，加载: {ckpt_path}")
    # 优先读训练时保存的超参,保证 hidden_size/num_heads 与 checkpoint 一致
    # (否则用 num_heads 不同的 checkpoint 会 state_dict 尺寸不匹配而崩)
    hparams_path = os.path.join(output_dir, "model_hparams.json")
    if not os.path.exists(hparams_path):
        raise ValueError("model_hparams.json is required; legacy checkpoints are unsupported")
    with open(hparams_path, "r", encoding="utf-8") as f:
        hp = json.load(f)
    if hp.get("model_contract_version") != MODEL_CONTRACT_VERSION:
        raise ValueError(
            f"checkpoint contract mismatch: expected {MODEL_CONTRACT_VERSION}, "
            f"got {hp.get('model_contract_version')}"
        )
    hidden_size = int(hp["hidden_size"])
    num_heads = int(hp["num_heads"])
    num_lstm_layers = int(hp["num_lstm_layers"])
    dropout = float(hp["dropout"])
    spatial_mode = str(hp["spatial_mode"])
    print(f"    超参: hidden_size={hidden_size}, num_heads={num_heads}, "
          f"num_lstm_layers={num_lstm_layers}, dropout={dropout}, spatial_mode={spatial_mode}")
    model = TFTEncoderForYieldPrediction(
        soil_dim=SOIL_DIM,
        dynamic_feature_names=dynamic_feature_names,
        hidden_size=hidden_size,
        num_lstm_layers=num_lstm_layers,
        dropout=dropout,
        output_size=1,
        num_heads=num_heads,
        spatial_mode=spatial_mode,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"    参数: {sum(p.numel() for p in model.parameters()):,}")

    # ========== 5. 推理：逐时间步收集 ==========
    print(f"[5] 推理中...")
    # step_preds[t] / step_labels[t]: 第 t 步所有县的 pred / 真实单产
    step_preds: Dict[int, List[float]] = {}
    step_labels: Dict[int, List[float]] = {}
    # 提前预报节点:每节点所有县的 pred / 真实单产
    node_preds: Dict[Tuple[int, int], List[float]] = {k: [] for k in cutoff_list}
    node_labels: Dict[Tuple[int, int], List[float]] = {k: [] for k in cutoff_list}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Infer"):
            grid_feats, grid_coords, grid_mask, month_ids, soil_feats, labels, seq_lens, states, years, _, _ = batch
            grid_feats = grid_feats.to(device)
            grid_coords = grid_coords.to(device)
            grid_mask = grid_mask.to(device)
            month_ids = month_ids.to(device)
            soil_feats = soil_feats.to(device)
            labels = labels.to(device)
            seq_lens = seq_lens.to(device)

            pred_all, _, _ = model(
                grid_feats=grid_feats,
                grid_coords=grid_coords,
                grid_mask=grid_mask,
                soil_feats=soil_feats,
                seq_lens=seq_lens,
            )  # (B, T, 1)

            B, T, _ = pred_all.shape

            # 无归一化,预测/标签已在原始单产空间 (bu/ac)
            pred_raw = pred_all.squeeze(-1)   # (B, T)
            label_raw = labels.expand(-1, T)  # labels 形状 (B,1) -> (B, T)

            # 逐时间步收集（只取 >=8 月有效步，按 t 索引对齐）
            pad_mask = torch.arange(T, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
            month_mask = month_ids >= 8
            valid_mask = pad_mask & month_mask  # (B, T)

            # 提前预报节点:每节点取"截至该日期前最后一个有效时间步"的预测。
            # 序列锚定在 11 月 30 日结束,故位置 t 对应日历日 = 11/30 - (seq_len-1-t) 天。
            for b in range(B):
                sl = int(seq_lens[b].item())
                yr = int(years[b])
                end_d = _date(yr, 11, 30)
                lab = float(label_raw[b, 0].item())
                for (mm, dd) in cutoff_list:
                    t_idx = sl - 1 - (end_d - _date(yr, mm, dd)).days
                    if 0 <= t_idx < sl:
                        node_preds[(mm, dd)].append(float(pred_raw[b, t_idx].item()))
                        node_labels[(mm, dd)].append(lab)

            for t in range(T):
                t_mask = valid_mask[:, t]
                if t_mask.any():
                    p = pred_raw[t_mask, t].detach().cpu().tolist()
                    l = label_raw[t_mask, t].detach().cpu().tolist()
                    step_preds.setdefault(t, []).extend(p)
                    step_labels.setdefault(t, []).extend(l)

    # ========== 6. 计算逐时间步指标 ==========
    print(f"[6] 计算指标")
    if not step_preds:
        print("[错误] 无有效时间步（month >= 8）")
        sys.exit(1)

    # 按 t 排序
    sorted_ts = sorted(step_preds.keys())
    t_axis = list(range(1, len(sorted_ts) + 1))  # x 轴从 1 开始连续编号
    rmse_list = []
    r2_list = []
    corr_list = []
    n_list = []

    steps = len(sorted_ts)
    for t in sorted_ts:
        p = np.array(step_preds[t])
        l = np.array(step_labels[t])
        n = len(p)
        n_list.append(n)
        if n < 2:
            rmse_list.append(float("nan"))
            r2_list.append(float("nan"))
            corr_list.append(float("nan"))
            continue

        resid = p - l
        ss_res = float((resid ** 2).sum())
        rmse_t = math.sqrt(ss_res / n)

        ss_tot = float(((l - l.mean()) ** 2).sum())
        r2_t = 1.0 - ss_res / max(ss_tot, 1e-12)

        p_c = p - p.mean()
        l_c = l - l.mean()
        cov = float((p_c * l_c).sum())
        std_p = float(np.sqrt((p_c ** 2).sum()))
        std_l = float(np.sqrt((l_c ** 2).sum()))
        corr_t = cov / max(std_p * std_l, 1e-12)
        corr_t = max(-1.0, min(1.0, corr_t))

        rmse_list.append(rmse_t)
        r2_list.append(r2_t)
        corr_list.append(corr_t)

    # 最后一步指标
    last_idx = steps - 1
    last_rmse = rmse_list[last_idx] if last_idx >= 0 else float("nan")
    last_r2 = r2_list[last_idx] if last_idx >= 0 else float("nan")
    last_corr = corr_list[last_idx] if last_idx >= 0 else float("nan")

    print(f"\n最后时间步指标 (step {t_axis[last_idx]}，共 {steps} 步):")
    print(f"  RMSE: {last_rmse:.4f}")
    print(f"  R²:   {last_r2:.4f}")
    print(f"  Corr: {last_corr:.4f}")
    print(f"  有效样本数: {n_list[last_idx] if last_idx < len(n_list) else 0}")

    # ========== 6.5 提前预报节点指标（每节点统计全部县） ==========
    cutoff_metrics: Dict[str, Dict] = {}
    if cutoff_list:
        print(f"\n提前预报节点指标 (每节点统计全部县, val_year={args.val_year}):")
        print(f"  {'节点':<10} {'RMSE':>8} {'R²':>8} {'Corr':>8} {'N':>6}")
        print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*6}")
        for (mm, dd) in cutoff_list:
            key = f"{mm:02d}-{dd:02d}"
            p = np.array(node_preds[(mm, dd)])
            l = np.array(node_labels[(mm, dd)])
            n = int(len(p))
            m = {
                "date": key,
                "n_samples": n,
                "rmse": None,
                "r2": None,
                "corr": None,
            }
            if n >= 2:
                resid = p - l
                ss_res = float((resid ** 2).sum())
                rmse_t = math.sqrt(ss_res / n)
                ss_tot = float(((l - l.mean()) ** 2).sum())
                r2_t = 1.0 - ss_res / max(ss_tot, 1e-12)
                p_c = p - p.mean()
                l_c = l - l.mean()
                cov = float((p_c * l_c).sum())
                std_p = float(np.sqrt((p_c ** 2).sum()))
                std_l = float(np.sqrt((l_c ** 2).sum()))
                corr_t = cov / max(std_p * std_l, 1e-12)
                corr_t = max(-1.0, min(1.0, corr_t))
                m["rmse"] = round(rmse_t, 6)
                m["r2"] = round(r2_t, 6)
                m["corr"] = round(corr_t, 6)
                rmse_s = f"{rmse_t:.4f}"
                r2_s = f"{r2_t:.4f}"
                corr_s = f"{corr_t:.4f}"
            else:
                rmse_s, r2_s, corr_s = "N/A", "N/A", "N/A"
            cutoff_metrics[key] = m
            print(f"  {key:<10} {rmse_s:>8} {r2_s:>8} {corr_s:>8} {n:>6}")

    # ========== 7. 绘图 ==========
    print(f"\n[7] 绘图")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(t_axis, rmse_list, color="#d62728", linewidth=1.5, marker=".")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title(f"Per-timestep Metrics (val_year={args.val_year}, ≥Aug, {steps} steps)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_axis, r2_list, color="#2ca02c", linewidth=1.5, marker=".")
    axes[1].set_ylabel("R²")
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color="gray", linestyle=":", alpha=0.5)

    axes[2].plot(t_axis, corr_list, color="#1f77b4", linewidth=1.5, marker=".")
    axes[2].set_ylabel("Corr (Pearson)")
    axes[2].set_xlabel("Time Step (sequential, ≥Aug only)")
    axes[2].grid(True, alpha=0.3)

    # 标注最后一步数值
    for ax, val, name in zip(
        axes, [last_rmse, last_r2, last_corr], ["RMSE", "R²", "Corr"]
    ):
        if not math.isnan(val):
            ax.annotate(f"{name}={val:.4f}",
                        xy=(t_axis[last_idx], val),
                        xytext=(t_axis[last_idx] + steps * 0.02, val),
                        fontsize=9,
                        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    fig.tight_layout()
    plot_path = os.path.join(output_dir, "infer_metrics.png")
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  曲线已保存: {plot_path}")

    # ========== 8. 输出 JSON ==========
    result = {
        "val_year": args.val_year,
        "num_steps": steps,
        "last_step_metrics": {
            "rmse": round(last_rmse, 6) if not math.isnan(last_rmse) else None,
            "r2": round(last_r2, 6) if not math.isnan(last_r2) else None,
            "corr": round(last_corr, 6) if not math.isnan(last_corr) else None,
            "n_samples": n_list[-1] if n_list else 0,
        },
        "cutoff_metrics": cutoff_metrics,
    }
    result_path = os.path.join(output_dir, "infer_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  结果已保存: {result_path}")
    print(f"\n{'='*50}")
    print(f"推理完成！最后时间步指标 (共 {steps} 个有效步):")
    print(f"  RMSE: {last_rmse:.6f}" if not math.isnan(last_rmse) else "  RMSE: N/A")
    print(f"  R²:   {last_r2:.6f}" if not math.isnan(last_r2) else "  R²:   N/A")
    print(f"  Corr: {last_corr:.6f}" if not math.isnan(last_corr) else "  Corr: N/A")


if __name__ == "__main__":
    infer()
