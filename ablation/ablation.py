#!/usr/bin/env python3
"""
消融实验：直接网格均值 vs 空间注意力聚合
==========================================
对比 TFT 编码器在两种"网格 → 县"聚合方式下的表现：
  - spatial_mode="mean"      ：县内 G 个网格按有效掩码加权平均（退化为县均值时序）
  - spatial_mode="attention" ：本文的逐特征网格空间注意力聚合（默认，完整模型）

训练/验证/测试划分与主实验完全一致：
  训练 2017-2020 / 验证 2021（早停） / 测试 2022（最终评估）
两种模式除空间聚合方式外，超参数完全相同（hidden=32, heads=2, lstm=1...）。

用法：
  python ablation.py                  # 依次训练两种模式并在 2022 测试集评估
  python ablation.py --device cpu
  python ablation.py --dry_run        # 只打印配置，不训练
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# ========== 项目内导入 ==========
_THIS_DIR = Path(__file__).resolve().parent          # ablation/
_TFT_DIR = _THIS_DIR.parent / "TFT_model"
for p in (str(_TFT_DIR), str(_THIS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from models import TFTEncoderForYieldPrediction
from data import (
    load_jsonl,
    load_grid_cache,
    load_county_soil,
    compute_grid_global_stats,
    compute_soil_stats,
    build_grid_samples,
    GridTimeSeriesDataset,
    make_grid_collate_fn,
    SOIL_DIM,
    DEFAULT_DYNAMIC_FEATURE_NAMES,
    DEFAULT_DATA_JSONL,
    DEFAULT_GRID_CACHE,
    DEFAULT_COUNTY_SOIL,
)
from train import (
    train_model,
    _split_pairs_by_year,
    filter_valid_label_pairs,
    EARLY_STOP_PATIENCE,
)

# ========== 消融配置 ==========
MODES = [
    {"name": "mean",       "spatial_mode": "mean",       "desc": "直接网格均值"},
    {"name": "attention",  "spatial_mode": "attention",  "desc": "空间注意力聚合(本文)"},
]
STATES = "minnesota,wisconsin,michigan,iowa,illinois,indiana,ohio,missouri,kentucky"

# 两种模式共用同一套固定超参（非网格搜索）
HIDDEN_SIZE = 32
NUM_HEADS = 2
NUM_LSTM_LAYERS = 1
DROPOUT = 0.2
BATCH_SIZE = 8
EPOCHS = 500
LR = 5e-4
WEIGHT_DECAY = 5e-4
VAL_YEAR = "2021"     # 验证年：早停
TEST_YEAR = "2022"    # 测试年：最终评估
SEED = 42

OUTPUT_ROOT = _THIS_DIR / "ablation_output"
INFER_SCRIPT = _TFT_DIR / "infer.py"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_build():
    """加载数据、按年划分并构建 train/val DataLoader（口径与 train.py main 一致）。"""
    jsonl_path = DEFAULT_DATA_JSONL
    grid_cache_path = DEFAULT_GRID_CACHE
    print(f"\n[数据] 加载数据：\n    jsonl: {jsonl_path}\n    网格缓存: {grid_cache_path}")
    meta_lines = load_jsonl(jsonl_path)
    cache = load_grid_cache(grid_cache_path)
    cache_entries = cache["entries"]
    if len(cache_entries) != len(meta_lines):
        raise ValueError(f"grid_cache 行数 {len(cache_entries)} 与 jsonl 行数 {len(meta_lines)} 不一致")

    soil_dict = load_county_soil(DEFAULT_COUNTY_SOIL)
    dynamic_feature_names = list(DEFAULT_DYNAMIC_FEATURE_NAMES)
    pairs = list(zip(meta_lines, cache_entries))
    state_set = {s.strip().lower() for s in STATES.split(",")}
    pairs = [p for p in pairs if str(p[0].get("State", "")).lower() in state_set]

    train_pairs, val_pairs = _split_pairs_by_year(pairs, [int(VAL_YEAR)])
    train_pairs, _ = filter_valid_label_pairs(train_pairs)
    val_pairs, _ = filter_valid_label_pairs(val_pairs)
    print(f"    训练: {len(train_pairs)}, 验证({VAL_YEAR}): {len(val_pairs)}")

    # 特征标准化统计量（只在训练年份/训练县上算，避免泄漏）
    global_stats = compute_grid_global_stats(train_pairs, dynamic_feature_names)
    train_fips = {str(m.get("FIPS", "")).zfill(5) for m, _ in train_pairs}
    soil_mean, soil_std = compute_soil_stats(soil_dict, fips_subset=train_fips)
    global_stats["soil"] = (soil_mean, soil_std)

    payload = {}
    for k, (m, s) in global_stats.items():
        payload[k] = {"mean": m.detach().cpu().tolist(), "std": s.detach().cpu().tolist()}

    collate_fn = make_grid_collate_fn(global_stats, dynamic_feature_names)
    train_samples = build_grid_samples(train_pairs, soil_dict=soil_dict, dynamic_feature_names=dynamic_feature_names)
    val_samples = build_grid_samples(val_pairs, soil_dict=soil_dict, dynamic_feature_names=dynamic_feature_names)
    train_loader = DataLoader(
        GridTimeSeriesDataset(train_samples), batch_size=BATCH_SIZE,
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        GridTimeSeriesDataset(val_samples), batch_size=BATCH_SIZE,
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )
    print(f"    Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "dynamic_feature_names": dynamic_feature_names,
        "feature_norm_payload": payload,
    }


def run_mode(mode_cfg: dict, data: dict, device: str, dry_run: bool) -> dict:
    name = mode_cfg["name"]
    spatial_mode = mode_cfg["spatial_mode"]
    out_dir = OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  消融分支: {mode_cfg['desc']}  (spatial_mode={spatial_mode})")
    print(f"  输出目录: {out_dir}")
    print(f"{'='*70}")

    if dry_run:
        print(f"[DRY_RUN] 跳过训练与推理: {name}")
        return {"name": name, "spatial_mode": spatial_mode, "desc": mode_cfg["desc"], "status": "dry_run"}

    # 特征标准化统计量（infer.py 读取）
    fn_path = out_dir / "feature_norm.json"
    with open(fn_path, "w", encoding="utf-8") as f:
        json.dump(data["feature_norm_payload"], f, ensure_ascii=False, indent=2)

    # 模型超参（infer.py 读取，含 spatial_mode）
    hparams = {
        "model_contract_version": 4,
        "hidden_size": HIDDEN_SIZE,
        "num_heads": NUM_HEADS,
        "num_lstm_layers": NUM_LSTM_LAYERS,
        "dropout": DROPOUT,
        "dynamic_feature_dim": len(data["dynamic_feature_names"]),
        "soil_dim": SOIL_DIM,
        "loss": "mse",
        "spatial_mode": spatial_mode,
    }
    with open(out_dir / "model_hparams.json", "w", encoding="utf-8") as f:
        json.dump(hparams, f, ensure_ascii=False, indent=2)

    ckpt_path = out_dir / "best_model.pth"
    curve_path = out_dir / "loss_curve.png"

    set_seed(SEED)
    model = TFTEncoderForYieldPrediction(
        soil_dim=SOIL_DIM,
        dynamic_feature_names=data["dynamic_feature_names"],
        hidden_size=HIDDEN_SIZE,
        num_lstm_layers=NUM_LSTM_LAYERS,
        dropout=DROPOUT,
        output_size=1,
        num_heads=NUM_HEADS,
        spatial_mode=spatial_mode,
    )
    print(f"    模型参数: {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.time()
    best_rmse = train_model(
        model=model,
        train_loader=data["train_loader"],
        val_loader=data["val_loader"],
        epochs=EPOCHS,
        lr=LR,
        device=device,
        ckpt_path=str(ckpt_path),
        curve_path=str(curve_path),
        weight_decay=WEIGHT_DECAY,
        early_stop_patience=EARLY_STOP_PATIENCE,
        use_crucial=False,
    )
    train_sec = round(time.time() - t0, 1)
    print(f"[OK] {name} 训练完成，最佳验证 RMSE={best_rmse:.4f}，耗时 {train_sec}s")

    # 测试集评估（2022）：子进程调用 infer.py（含 cutoff 节点指标）
    infer_cmd = [
        sys.executable, str(INFER_SCRIPT),
        "--val_year", TEST_YEAR,
        "--output_dir", str(out_dir),
        "--ckpt", str(ckpt_path),
        "--batch_size", str(BATCH_SIZE),
    ]
    if device:
        infer_cmd += ["--device", device]
    print(f"\n  测试集评估({TEST_YEAR}): {' '.join(infer_cmd)}")
    try:
        subprocess.run(infer_cmd, check=True)
        infer_status = "success"
    except subprocess.CalledProcessError as e:
        infer_status = "failed"
        print(f"[FAIL] 推理失败: {e}")

    result = {
        "name": name,
        "spatial_mode": spatial_mode,
        "desc": mode_cfg["desc"],
        "best_val_rmse": round(best_rmse, 4) if not math.isnan(best_rmse) else None,
        "train_elapsed_sec": train_sec,
        "infer_status": infer_status,
    }

    infer_result_path = out_dir / "infer_results.json"
    if infer_result_path.exists():
        with open(infer_result_path, "r", encoding="utf-8") as f:
            ir = json.load(f)
        last = ir.get("last_step_metrics", {})
        result["test_rmse"] = last.get("rmse")
        result["test_r2"] = last.get("r2")
        result["test_corr"] = last.get("corr")
        result["test_n"] = last.get("n_samples")
        result["cutoff_metrics"] = ir.get("cutoff_metrics", {})

    return result


def main():
    parser = argparse.ArgumentParser(description="消融实验：直接网格均值 vs 空间注意力聚合")
    parser.add_argument("--device", type=str, default=None, help="训练/推理设备，默认自动选择")
    parser.add_argument("--dry_run", action="store_true", help="仅打印配置，不训练")
    parser.add_argument("--output", type=str, default="ablation_results.json", help="结果 JSON 路径")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"消融分支: {[m['desc'] for m in MODES]}")
    print(f"共用超参: hidden={HIDDEN_SIZE}, heads={NUM_HEADS}, lstm={NUM_LSTM_LAYERS}, "
          f"dropout={DROPOUT}, batch={BATCH_SIZE}, epochs={EPOCHS}, lr={LR}, wd={WEIGHT_DECAY}")
    print(f"数据划分: 训练 2017-2020 / 验证 {VAL_YEAR}(早停) / 测试 {TEST_YEAR}(最终评估)")

    data = None if args.dry_run else load_and_build()

    results = []
    for mode_cfg in MODES:
        r = run_mode(mode_cfg, data, device, args.dry_run)
        results.append(r)

    if not args.dry_run:
        print(f"\n{'='*70}")
        print(f"  消融实验对比（{TEST_YEAR} 测试集）")
        print(f"{'='*70}")
        print(f"  {'聚合方式':<22} {'RMSE':>8} {'R²':>8} {'Corr':>8} {'N':>6}  {'最佳验证RMSE':>12}")
        for r in results:
            rmse = f"{r['test_rmse']:.4f}" if r.get("test_rmse") else "N/A"
            r2 = f"{r['test_r2']:.4f}" if r.get("test_r2") else "N/A"
            corr = f"{r['test_corr']:.4f}" if r.get("test_corr") else "N/A"
            n = r.get("test_n", 0)
            bvr = f"{r['best_val_rmse']:.4f}" if r.get("best_val_rmse") else "N/A"
            print(f"  {r['desc']:<22} {rmse:>8} {r2:>8} {corr:>8} {n:>6}  {bvr:>12}")

        report = {
            "modes": MODES,
            "hyperparams": {
                "hidden_size": HIDDEN_SIZE, "num_heads": NUM_HEADS,
                "num_lstm_layers": NUM_LSTM_LAYERS, "dropout": DROPOUT,
                "batch_size": BATCH_SIZE, "epochs": EPOCHS,
                "lr": LR, "weight_decay": WEIGHT_DECAY,
                "early_stop_patience": EARLY_STOP_PATIENCE,
                "val_year": VAL_YEAR, "test_year": TEST_YEAR, "seed": SEED,
            },
            "results": results,
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\n[结果] 已保存到 {args.output}")
    else:
        print(f"\n[DRY_RUN] 共 {len(MODES)} 个分支，跳过训练")


if __name__ == "__main__":
    main()
