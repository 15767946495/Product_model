#!/usr/bin/env python3
"""
cropnet 超参数网格搜索脚本
============================
搜索空间：
  - hidden_size: 32, 36, 48, 64
  - num_heads: 1, 2, 4
  - num_lstm_layers: 1, 2
  - dropout: 固定 0.2

约束：hidden_size 须能被 num_heads 整除，不合法组合自动跳过。
每次实验使用 MSE 损失（默认），训练完成后自动推理获取指标。

用法：
  cd cropnet_model/grid_search
  python grid_search.py
  python grid_search.py --device cpu
  python grid_search.py --dry_run
"""

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# ========== 搜索空间 ==========
HIDDEN_SIZES = [16,24,32,36]
NUM_HEADS = [1, 2]
NUM_LSTM_LAYERS = [1, 2]
DROPOUT = 0.2
BATCH_SIZE = 8
EPOCHS = 500
LR = 5e-4
WEIGHT_DECAY = 5e-4

# ========== 路径 ==========
_THIS_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = _THIS_DIR.parent / "TFT_model" / "train.py"
INFER_SCRIPT = _THIS_DIR.parent / "TFT_model" / "infer.py"
TRAIN_OUTPUT_DIR = _THIS_DIR.parent / "TFT_model" / "train_output"


def is_valid_combo(hidden_size: int, num_heads: int) -> bool:
    return hidden_size % num_heads == 0


def build_run_tag(hidden_size: int, num_heads: int, num_lstm_layers: int) -> str:
    return f"hs{hidden_size}_h{num_heads}_lstm{num_lstm_layers}"


def run_experiment(
    hidden_size: int,
    num_heads: int,
    num_lstm_layers: int,
    val_year: str,
    device: str = None,
    dry_run: bool = False,
) -> dict:
    tag = build_run_tag(hidden_size, num_heads, num_lstm_layers)
    run_tag = f"grid_{tag}"
    artifact_dir = TRAIN_OUTPUT_DIR / f"val_{val_year}_{run_tag}"

    # 统一 MSE 损失
    train_cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--val_year", val_year,
        "--epochs", str(EPOCHS),
        "--lr", str(LR),
        "--hidden_size", str(hidden_size),
        "--num_heads", str(num_heads),
        "--num_lstm_layers", str(num_lstm_layers),
        "--dropout", str(DROPOUT),
        "--batch_size", str(BATCH_SIZE),
        "--weight_decay", str(WEIGHT_DECAY),
        "--output_dir", str(artifact_dir),
    ]
    if device:
        train_cmd += ["--device", device]

    infer_cmd = [
        sys.executable, str(INFER_SCRIPT),
        "--val_year", val_year,
        "--output_dir", str(artifact_dir),
        "--batch_size", str(BATCH_SIZE),   # 与训练一致，避免推理 OOM
    ]
    if device:
        infer_cmd += ["--device", device]

    result = {
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_lstm_layers": num_lstm_layers,
        "dropout": DROPOUT,
        "run_tag": run_tag,
        "artifact_dir": str(artifact_dir),
        "loss": "mse",
    }

    if dry_run:
        print(f"[DRY_RUN] {tag}")
        print(f"          训练: {' '.join(train_cmd)}")
        print(f"          推理: {' '.join(infer_cmd)}")
        print()
        result["status"] = "dry_run"
        return result

    # ---- 训练 ----
    print(f"\n{'='*70}")
    print(f"  实验: {tag}  (hidden_size={hidden_size}, heads={num_heads}, lstm={num_lstm_layers})")
    print(f"  产物: {artifact_dir}")
    print(f"{'='*70}\n")

    t_start = time.time()
    try:
        subprocess.run(train_cmd, check=True)
        result["status"] = "success"
        result["train_elapsed_sec"] = round(time.time() - t_start, 1)
        print(f"\n[OK] {tag} 训练完成，耗时 {result['train_elapsed_sec']:.1f}s")
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - t_start
        result["status"] = "failed"
        result["elapsed_sec"] = round(elapsed, 1)
        result["error"] = str(e)
        print(f"\n[FAIL] {tag} 训练失败: {e}")
        return result

    # ---- 推理 ----
    if result["status"] == "success":
        print(f"\n  推理: {tag}")
        try:
            subprocess.run(infer_cmd, check=True)
            result["infer_status"] = "success"
        except subprocess.CalledProcessError as e:
            result["infer_status"] = "failed"
            result["infer_error"] = str(e)

    # ---- 提取指标 ----
    infer_result_path = artifact_dir / "infer_results.json"
    if infer_result_path.exists():
        with open(infer_result_path, "r", encoding="utf-8") as f:
            ir = json.load(f)
        last = ir.get("last_step_metrics", {})
        result["best_val_rmse"] = last.get("rmse")
        result["best_val_r2"] = last.get("r2")
        result["best_val_corr"] = last.get("corr")
        result["last_step_n"] = last.get("n_samples")

        rmse_str = f"{result['best_val_rmse']:.6f}" if result.get("best_val_rmse") else "N/A"
        r2_str = f"{result['best_val_r2']:.4f}" if result.get("best_val_r2") else "N/A"
        corr_str = f"{result['best_val_corr']:.4f}" if result.get("best_val_corr") else "N/A"
        print(f"\n[RESULT] {tag}  RMSE={rmse_str}  R²={r2_str}  Corr={corr_str}")
    else:
        # 没有推理结果，检查模型是否生成
        ckpt = artifact_dir / "best_model.pth"
        if ckpt.exists():
            result["best_val_rmse"] = None
            print(f"\n[RESULT] {tag} 训练完成（模型存在，无推理结果）")

    result["elapsed_sec"] = round(time.time() - t_start, 1)
    return result


def save_results(results: List[dict], output_path: str = "grid_search_results.json"):
    """保存结果并按 RMSE 排序。"""
    completed = [
        r for r in results
        if r.get("status") == "success" and r.get("best_val_rmse") is not None
    ]

    report = {
        "search_space": {
            "hidden_sizes": HIDDEN_SIZES,
            "num_heads": NUM_HEADS,
            "num_lstm_layers": NUM_LSTM_LAYERS,
            "dropout": DROPOUT,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
        },
        "total_combinations": len(results),
        "success_count": sum(1 for r in results if r.get("status") == "success"),
        "failed_count": sum(1 for r in results if r.get("status") == "failed"),
        "results": results,
    }

    if completed:
        completed.sort(key=lambda r: r["best_val_rmse"])
        report["top5"] = [
            {
                "rank": i + 1,
                "hidden_size": r["hidden_size"],
                "num_heads": r["num_heads"],
                "num_lstm_layers": r["num_lstm_layers"],
                "rmse": r["best_val_rmse"],
                "r2": r.get("best_val_r2"),
                "corr": r.get("best_val_corr"),
                "run_tag": r["run_tag"],
            }
            for i, r in enumerate(completed[:5])
        ]
    else:
        report["top5"] = []

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n[结果] 已保存到 {output_path}")
    return report


def print_summary(report: dict):
    total = report["total_combinations"]
    success = report["success_count"]
    failed = report["failed_count"]

    print(f"\n{'='*60}")
    print(f"  网格搜索完成")
    print(f"  总组合: {total} | 成功: {success} | 失败: {failed}")
    print(f"{'='*60}")

    if report["top5"]:
        print(f"\n  Top 5 最佳配置 (按 RMSE 升序):")
        print(f"  {'排名':>4}  {'hidden_size':>12}  {'num_heads':>9}  {'lstm_layers':>11}  {'RMSE':>8}  {'R²':>8}  {'Corr':>8}")
        print(f"  {'-'*4}  {'-'*12}  {'-'*9}  {'-'*11}  {'-'*8}  {'-'*8}  {'-'*8}")
        for entry in report["top5"]:
            rmse = f"{entry['rmse']:.4f}" if entry.get("rmse") else "N/A"
            r2 = f"{entry['r2']:.4f}" if entry.get("r2") else "N/A"
            corr = f"{entry['corr']:.4f}" if entry.get("corr") else "N/A"
            print(f"  {entry['rank']:>4}  {entry['hidden_size']:>12}  {entry['num_heads']:>9}  {entry['num_lstm_layers']:>11}  {rmse:>8}  {r2:>8}  {corr:>8}")

    if failed > 0:
        print(f"\n  失败实验:")
        for r in report["results"]:
            if r.get("status") == "failed":
                print(f"    - {r['run_tag']}: {r.get('error', 'unknown')}")


def main():
    parser = argparse.ArgumentParser(description="cropnet 超参数网格搜索")
    parser.add_argument("--device", type=str, default=None, help="训练/推理设备，默认自动选择 (cuda/cpu)")
    parser.add_argument("--dry_run", action="store_true", help="仅打印组合，不训练")
    parser.add_argument("--output", type=str, default="grid_search_results.json", help="结果 JSON 路径")
    parser.add_argument("--val_year", type=str, default="2021", help="验证年份（默认 2021，用于早停与超参选择；最终评估在 2022 测试集上进行）")
    args = parser.parse_args()

    all_combinations = []
    for hs, nh, nl in itertools.product(HIDDEN_SIZES, NUM_HEADS, NUM_LSTM_LAYERS):
        if is_valid_combo(hs, nh):
            all_combinations.append((hs, nh, nl))

    total = len(all_combinations)
    print(f"网格搜索: {total} 个合法组合 ({len(HIDDEN_SIZES)}×{len(NUM_HEADS)}×{len(NUM_LSTM_LAYERS)})")
    print(f"  hidden_size: {HIDDEN_SIZES}")
    print(f"  num_heads: {NUM_HEADS}")
    print(f"  num_lstm_layers: {NUM_LSTM_LAYERS}")
    print(f"  dropout: {DROPOUT} (固定)")
    print(f"  loss: MSE (固定)")
    print(f"  val_year: {args.val_year}")
    print(f"  device: {args.device}")
    if args.dry_run:
        print(f"  模式: DRY RUN\n")

    results = []
    for hs, nh, nl in all_combinations:
        result = run_experiment(
            hidden_size=hs,
            num_heads=nh,
            num_lstm_layers=nl,
            val_year=args.val_year,
            device=args.device,
            dry_run=args.dry_run,
        )
        results.append(result)

    if not args.dry_run:
        report = save_results(results, args.output)
        print_summary(report)
    else:
        print(f"\n[DRY_RUN] 共 {total} 个组合，跳过训练")


if __name__ == "__main__":
    main()
