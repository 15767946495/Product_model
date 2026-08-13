#!/usr/bin/env python3
"""
四组消融: 变量选择位置(county/grid) × 空间聚合(mean/additive)，农学构造特征固定开启。

每组:
  1) python train.py --output_dir <combo_dir> --val_year <V> <开关> ...
  2) python infer.py  --output_dir <combo_dir> --val_year <V>
组合编号: 0=county/mean, 1=county/additive, 2=grid/mean, 3=grid/additive。
空间注意力逐时间步执行，CLS 与网格 token 均使用 WeatherFormer 四槽时空加性编码。
输出目录前缀带 "gridvsn2"，与旧实验隔离。

已训练完成的组(目录里已有 best_model.pth)默认跳过,加 --force 重训。

用法:
  python ablation_rope.py --constructed            # 4 组联合消融
  python ablation_rope.py --constructed --val_year 2022 --force
  python ablation_rope.py --constructed --combo 0 1
  python ablation_rope.py --constructed --infer-only
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent


def constructed_combo(i: int) -> dict:
    combos = [
        {"use_constructed": True, "spatial_mode": "mean", "variable_selection_stage": "county"},
        {"use_constructed": True, "spatial_mode": "attention", "variable_selection_stage": "county"},
        {"use_constructed": True, "spatial_mode": "mean", "variable_selection_stage": "grid"},
        {"use_constructed": True, "spatial_mode": "attention", "variable_selection_stage": "grid"},
    ]
    return combos[i]


def run(cmd: list, log_path: Path, env=None) -> int:
    """运行子进程,stdout/stderr 存日志,返回退出码。"""
    env = dict(env or os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd, cwd=str(_THIS_DIR), env=env, stdout=f, stderr=subprocess.STDOUT,
        )
    return proc.returncode


def parse_train_best_rmse(log_path: Path):
    """从 train 日志里解析 '最佳 Val RMSE: X'。"""
    if not log_path.exists():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    m = re.search(r"最佳 Val RMSE:\s*([\d.]+)", text)
    return float(m.group(1)) if m else None


def read_infer_results(combo_dir: Path):
    p = combo_dir / "infer_results.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def parse_visible_devices(value: str) -> list[str]:
    devices = [item.strip() for item in value.split(",")]
    if not devices or any(not item for item in devices):
        raise ValueError("CUDA_VISIBLE_DEVICES must contain valid GPU ids")
    return devices


def validate_parallel_args(
    constructed: bool,
    parallel: bool,
    visible_devices: list[str],
    combo_count: int,
) -> None:
    if not parallel:
        return
    if not constructed:
        raise ValueError("--parallel requires --constructed")
    if len(visible_devices) < combo_count:
        raise ValueError(
            f"--parallel requires at least {combo_count} visible GPU(s), "
            f"but got {len(visible_devices)}"
        )


def validate_unique_combos(combos: list[int]) -> None:
    if len(combos) != len(set(combos)):
        raise ValueError("--parallel does not allow duplicate combo ids")


def run_combo(
    i: int,
    gpu_id: str | None,
    combo_fn,
    combo_name,
    args,
    base: Path,
    feat_col: str,
):
    f = combo_fn(i)
    cn = combo_name(f)
    combo_dir = base / cn
    combo_dir.mkdir(parents=True, exist_ok=True)
    ckpt = combo_dir / "best_model.pth"
    train_log = combo_dir / "train.log"

    print(f"\n{'='*64}\n  {cn}: {feat_col} 固定 WeatherFormer 时空加性编码"
          f"\n{'='*64}", flush=True)

    if gpu_id is not None:
        worker_env = dict(os.environ)
        worker_env["CUDA_VISIBLE_DEVICES"] = gpu_id
    else:
        worker_env = None

    if not args.infer_only and (args.force or not ckpt.exists()):
        cmd = [sys.executable, "train.py",
               "--val_year", args.val_year,
               "--hidden_size", str(args.hidden_size),
               "--num_heads", str(args.num_heads),
               "--num_lstm_layers", str(args.num_lstm_layers),
               "--dropout", str(args.dropout),
               "--epochs", str(args.epochs),
               "--early_stop_patience", str(args.early_stop_patience),
               "--batch_size", str(args.batch_size),
               "--lr", str(args.lr),
               "--output_dir", str(combo_dir)]
        if args.constructed:
            cmd.append("--use_constructed")
        cmd += ["--spatial_mode", f.get("spatial_mode", "attention"),
                "--variable_selection_stage", f.get("variable_selection_stage", "grid")]
        if args.use_crucial:
            cmd.append("--use_crucial")
        print(f"[训练] {' '.join(cmd)} GPU={gpu_id or 'default'}", flush=True)
        rc = run(cmd, train_log, env=worker_env)
        if rc != 0:
            print(f"  !! 训练失败 (exit={rc}),日志见 {train_log.name}", flush=True)
            return cn, {"train_failed": True}
    else:
        print(f"[跳过训练] {'已存在 best_model.pth' if ckpt.exists() else '--infer-only 模式'}", flush=True)

    if not ckpt.exists():
        print("  !! 无 checkpoint,无法推理", flush=True)
        return cn, {"no_checkpoint": True}
    cmd = [sys.executable, "infer.py", "--val_year", args.val_year,
           "--output_dir", str(combo_dir)]
    print(f"[推理] {' '.join(cmd)} GPU={gpu_id or 'default'}", flush=True)
    rc = run(cmd, combo_dir / "infer.log", env=worker_env)
    if rc != 0:
        print(f"  !! 推理失败 (exit={rc})", flush=True)
        return cn, {"infer_failed": True}

    infer_res = read_infer_results(combo_dir)
    last = infer_res.get("last_step_metrics", {}) if infer_res else {}
    return cn, {
        "use_constructed": f.get("use_constructed", False),
        "spatial_mode": f.get("spatial_mode", "attention"),
        "encoding": "none" if f["spatial_mode"] == "mean" else "additive",
        "variable_selection_stage": f.get("variable_selection_stage", "grid"),
        "best_train_val_rmse": parse_train_best_rmse(train_log),
        "last_rmse": last.get("rmse"),
        "last_r2": last.get("r2"),
        "last_corr": last.get("corr"),
        "n": last.get("n_samples"),
    }


def main():
    parser = argparse.ArgumentParser(description="联合消融: 变量选择位置 × 空间聚合方式")
    parser.add_argument("--val_year", type=str, default="2021",
                        help="验证/测试年,与 train.py/infer.py 一致(默认 2021,即网格搜索最优配置的年)")
    parser.add_argument("--constructed", action="store_true",
                        help="4 组联合消融，农学构造特征(15维)固定开启")
    parser.add_argument("--hidden_size", type=int, default=36)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--num_lstm_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--use_crucial", action="store_true")
    parser.add_argument("--combo", type=int, nargs="*", default=None,
                        help="只跑指定组合编号(默认全部,范围0-1)")
    parser.add_argument("--force", action="store_true", help="已训练完成的组也重训")
    parser.add_argument("--infer-only", action="store_true",
                        help="不训练,只对已有 checkpoint 跑推理")
    parser.add_argument("--parallel", action="store_true",
                        help="仅 --constructed: 每个组合使用一张可见 GPU 并行运行")
    args = parser.parse_args()

    if not args.constructed:
        parser.error("当前脚本仅支持 --constructed 两组空间消融")
    combo_fn, max_combo, feat_col = constructed_combo, 3, "构造"
    combo_name = lambda f: f"{f['variable_selection_stage']}_{'mean' if f['spatial_mode'] == 'mean' else 'additive'}"

    combos = args.combo if args.combo is not None else list(range(max_combo + 1))
    valid_combos = []
    for i in combos:
        if not 0 <= i <= max_combo:
            print(f"[跳过] 非法组合编号 {i} (需 0-{max_combo})")
            continue
        valid_combos.append(i)

    visible_devices = []
    if args.parallel:
        visible_value = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not visible_value:
            parser.error("--parallel requires CUDA_VISIBLE_DEVICES, e.g. 2,3,4,5,6")
        try:
            validate_unique_combos(valid_combos)
            visible_devices = parse_visible_devices(visible_value)
            validate_parallel_args(args.constructed, args.parallel, visible_devices, 0)
        except ValueError as exc:
            parser.error(str(exc))
    tag = f"gridvsn3_val{args.val_year.replace(',', '_')}_hs{args.hidden_size}_h{args.num_heads}_lstm{args.num_lstm_layers}_c15"
    base = _THIS_DIR / "train_output" / tag
    base.mkdir(parents=True, exist_ok=True)

    results = {}
    pending_combos = valid_combos

    if args.parallel:
        try:
            validate_parallel_args(
                args.constructed, args.parallel, visible_devices, len(pending_combos)
            )
        except ValueError as exc:
            parser.error(str(exc))

    if args.parallel and pending_combos:
        assignments = list(zip(pending_combos, visible_devices))
        print("[并行 GPU 映射] " + ", ".join(
            f"{combo_name(combo_fn(i))}->GPU {gpu_id}" for i, gpu_id in assignments
        ))
        with ThreadPoolExecutor(max_workers=len(assignments)) as executor:
            futures = {
                executor.submit(
                    run_combo, i, gpu_id, combo_fn, combo_name, args, base, feat_col
                ): combo_name(combo_fn(i))
                for i, gpu_id in assignments
            }
            for future in as_completed(futures):
                cn = futures[future]
                try:
                    result_name, result = future.result()
                    results[result_name] = result
                except Exception as exc:
                    print(f"  !! {cn} worker 异常: {exc}", flush=True)
                    results[cn] = {"worker_failed": True, "error": str(exc)}
    else:
        for i in pending_combos:
            cn, result = run_combo(
                i, None, combo_fn, combo_name, args, base, feat_col
            )
            results[cn] = result

    # ---------- 汇总 ----------
    out = base / "ablation_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n{'='*70}\n  汇总 ({tag})  验证年={args.val_year}\n{'='*70}")
    print(f"  {'combo':<10} {'mode':<6} {feat_col:<5} {'encoding':<10} {'train RMSE':>11} "
          f"{'test RMSE':>10} {'R2':>8} {'Corr':>7}")
    for i in combos:
        if not 0 <= i <= max_combo:
            continue
        f = combo_fn(i)
        cn = combo_name(f)
        r = results.get(cn, {})
        sm = f.get("spatial_mode", "attention")[:3]
        g = "on" if (f.get("use_constructed", False) or f.get("use_gdd", False)) else "-"
        encoding = "none" if f["spatial_mode"] == "mean" else "additive"
        t_rmse = r.get("best_train_val_rmse")
        i_rmse, i_r2, i_corr = r.get("last_rmse"), r.get("last_r2"), r.get("last_corr")
        fmt = lambda v: f"{v:.4f}" if v is not None else "  N/A"
        print(f"  {cn:<10} {sm:<6} {g:<5} {encoding:<10} {fmt(t_rmse):>11} {fmt(i_rmse):>10} "
              f"{fmt(i_r2):>8} {fmt(i_corr):>7}")
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
