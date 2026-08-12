#!/usr/bin/env python3
"""
8 组合消融: use_gdd × grid_rope × time_rope (每个开关开/关, 2^3 = 8 组)。

每组:
  1) python train.py --output_dir <combo_dir> --val_year <V> <开关> ...
  2) python infer.py  --output_dir <combo_dir> --val_year <V>
infer.py 会从 model_hparams.json 自动读回三个开关(旧 checkpoint 无键默认 False)。

组合编号(位图): i 的 bit2=use_gdd, bit1=grid_rope, bit0=time_rope
  g0r0t0  基线(现有无月份模型 + 加性位置编码)
  g0r0t1  仅时间 RoPE
  g0r1t0  仅网格 2D RoPE(lat/lng,CLS 空间中性,不含时间)
  g0r1t1  RoPE 全开(无 GDD)
  g1r0t0  仅 GDD 通道
  g1r0t1  GDD + 时间 RoPE
  g1r1t0  GDD + 网格 2D RoPE
  g1r1t1  GDD + RoPE 全开

注:网格 RoPE 已从 3D(time,lat,lng)改为 2D(lat/lng),去掉时间轴
(逐时间步网格注意力里时间只能当弱绝对信号,季节交给时序分支)。
输出目录前缀带 "ablation2d",与旧的 ablation_val* 目录隔离(旧 checkpoint 作废)。

已训练完成的组(目录里已有 best_model.pth)默认跳过,加 --force 重训。

--constructed 模式: 5 组联合消融(1 组网格均值 TFT 基线 + 4 组网格注意力
grid_rope × time_rope), 农学构造特征(11+4=15 维, --use_constructed)固定开启;
目录名 mean 与 r{grid}t{time}(grid_rope=False 时网格注意力用加性正余弦位置编码)。
不带该开关时保持旧 8 组合位图(bit2=use_gdd)。

用法:
  python ablation_rope.py                          # 默认 hs16_h1_lstm1, val 2021(8 组合旧口径)
  python ablation_rope.py --constructed            # 5 组: mean 基线 + r0t0/r0t1/r1t0/r1t1
  python ablation_rope.py --val_year 2022 --force  # 换验证年并重训
  python ablation_rope.py --combo 0 7              # 只跑 0 和 7 两组
  python ablation_rope.py --infer-only             # 不训练,只对已有组跑推理
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent


def combo_to_flags(i: int) -> dict:
    """i (0-7) -> {use_gdd, grid_rope, time_rope}。"""
    return {
        "use_gdd": bool((i >> 2) & 1),
        "grid_rope": bool((i >> 1) & 1),
        "time_rope": bool(i & 1),
    }


def run(cmd: list, log_path: Path) -> int:
    """运行子进程,stdout/stderr 存日志,返回退出码。"""
    env = dict(os.environ)
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


def main():
    parser = argparse.ArgumentParser(description="联合消融: 网格均值 TFT 基线 vs 网格注意力(加性正余弦/2D RoPE) × 时间 RoPE")
    parser.add_argument("--val_year", type=str, default="2021",
                        help="验证/测试年,与 train.py/infer.py 一致(默认 2021,即网格搜索最优配置的年)")
    parser.add_argument("--constructed", action="store_true",
                        help="5 组联合消融(1 组网格均值基线 mean + 4 组网格注意力 r0t0/r0t1/r1t0/r1t1), "
                             "农学构造特征(15 维)固定开启(--use_constructed); 否则旧 8 组合(bit2=use_gdd)")
    parser.add_argument("--hidden_size", type=int, default=36)
    parser.add_argument("--num_heads", type=int, default=1)
    parser.add_argument("--num_lstm_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--use_crucial", action="store_true")
    parser.add_argument("--combo", type=int, nargs="*", default=None,
                        help="只跑指定组合编号(默认全部;构造模式 0-4,旧模式 0-7)")
    parser.add_argument("--force", action="store_true", help="已训练完成的组也重训")
    parser.add_argument("--infer-only", action="store_true",
                        help="不训练,只对已有 checkpoint 跑推理")
    args = parser.parse_args()

    if args.constructed:
        # 5 组联合消融: 1 组网格均值 TFT 基线(spatial_mode=mean, 不加任何位置编码)
        # + 4 组网格注意力(网格位置编码: grid_rope=False->加性正余弦 / True->2D RoPE;
        #                    时间 RoPE 开关)。构造特征(15 维)固定开启。
        def combo_fn(i):
            if i == 0:
                return {"use_constructed": True, "spatial_mode": "mean",
                        "grid_rope": False, "time_rope": False}
            return {"use_constructed": True, "spatial_mode": "attention",
                    "grid_rope": bool(((i - 1) >> 1) & 1),
                    "time_rope": bool((i - 1) & 1)}
        max_combo, feat_col = 4, "构造"

        def combo_name(f):
            if f["spatial_mode"] == "mean":
                return "mean"
            return f"r{int(f['grid_rope'])}t{int(f['time_rope'])}"
    else:
        combo_fn, max_combo, feat_col = combo_to_flags, 7, "GDD"
        combo_name = lambda f: f"g{int(f['use_gdd'])}r{int(f['grid_rope'])}t{int(f['time_rope'])}"

    combos = args.combo if args.combo is not None else list(range(max_combo + 1))
    suffix = "_c15" if args.constructed else ""
    tag = f"ablation2d_val{args.val_year.replace(',', '_')}_hs{args.hidden_size}_h{args.num_heads}_lstm{args.num_lstm_layers}{suffix}"
    base = _THIS_DIR / "train_output" / tag
    base.mkdir(parents=True, exist_ok=True)

    results = {}
    for i in combos:
        if not 0 <= i <= max_combo:
            print(f"[跳过] 非法组合编号 {i} (需 0-{max_combo})")
            continue
        f = combo_fn(i)
        cn = combo_name(f)
        combo_dir = base / cn
        combo_dir.mkdir(parents=True, exist_ok=True)
        ckpt = combo_dir / "best_model.pth"
        train_log = combo_dir / "train.log"

        print(f"\n{'='*64}\n  {cn}: {feat_col} 固定 grid_rope={f['grid_rope']} "
              f"time_rope={f['time_rope']}\n{'='*64}")

        # ---------- 训练 ----------
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
            elif f["use_gdd"]:
                cmd.append("--use_gdd")
            cmd += ["--spatial_mode", f.get("spatial_mode", "attention")]
            if f["grid_rope"]:
                cmd.append("--grid_rope")
            if f["time_rope"]:
                cmd.append("--time_rope")
            if args.use_crucial:
                cmd.append("--use_crucial")
            print(f"[训练] {' '.join(cmd)}")
            rc = run(cmd, train_log)
            if rc != 0:
                print(f"  !! 训练失败 (exit={rc}),日志见 {train_log.name}")
                results[cn] = {"train_failed": True}
                continue
        else:
            print(f"[跳过训练] {'已存在 best_model.pth' if ckpt.exists() else '--infer-only 模式'}")

        # ---------- 推理 ----------
        if not ckpt.exists():
            print(f"  !! 无 checkpoint,无法推理")
            results[cn] = {"no_checkpoint": True}
            continue
        cmd = [sys.executable, "infer.py",
               "--val_year", args.val_year,
               "--output_dir", str(combo_dir)]
        print(f"[推理] {' '.join(cmd)}")
        rc = run(cmd, combo_dir / "infer.log")
        if rc != 0:
            print(f"  !! 推理失败 (exit={rc})")
            results[cn] = {"infer_failed": True}
            continue

        infer_res = read_infer_results(combo_dir)
        last = infer_res.get("last_step_metrics", {}) if infer_res else {}
        results[cn] = {
            "use_constructed": f.get("use_constructed", False),
            "use_gdd": f.get("use_gdd", False),
            "spatial_mode": f.get("spatial_mode", "attention"),
            "grid_rope": f["grid_rope"],
            "time_rope": f["time_rope"],
            "best_train_val_rmse": parse_train_best_rmse(train_log),
            "last_rmse": last.get("rmse"),
            "last_r2": last.get("r2"),
            "last_corr": last.get("corr"),
            "n": last.get("n_samples"),
        }
        print(f"  best_val_rmse={results[cn]['best_train_val_rmse']}  "
              f"test_rmse={last.get('rmse')}  r2={last.get('r2')}  corr={last.get('corr')}")

    # ---------- 汇总 ----------
    out = base / "ablation_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n{'='*70}\n  汇总 ({tag})  验证年={args.val_year}\n{'='*70}")
    print(f"  {'combo':<8} {'mode':<6} {feat_col:<5} {'gRoPE':<6} {'tRoPE':<6} {'train RMSE':>11} "
          f"{'test RMSE':>10} {'R2':>8} {'Corr':>7}")
    for i in combos:
        if not 0 <= i <= max_combo:
            continue
        f = combo_fn(i)
        cn = combo_name(f)
        r = results.get(cn, {})
        sm = f.get("spatial_mode", "attention")[:3]
        g = "on" if (f.get("use_constructed", False) or f.get("use_gdd", False)) else "-"
        gr = "on" if f["grid_rope"] else "-"
        tr = "on" if f["time_rope"] else "-"
        t_rmse = r.get("best_train_val_rmse")
        i_rmse, i_r2, i_corr = r.get("last_rmse"), r.get("last_r2"), r.get("last_corr")
        fmt = lambda v: f"{v:.4f}" if v is not None else "  N/A"
        print(f"  {cn:<8} {sm:<6} {g:<5} {gr:<6} {tr:<6} {fmt(t_rmse):>11} {fmt(i_rmse):>10} "
              f"{fmt(i_r2):>8} {fmt(i_corr):>7}")
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
