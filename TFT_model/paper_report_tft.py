"""PA-TFT 在新面板（29 州 741 县，2017—2022）上的论文数值：2022 折的模型比较。

数据来源：
- 运行目录 `/data/raid0/hqx/TFT_out/tft`（环境变量 LOYO_RUNS 可覆盖），编号 `paper-{配置}-y{年}-s{种子}`
- 比较模型的运行目录见 BASELINES（缺测的自动跳过）

主模型 = PA-TFT（main 配置：水平目标、全局标准化、学习率 1e-4，种子 0，单次运行）。其他模型同样只报告一次运行。
岭回归面板模型不在本脚本里：它只作为第5.8节置换检验的估计器，不使用测试折作预测比较。
写入 `paper_figures/tft_main.json`。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from TFT_model.paper_report import cluster_bootstrap, metrics, STATE_NAMES  # noqa: E402
from TFT_model.rs_signal_test import _state_demean, load as load_features  # noqa: E402

F = ROOT / "paper_figures"
RUNS = Path(os.environ.get("LOYO_RUNS", "/data/raid0/hqx/TFT_out/tft"))
FEATURES = "/data/raid0/hqx/train_dataset/rs_signal_features_31states.npz"
# 种子列表可用环境变量 LOYO_SEEDS 覆盖（例如单种子复核时 LOYO_SEEDS=0）
SEEDS = tuple(int(s) for s in os.environ.get("LOYO_SEEDS", "0").split())
TAG = os.environ.get("LOYO_TAG", "paper")
# 统一协议：训练 2017--2021，各方法固定 20 轮（不使用验证集），2022 测试。县均值基准与固定效应用训练年份。
TRAIN_LAST_YEAR = 2021

# 比较模型的 2022 折运行目录（新面板重跑后填写；不存在的自动跳过）
BASELINES = {
    "UNet-ConvLSTM": Path("/data/raid0/hqx/baselines/unet_convlstm/loyo31-unet"),
    "GNN-RNN": Path("/data/raid0/hqx/baselines/gnn_rnn/diag-gnn-vi-e100"),
    "PhenoYieldNet": Path("/data/raid0/hqx/baselines/phenoyieldnet/loyo31-pheno-e20"),
}


def load(run_dir: Path):
    rows = sorted((json.loads(line) for line in open(run_dir / "county_predictions.jsonl")), key=lambda r: r["fips"])
    return (np.array([r["fips"] for r in rows]), np.array([r["y_pred"] for r in rows]),
            np.array([r["y_true"] for r in rows]))


def anomaly_corr(pred, truth, base_log, fips):
    states = np.array([f[:2] for f in fips])
    a_hat = _state_demean(np.log(pred) - base_log, states)
    a = _state_demean(np.log(truth) - base_log, states)
    return float(np.corrcoef(a_hat, a)[0, 1]) if a_hat.std() > 1e-9 else 0.0


def ensemble_runs(pattern, seeds=None):
    """读取若干种子的运行；返回 (fips, truth, [preds])，缺失的种子跳过。"""
    out = []
    for seed in (SEEDS if seeds is None else seeds):
        d = RUNS / pattern.format(s=seed)
        if (d / "county_predictions.jsonl").exists():
            out.append(load(d))
    if not out:
        return None
    fips, _, truth = out[0]
    return fips, truth, [p for _, p, _ in out]


def main():
    # 单折协议：只报告 2022 折，不再生成逐年留一汇总

    # ---- 2022 折：仅县固定效应基准（训练年份 2017—2021 的县均值） ----
    data = load_features(FEATURES)
    train = data["year"] <= TRAIN_LAST_YEAR
    county_mean = {c: data["z"][train & (data["fips"] == c)].mean() for c in np.unique(data["fips"])}

    fips, truth, preds = ensemble_runs(f"{TAG}-main-y2022-s{{s}}")
    base_log = np.array([county_mean[c] for c in fips])

    singles = {}
    for seed, p in zip(SEEDS[:len(preds)], preds):
        singles[f"main-s{seed}"] = {**metrics(truth, p), "anom_corr": anomaly_corr(p, truth, base_log, fips)}
    ens = np.mean(preds, axis=0)
    main_2022 = {"single_runs": singles, "n": int(len(fips)),
                 "single_rmse_mean": float(np.mean([v["rmse"] for v in singles.values()])),
                 "single_rmse_sd": float(np.std([v["rmse"] for v in singles.values()], ddof=1)) if len(singles) > 1 else None,
                 "single_rmse_min": float(np.min([v["rmse"] for v in singles.values()])),
                 "single_rmse_max": float(np.max([v["rmse"] for v in singles.values()])),
                 "ensemble": {**metrics(truth, ens), "anom_corr": anomaly_corr(ens, truth, base_log, fips)}}

    # ---- 2022 折：全部比较模型 ----
    others = {}
    for name, path in BASELINES.items():
        if (path / "county_predictions.jsonl").exists():
            f, p, t = load(path)
            assert (f == fips).all() and np.allclose(t, truth), name
            others[name] = p
    # 岭回归面板模型不进模型比较：它只在第5.8节的置换检验中作为估计器（全样本、按年份CV）

    rows, rng = {}, np.random.default_rng(0)
    states = np.array([c[:2] for c in fips])
    e_main = ens - truth
    for name, pred in others.items():
        lo, hi, share = cluster_bootstrap(e_main, pred - truth, states, 5000, rng, by_state=True)
        lo_c, hi_c, share_c = cluster_bootstrap(e_main, pred - truth, states, 5000, rng, by_state=False)
        rows[name] = {**metrics(truth, pred), "anom_corr": anomaly_corr(pred, truth, base_log, fips),
                      "diff_state_ci": [lo, hi], "share_neg_state": share,
                      "diff_county_ci": [lo_c, hi_c], "share_neg_county": share_c}

    out = {"main_2022": main_2022, "compare_2022": rows,
           "state_rmse_2022": {
               STATE_NAMES[s]: {
                   "n": int((states == s).sum()),
                   "PA-TFT": float(np.sqrt(((ens - truth)[states == s] ** 2).mean())),
                   **{name: float(np.sqrt(((pred - truth)[states == s] ** 2).mean()))
                      for name, pred in others.items()},
               }
               for s in np.unique(states)},
           "runs": {"main": f"{TAG}-main"}}
    (F / "tft_main.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    m = main_2022
    sd_txt = f"±{m['single_rmse_sd']:.2f}" if m["single_rmse_sd"] else ""
    print(f"PA-TFT 2022: 单次 {m['single_rmse_mean']:.2f}{sd_txt} "
          f"({m['single_rmse_min']:.2f}-{m['single_rmse_max']:.2f}), 集成 {m['ensemble']['rmse']:.2f}, "
          f"anom_corr {m['ensemble']['anom_corr']:.3f}")
    for name, r in rows.items():
        print(f"  vs {name:26s} RMSE {r['rmse']:6.2f}  state CI [{r['diff_state_ci'][0]:+.2f},{r['diff_state_ci'][1]:+.2f}]"
              f"  county CI [{r['diff_county_ci'][0]:+.2f},{r['diff_county_ci'][1]:+.2f}]")


if __name__ == "__main__":
    main()
