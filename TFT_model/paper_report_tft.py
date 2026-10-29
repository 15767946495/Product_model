"""PA-TFT 在新面板（29 州 741 县，2017—2022）上的论文数值：2022 折的模型比较与逐年留一汇总。

数据来源：
- `paper_figures/loyo_summary.json`（`TFT_model/loyo_summary.py` 生成）：各配置在 2022 折的 RMSE 与异常相关
- `paper_figures/panel.json`（`TFT_model/panel_report.py` 生成）：仅县固定效应基线的逐年参照
- 运行目录 `/data/raid0/hqx/TFT_out/tft`（环境变量 LOYO_RUNS 可覆盖），编号 `loyo31-{配置}-y{年}-s{种子}`
- 比较模型的运行目录见 BASELINES / LOYO_BASELINES（缺测的自动跳过）

主模型 = PA-TFT（improved 配置，学习率 1e-4，种子 0，单次运行）。其他模型同样只报告一次运行。
岭回归面板模型不在本脚本里：它只作为第5.8节置换检验的估计器，不使用测试折作预测比较。
写入 `paper_figures/tft_main.json`。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from TFT_model.paper_report import cluster_bootstrap, metrics, STATE_NAMES  # noqa: E402
from TFT_model.rs_signal_test import _state_demean, load as load_features  # noqa: E402

F = ROOT / "paper_figures"
RUNS = Path(os.environ.get("LOYO_RUNS", "/data/raid0/hqx/TFT_out/tft"))
FEATURES = "/data/raid0/hqx/train_dataset/rs_signal_features_31states.npz"
# 种子列表可用环境变量 LOYO_SEEDS 覆盖（例如单种子复核时 LOYO_SEEDS=0）
SEEDS = tuple(int(s) for s in os.environ.get("LOYO_SEEDS", "42 1 2").split())
# PVT-TFT 的失败以三个种子的重复运行报告（都退化为常数解，不存在按测试结果挑种子的问题）
BASE_SEEDS = (42, 1, 2)
YEARS = [str(y) for y in range(2017, 2023)]
TAG = "loyo31"

# 比较模型的 2022 折运行目录（新面板重跑后填写；不存在的自动跳过）
BASELINES = {
    "UNet-ConvLSTM": Path("/data/raid0/hqx/baselines/unet_convlstm/loyo31-unet"),
    "GNN-RNN": Path("/data/raid0/hqx/baselines/gnn_rnn/loyo31-gnn"),
    "PhenoYieldNet": Path("/data/raid0/hqx/baselines/phenoyieldnet/loyo31-pheno"),
}
# 比较模型的逐年留一目录：{模型: {年份: 运行目录}}，2022 折复用上面的目录
LOYO_BASELINES: dict[str, dict[int, Path]] = {name: {2022: path} for name, path in BASELINES.items()}


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
    loyo = json.loads((F / "loyo_summary.json").read_text(encoding="utf-8"))
    panel = json.loads((F / "panel.json").read_text(encoding="utf-8"))

    # ---- 2022 折：县均值基准（2017—2021） ----
    data = load_features(FEATURES)
    train = data["year"] <= 2021
    county_mean = {c: data["z"][train & (data["fips"] == c)].mean() for c in np.unique(data["fips"])}

    fips, truth, preds = ensemble_runs(f"{TAG}-improved-y2022-s{{s}}")
    base_log = np.array([county_mean[c] for c in fips])

    singles = {}
    for seed, p in zip(SEEDS[:len(preds)], preds):
        singles[f"improved-s{seed}"] = {**metrics(truth, p), "anom_corr": anomaly_corr(p, truth, base_log, fips)}
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
    # PVT-TFT：同一骨干 + 端到端图像编码器、无面板适配的配置；只报一行（三个种子的平均）
    base = ensemble_runs(f"{TAG}-original-y2022-s{{s}}", BASE_SEEDS)
    orig_2022 = None
    if base is not None:
        _, _, base_preds = base
        others["PVT-TFT"] = np.mean(base_preds, axis=0)
        orig_2022 = {"single_runs": {
            f"original-s{seed}": {**metrics(truth, p), "anom_corr": anomaly_corr(p, truth, base_log, fips),
                                  "pred_sd": float(p.std())}
            for seed, p in zip(BASE_SEEDS[:len(base_preds)], base_preds)},
            "single_rmse_mean": float(np.mean([metrics(truth, p)["rmse"] for p in base_preds])),
            "single_rmse_sd": float(np.std([metrics(truth, p)["rmse"] for p in base_preds], ddof=1)) if len(base_preds) > 1 else None,
            "ensemble": {**metrics(truth, np.mean(base_preds, axis=0)),
                         "anom_corr": anomaly_corr(np.mean(base_preds, axis=0), truth, base_log, fips)}}
    # 仅县固定效应：PA-TFT 在 g≡0 时的特例
    others["仅县固定效应"] = np.exp(base_log)
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

    # ---- 逐年留一 ----
    fe = np.array([panel["loyo"]["county_fe_only"]["per_year_rmse"][y] for y in YEARS])
    pan = np.array([panel["loyo"]["weather+veg"]["per_year_rmse"][y] for y in YEARS])

    def summarize(name):
        c = loyo["configs"][name]
        full = [y for y in YEARS if y in c and "ensemble" in c[y]]
        if not full:
            return None
        ens_y = np.array([c[y]["ensemble"]["rmse"] for y in full])
        row = {"years": full,
               "ensemble_by_year": {y: c[y]["ensemble"]["rmse"] for y in full},
               "single_mean_by_year": {y: c[y]["single_rmse_mean"] for y in full},
               "single_sd_by_year": {y: c[y]["single_rmse_sd"] for y in full},
               "ensemble_anom_corr_by_year": {y: c[y]["ensemble"]["anom_corr"] for y in full},
               "ensemble_mean": float(ens_y.mean()),
               "single_mean": float(np.mean([c[y]["single_rmse_mean"] for y in full]))}
        sds = [c[y]["single_rmse_sd"] for y in full if c[y]["single_rmse_sd"]]
        row["seed_sd_mean"] = float(np.mean(sds)) if sds else None
        if len(full) == 6:
            row["vs_fe"] = paired(ens_y, fe)
            row["vs_panel"] = paired(ens_y, pan)
        return row

    def paired(a, b):
        return {"diff": float((a - b).mean()), "better": int((a < b).sum()),
                "t_p": float(stats.ttest_rel(a, b).pvalue), "w_p": float(stats.wilcoxon(a, b).pvalue)}

    loyo_rows = {name: summarize(name) for name in ("improved", "original", "noffset", "nodemean", "nors")}
    loyo_rows = {k: v for k, v in loyo_rows.items() if v is not None}

    # 已有深度模型的逐年留一（只比较双方都有结果的年份）
    loyo_baselines = {}
    for name, per_fold in LOYO_BASELINES.items():
        by_year = {}
        for year, path in per_fold.items():
            if (path / "county_predictions.jsonl").exists():
                _, p, t = load(path)
                by_year[str(year)] = metrics(t, p)["rmse"]
        if by_year:
            loyo_baselines[name] = {"by_year": by_year, "mean": float(np.mean(list(by_year.values()))),
                                    "n_years": len(by_year)}

    def paired_other(ref_name, ref_by_year):
        for name, row in loyo_rows.items():
            common = sorted(set(row["ensemble_by_year"]) & set(ref_by_year))
            if len(common) < 3:
                continue
            a = np.array([row["ensemble_by_year"][y] for y in common])
            b = np.array([ref_by_year[y] for y in common])
            row.setdefault("paired_other", {})[ref_name] = {"n_years": len(common), **paired(a, b)}

    if "original" in loyo_rows:
        paired_other("PVT-TFT", loyo_rows["original"]["ensemble_by_year"])
    for model, v in loyo_baselines.items():
        paired_other(model, v["by_year"])

    out = {"main_2022": main_2022, "compare_2022": rows, "original_2022": orig_2022,
           "state_rmse_2022": {
               STATE_NAMES[s]: {"n": int((states == s).sum()),
                                "tft": float(np.sqrt(((ens - truth)[states == s] ** 2).mean())),
                                "fe": float(np.sqrt(((np.exp(base_log) - truth)[states == s] ** 2).mean()))}
               for s in np.unique(states)},
           "loyo": {"fe_by_year": dict(zip(YEARS, fe.tolist())), "panel_by_year": dict(zip(YEARS, pan.tolist())),
                    "fe_mean": float(fe.mean()), "panel_mean": float(pan.mean()),
                    "compare": loyo_baselines, "configs": loyo_rows},
           "runs": {"improved": f"{TAG}-improved", "original": f"{TAG}-original"}}
    (F / "tft_main.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    m = main_2022
    sd_txt = f"±{m['single_rmse_sd']:.2f}" if m["single_rmse_sd"] else ""
    print(f"PA-TFT 2022: 单次 {m['single_rmse_mean']:.2f}{sd_txt} "
          f"({m['single_rmse_min']:.2f}-{m['single_rmse_max']:.2f}), 集成 {m['ensemble']['rmse']:.2f}, "
          f"anom_corr {m['ensemble']['anom_corr']:.3f}")
    for name, r in rows.items():
        print(f"  vs {name:26s} RMSE {r['rmse']:6.2f}  state CI [{r['diff_state_ci'][0]:+.2f},{r['diff_state_ci'][1]:+.2f}]"
              f"  county CI [{r['diff_county_ci'][0]:+.2f},{r['diff_county_ci'][1]:+.2f}]")
    for name, row in loyo_rows.items():
        extra = ""
        if "vs_fe" in row:
            extra = f", 优于县FE {row['vs_fe']['better']}/6 (t p={row['vs_fe']['t_p']:.3f})"
        print(f"LOYO {name:9s} {len(row['years'])}折 集成均值 {row['ensemble_mean']:.2f}, 单次均值 {row['single_mean']:.2f}{extra}")
    print(f"参考: 县FE 逐年留一均值 {fe.mean():.2f}, 线性面板 6 年 {pan.mean():.2f}")


if __name__ == "__main__":
    main()
