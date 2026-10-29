"""运行结果汇总：PA-TFT、PVT-TFT 与组件消融，与县均值基线对照。

论文只保留 2022 折、每个配置只保留种子 0 的单次运行；未跑满 SEEDS 的折不再作集成。
州内异常相关以训练折县均值为基准、去除州均值后计算。写入 paper_figures/loyo_summary.json。

运行编号为 {TAG}{配置}-y{年份}-s{种子}，默认 TAG=loyo31-（31 州新面板，输出在 /data/raid0/hqx/TFT_out）；
用环境变量 LOYO_RUNS、LOYO_TAG 指向别的批次，LOYO_SEEDS 指定种子（默认 "42 1 2"，单折口径用 "0"）。
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("LOYO_RUNS", "/data/raid0/hqx/TFT_out/tft"))
TAG = os.environ.get("LOYO_TAG", "loyo31-")
YEARS = range(2017, 2023)
# 种子列表可用环境变量 LOYO_SEEDS 覆盖（例如单种子复核时 LOYO_SEEDS=0）
SEEDS = tuple(int(s) for s in os.environ.get("LOYO_SEEDS", "42 1 2").split())
CONFIGS = {# PA-TFT：县均值偏移 + 植被指数县内去均值 + 梯度裁剪，植被指数走数值特征
           "improved": f"{TAG}improved-y{{y}}-s{{s}}",
           # 基线 TFT：第4章的直接适配配置（PVT 图像编码器端到端训练）
           "original": f"{TAG}original-y{{y}}-s{{s}}",
           # 组件消融：去掉县均值偏移 / 去掉植被指数的县内去均值
           "noffset": f"{TAG}noffset-y{{y}}-s{{s}}", "nodemean": f"{TAG}nodemean-y{{y}}-s{{s}}",
           # 去掉遥感：植被指数全部置零（同一模型结构），检验遥感的增量贡献
           "nors": f"{TAG}nors-y{{y}}-s{{s}}"}


def load_labels():
    labels = {}
    for line in open("/data/raid0/hqx/train_dataset/dataset.jsonl"):
        head = line[:600]
        fips = re.search(r'"fips":"(\d+)"', head).group(1)
        year = int(re.search(r'"year":(\d+)', head).group(1))
        labels[(fips, year)] = float(re.search(r'"yield_bu_ac":([\d.]+)', head).group(1))
    return labels


def demean(x, groups):
    x = x.copy()
    for g in set(groups):
        x[groups == g] -= x[groups == g].mean()
    return x


def score(pred, truth, fips, year, labels):
    train = [u for u in YEARS if u != year]
    county_mean = np.array([np.mean([np.log(labels[(c, u)]) for u in train]) for c in fips])
    states = np.array([c[:2] for c in fips])
    anomaly_hat, anomaly = np.log(pred) - county_mean, np.log(truth) - county_mean
    corr = float(np.corrcoef(demean(anomaly_hat, states), demean(anomaly, states))[0, 1]) if anomaly_hat.std() > 1e-6 else 0.0
    ss_res = float(((truth - pred) ** 2).sum())
    ss_tot = float(((truth - truth.mean()) ** 2).sum())
    return {"rmse": float(np.sqrt(((pred - truth) ** 2).mean())), "bias": float((pred - truth).mean()),
            "mae": float(np.abs(pred - truth).mean()), "r2": 1.0 - ss_res / ss_tot,
            "pred_sd": float(pred.std()), "anom_corr": corr}


def read(run):
    path = RUNS / run / "county_predictions.jsonl"
    if not path.exists():
        return None
    rows = sorted((json.loads(line) for line in open(path)), key=lambda r: r["fips"])
    return (np.array([r["fips"] for r in rows]), np.array([r["y_pred"] for r in rows]), np.array([r["y_true"] for r in rows]))


def main():
    labels = load_labels()
    # 线性面板模型的逐年留一结果（TFT_model/panel_report.py 写入 panel.json 的 loyo 字段）
    panel = json.loads((ROOT / "paper_figures" / "panel.json").read_text())["loyo"]
    out = {"panel": panel, "configs": {}}
    for name, pattern in CONFIGS.items():
        per_year = {}
        for year in YEARS:
            runs = [read(pattern.format(y=year, s=s)) for s in SEEDS]
            runs = [(s, r) for s, r in zip(SEEDS, runs) if r is not None]
            if not runs:
                continue
            fips, _, truth = runs[0][1]
            single = {s: score(r[1], truth, fips, year, labels) for s, r in runs}
            entry = {"seeds": sorted(single), "single": single,
                     "single_rmse_mean": float(np.mean([v["rmse"] for v in single.values()])),
                     "single_rmse_sd": float(np.std([v["rmse"] for v in single.values()], ddof=1)) if len(single) > 1 else None}
            if len(runs) == len(SEEDS):
                entry["ensemble"] = score(np.mean([r[1] for _, r in runs], axis=0), truth, fips, year, labels)
            per_year[year] = entry
        out["configs"][name] = per_year
    (ROOT / "paper_figures" / "loyo_summary.json").write_text(json.dumps(out, indent=1))
    ablation = {}
    for key, name in (("improved", "improved"), ("noffset", "noffset"), ("nodemean", "nodemean"), ("nors", "nors")):
        rows = out["configs"][name]
        if all(y in rows and "ensemble" in rows[y] for y in YEARS):
            ablation[key] = {
                "ensemble_by_year": {str(y): rows[y]["ensemble"]["rmse"] for y in YEARS},
                "ensemble_mean": float(np.mean([rows[y]["ensemble"]["rmse"] for y in YEARS])),
                "single_mean": float(np.mean([rows[y]["single_rmse_mean"] for y in YEARS])),
                "anom_corr_mean": float(np.mean([rows[y]["ensemble"]["anom_corr"] for y in YEARS])),
            }
    if all(k in ablation for k in ("improved", "noffset", "nodemean")):
        from scipy import stats
        full = np.array([ablation["improved"]["ensemble_by_year"][str(y)] for y in YEARS])
        ablation["tests_vs_improved"] = {}
        for key in [k for k in ("noffset", "nodemean", "nors") if k in ablation]:
            e = np.array([ablation[key]["ensemble_by_year"][str(y)] for y in YEARS])
            ablation["tests_vs_improved"][key] = {
                "diff": float((e - full).mean()), "worse_years": int((e > full).sum()),
                "t_p": float(stats.ttest_rel(e, full).pvalue), "w_p": float(stats.wilcoxon(e, full).pvalue)}
        ablation["seed_sd_mean"] = {key: float(np.mean([out["configs"][name][y]["single_rmse_sd"] for y in YEARS]))
                                    for key, name in (("improved", "improved"), ("noffset", "noffset"), ("nodemean", "nodemean"), ("nors", "nors"))
                                    if key in ablation}
        (ROOT / "paper_figures" / "loyo_ablation.json").write_text(json.dumps(ablation, indent=1))

    fe, pan = panel["county_fe_only"]["per_year_rmse"], panel["weather+veg"]["per_year_rmse"]
    print(f"{'year':4s} | {'FE':>6s} {'panel':>6s} | " + " | ".join(f"{n:>21s}" for n in CONFIGS))
    print(f"{'':4s} | {'':6s} {'':6s} | " + " | ".join(f"{'single mean±sd  ens':>21s}" for _ in CONFIGS))
    for year in YEARS:
        cells = []
        for name in CONFIGS:
            e = out["configs"][name].get(year)
            if e is None:
                cells.append(f"{'-':>21s}"); continue
            sd = f"±{e['single_rmse_sd']:.2f}" if e["single_rmse_sd"] is not None else "     "
            ens = f"{e['ensemble']['rmse']:6.2f}" if "ensemble" in e else "   n/a"
            cells.append(f"{e['single_rmse_mean']:6.2f}{sd} {ens} (n={len(e['seeds'])})"[:21].rjust(21))
        print(f"{year} | {fe[str(year)]:6.2f} {pan[str(year)]:6.2f} | " + " | ".join(cells))
    for name in CONFIGS:
        rows = out["configs"][name]
        full = [y for y in YEARS if y in rows and "ensemble" in rows[y]]
        if len(full) == 6:
            ens = [rows[y]["ensemble"]["rmse"] for y in YEARS]
            single = [rows[y]["single_rmse_mean"] for y in YEARS]
            print(f"{name}: 6-yr mean single {np.mean(single):.2f}, ensemble {np.mean(ens):.2f}; ensemble beats FE {sum(e < fe[str(y)] for e, y in zip(ens, YEARS))}/6, "
                  f"beats panel {sum(e < pan[str(y)] for e, y in zip(ens, YEARS))}/6; ensemble anom corr mean "
                  f"{np.mean([rows[y]['ensemble']['anom_corr'] for y in YEARS]):.3f}")
        else:
            print(f"{name}: {len(full)}/6 years have all {len(SEEDS)} seeds")
    print(f"reference: FE 6-yr mean {panel['county_fe_only']['mean_of_years']:.2f}, panel {panel['weather+veg']['mean_of_years']:.2f}")


if __name__ == "__main__":
    sys.exit(main())
