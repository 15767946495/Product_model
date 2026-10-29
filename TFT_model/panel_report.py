"""线性面板模型的独立报告（不含任何深度模型），用于新面板。

与 paper_report.py 使用同一套函数与同一套滚动起点协议（forward_only=True），
只输出面板部分：滚动起点逐折与合并、2022 州分解、季内截止、月份消融、系数范数。

用法：
  python TFT_model/panel_report.py --features /data/raid0/hqx/train_dataset/rs_signal_features_31states.npz \
      --output paper_figures/panel.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from TFT_model.paper_report import (  # noqa: E402
    SETS, STATE_NAMES, blocks_for, evaluate_years, fold, metrics, _prepare_block,
)
from TFT_model.rs_signal_test import _ridge_by_year_cv, fold_predict, load, within  # noqa: E402

SETS_PANEL = ("county_fe_only", "weather", "veg_indices", "weather+veg")


def main(args):
    data = load(args.features)
    roll_years = (2020, 2021, 2022)
    report = {}

    report["rolling"] = {name: evaluate_years(data, name, roll_years) for name in SETS}
    for name in SETS:
        print(name, json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                                for k, v in report["rolling"][name]["pooled"].items()}), flush=True)

    # 逐年留一（训练折含除测试年之外的全部年份），与 PA-TFT 的 LOYO 协议同口径
    report["loyo"] = {}
    for name in ("county_fe_only", "weather", "weather+veg"):
        blocks = blocks_for(data, name)
        per_year = {}
        for year in range(2017, 2023):
            if blocks:
                test, pred, _ = fold_predict(blocks, data["z"], data["fips"], data["year"], year, forward_only=False)
            else:
                train = data["year"] != year
                z_mean = {f: data["z"][train & (data["fips"] == f)].mean() for f in np.unique(data["fips"][train])}
                test = data["year"] == year
                pred = np.array([z_mean[f] for f in data["fips"][test]])
            per_year[str(year)] = float(np.sqrt(((np.exp(pred) - np.exp(data["z"][test])) ** 2).mean()))
        report["loyo"][name] = {"per_year_rmse": per_year,
                                "mean_of_years": float(np.mean(list(per_year.values())))}
        print("loyo", name, round(report["loyo"][name]["mean_of_years"], 3), flush=True)

    idx22, pred22, base22, alpha22 = fold(data, blocks_for(data, "weather+veg"), 2022)
    fips22, y22 = data["fips"][idx22], np.exp(data["z"][idx22])
    _, fe_pred, _, _ = fold(data, [], 2022)
    states = np.array([f[:2] for f in fips22])
    report["state_rmse_2022"] = {
        STATE_NAMES[s]: {"n": int((states == s).sum()),
                         "weather+veg": float(np.sqrt(((np.exp(pred22) - y22)[states == s] ** 2).mean())),
                         "county_fe_only": float(np.sqrt(((np.exp(fe_pred) - y22)[states == s] ** 2).mean()))}
        for s in np.unique(states)}
    report["main_2022"] = {**metrics(y22, np.exp(pred22)), "alpha": alpha22, "n": int(len(idx22)),
                           "anom_corr": report["rolling"]["weather+veg"]["per_year"][2022]["anom_corr"]}
    print("2022", json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                              for k, v in report["main_2022"].items()}), flush=True)

    report["cutoff"] = {}
    for months in range(1, 7):
        report["cutoff"][months + 3] = {name: evaluate_years(data, name, roll_years, months)
                                        for name in ("county_fe_only", "weather", "weather+veg")}
        print("cutoff", months + 3, round(report["cutoff"][months + 3]["weather+veg"]["pooled"]["rmse"], 3), flush=True)

    weather_block, veg_block = blocks_for(data, "weather+veg")
    X, valid, _, _ = veg_block
    k = X.shape[1] // 6
    report["month_ablation"] = {}
    for m in range(6):
        sel = np.zeros(X.shape[1], bool)
        sel[m * k:(m + 1) * k] = True
        for tag, keep in (("only", sel), ("drop", ~sel)):
            blk = [weather_block, (X[:, keep], valid[:, keep], True, None)]
            ys, ps = [], []
            for year in roll_years:
                idx, pred, _, _ = fold(data, blk, year)
                ys.append(np.exp(data["z"][idx]))
                ps.append(np.exp(pred))
            report["month_ablation"][f"{tag}_{m + 4}"] = metrics(np.concatenate(ys), np.concatenate(ps))
    print("month ablation done", flush=True)

    train = data["year"] < 2022
    parts = [_prepare_block(Xb, vb, data["fips"], train, nb, 0)
             for Xb, vb, _, nb in blocks_for(data, "weather+veg")]
    Xw = np.concatenate(parts, axis=1)
    z_mean = {f: data["z"][train & (data["fips"] == f)].mean() for f in np.unique(data["fips"][train])}
    base_all = np.array([z_mean.get(f, np.nan) for f in data["fips"]])
    model, alpha = _ridge_by_year_cv(Xw[train], (data["z"] - base_all)[train], data["year"][train])
    coef = model.coef_
    names_w = [str(n) for n in np.load(args.features)["weather_features"]]
    names_v = [str(n) for n in np.load(args.features)["veg_features"]]
    layout = [(f"天气:{n}", m) for m in range(6) for n in names_w] + [(f"植被:{n}", m) for m in range(6) for n in names_v]
    kept = np.concatenate([within(Xb, data["fips"], train, vb)[train].std(0) > 1e-10
                           for Xb, vb, _, _ in blocks_for(data, "weather+veg")])
    labels = [l for l, keep_flag in zip(layout, kept) if keep_flag]
    assert len(labels) == len(coef), (len(labels), len(coef))
    by_group = {}
    for (name, month), c in zip(labels, coef):
        kind = name.split(":")[0]
        by_group.setdefault(f"{kind}|{month + 4}", []).append(c * c)
    report["coef_norm_by_group"] = {k_: float(np.sqrt(sum(v))) for k_, v in by_group.items()}
    top = sorted(zip(labels, coef), key=lambda t: -abs(t[1]))[:12]
    report["top_coef"] = [{"feature": f"{n}@{m + 4}月", "coef": float(c)} for (n, m), c in top]
    report["alpha_final"] = alpha
    report["n_features_kept"] = int(Xw.shape[1])

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", out, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="/data/raid0/hqx/train_dataset/rs_signal_features_31states.npz")
    parser.add_argument("--output", default=str(_PROJECT_ROOT / "paper_figures" / "panel.json"))
    main(parser.parse_args())
