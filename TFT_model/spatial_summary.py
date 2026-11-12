"""县分组五折空间交叉验证的汇总：PA-TFT 与三个深度基线。

划分：741 个县按州内分层随机分成 5 折。每折留出 1 折县，这些县 2017—2022 年的全部
县年都不参与训练；训练集是其余 4 折县 2017—2022 年的全部县年，测试集是留出县的
全部县年。标准化统计量在每折内部只用该折训练县估计。5 折的留出县互不重叠，拼接后
覆盖全部 741 个县 × 6 个年份。

对比对象是同一划分下重新训练的深度基线：
  UNet-ConvLSTM   year-level，6 年一起评测；
  PhenoYieldNet   stage-1 无标签适配（该折训练县）+ 全部县年微调，6 年一起评测；
  GNN-RNN         4 年窗口模型，数据从 2017 年开始，只能给出 2019—2022 年的预测。

州内异常相关沿用第2.6节的定义：以该县其余年份的对数单产均值为基准，先减去县水平，
再在州—年单元格内去均值后求相关。

用法：python -I TFT_model/spatial_summary.py
输出：paper_figures/spatial_cv.json、paper_figures/fig5_4_spatial.png
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path(os.environ.get("LOYO_RUNS", "/data/raid0/hqx/TFT_out/tft"))
BASE = Path(os.environ.get("SPATIAL_BASE", "/data/raid0/hqx/TFT_out/spatial_baselines"))
FOLDS = Path(os.environ.get("SPATIAL_FOLDS", "/data/raid0/hqx/train_dataset/spatial_folds"))
OUT = ROOT / "paper_figures"
TAG = os.environ.get("SPATIAL_TAG", "spatial")
SEED = int(os.environ.get("LOYO_SEEDS", "0").split()[0])
K = 5
YEARS = (2017, 2018, 2019, 2020, 2021, 2022)
GNN_YEARS = (2019, 2020, 2021, 2022)
MODELS = ("PA-TFT", "UNet-ConvLSTM", "GNN-RNN", "PhenoYieldNet")


def load_labels():
    labels = {}
    for line in open(ROOT / "train_dataset" / "dataset.jsonl", encoding="utf-8"):
        record = json.loads(line)
        labels[(record["fips"], int(record["year"]))] = float(record["yield_bu_ac"])
    return labels


def demean(x, groups):
    x = x.copy()
    for g in set(groups):
        x[groups == g] -= x[groups == g].mean()
    return x


def cells_of(rows):
    return np.array([f"{r['fips'][:2]}_{r['year']}" for r in rows])


def anomaly_corr(pred, truth, base_log, cells):
    a_hat, a = np.log(pred) - base_log, np.log(truth) - base_log
    hat, obs = demean(a_hat, cells), demean(a, cells)
    if hat.std() < 1e-9 or obs.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(hat, obs)[0, 1])


def metrics(pred, truth, base_log, cells):
    e = pred - truth
    sst = ((truth - truth.mean()) ** 2).sum()
    return {"n": int(len(pred)),
            "rmse": float(np.sqrt((e ** 2).mean())), "mae": float(np.abs(e).mean()),
            "bias": float(e.mean()), "r2": float(1 - (e ** 2).sum() / sst),
            "r": float(np.corrcoef(truth, pred)[0, 1]),
            "pred_sd": float(pred.std()),
            "anom_corr": anomaly_corr(pred, truth, base_log, cells),
            "mse": float((e ** 2).mean()), "bias_sq": float(e.mean() ** 2),
            "err_var": float(((e - e.mean()) ** 2).mean())}


def county_base(labels, fips, year):
    """留一年份的县均值：该县其余年份对数单产的均值（只作为异常相关的基准）。"""
    others = [np.log(labels[(fips, y)]) for y in YEARS if y != year and (fips, y) in labels]
    return float(np.mean(others)) if others else float(np.mean([np.log(v) for v in labels.values()]))


def read_predictions(run_dir: Path, held: set) -> dict:
    path = run_dir / "county_predictions.jsonl"
    if not path.exists():
        raise SystemExit(f"missing run: {path}")
    out = {}
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        key = (row["fips"], int(row["year"]))
        if row["fips"] not in held:
            raise ValueError(f"{run_dir.name}: prediction outside held-out counties {key}")
        if key in out:
            raise ValueError(f"{run_dir.name}: duplicate prediction {key}")
        out[key] = (float(row["y_true"]), float(row["y_pred"]))
    return out


def load_model(name: str, fold: int, held: set) -> dict:
    if name == "PA-TFT":
        return read_predictions(RUNS / f"{TAG}-main-f{fold}-allyears-s{SEED}", held)
    if name == "UNet-ConvLSTM":
        return read_predictions(BASE / "unet_convlstm" / f"spatial-unet-fold{fold}", held)
    if name == "PhenoYieldNet":
        return read_predictions(BASE / "phenoyieldnet" / f"spatial-pheno-allyears-f{fold}", held)
    if name == "GNN-RNN":
        pooled = {}
        for year in GNN_YEARS:
            run = BASE / "gnn_rnn" / f"spatial-gnn-y{year}-f{fold}"
            if not run.exists():
                print(f"  note: {run.name} missing, GNN skips {year}")
                continue
            pooled.update(read_predictions(run, held))
        return pooled
    raise KeyError(name)


def evaluate(rows_by_model: dict, model: str, labels: dict) -> dict:
    """按年份与汇总统计；PA-TFT 在同一批县年上也给一组指标，便于成对比较。"""
    data = rows_by_model[model]
    patft = rows_by_model["PA-TFT"]

    def block(keys):
        keys = sorted(keys)
        r = [{"fips": f, "year": y} for f, y in keys]
        truth = np.array([data[k][0] for k in keys])
        pred = np.array([data[k][1] for k in keys])
        base = np.array([county_base(labels, x["fips"], x["year"]) for x in r])
        out = metrics(pred, truth, base, cells_of(r))
        matched = [k for k in keys if k in patft]
        if model != "PA-TFT" and matched:
            mr = [{"fips": f, "year": y} for f, y in matched]
            mt = np.array([patft[k][0] for k in matched])
            mp = np.array([patft[k][1] for k in matched])
            mb = np.array([county_base(labels, x["fips"], x["year"]) for x in mr])
            out["patft_matched"] = metrics(mp, mt, mb, cells_of(mr))
        return out

    per_year = {str(y): block([k for k in data if k[1] == y]) for y in YEARS if any(k[1] == y for k in data)}
    return {"pooled": block(data.keys()), "per_year": per_year}


def main():
    labels = load_labels()
    manifest = json.loads((FOLDS / "folds.json").read_text(encoding="utf-8"))
    rows_by_model, per_fold = {}, {}
    for name in MODELS:
        rows_by_model[name] = {}
        per_fold[name] = []
    for fold in range(K):
        held = set(manifest["folds"][fold]["counties"])
        for name in MODELS:
            data = load_model(name, fold, held)
            rows_by_model[name].update(data)
            print(f"fold {fold}  {name:14s} {len(data):4d} predictions")

    report = {"protocol": {"folds": K, "seed": SEED, "split": "county-grouped, stratified within state",
                           "train": "4/5 counties, all years 2017-2022",
                           "test": "held-out counties, all years 2017-2022",
                           "standardisation": "global mean/std pooled over that fold's training counties and all years",
                           "gnn_years": list(GNN_YEARS)},
              "models": {}, "per_year": {}, "n_by_year": {}}
    for name in MODELS:
        result = evaluate(rows_by_model, name, labels)
        report["models"][name] = result
        m = result["pooled"]
        print(f"{name:14s} n={m['n']:4d}  RMSE {m['rmse']:.2f}  MAE {m['mae']:.2f}  "
              f"R2 {m['r2']:.3f}  bias {m['bias']:+.2f}")
    for year in YEARS:
        key = str(year)
        report["per_year"][key] = {name: report["models"][name]["per_year"].get(key) for name in MODELS}
        report["n_by_year"][key] = max(
            (v["n"] for v in report["per_year"][key].values() if v), default=0)
        line = "  ".join(f"{name.split('-')[0]} {v['rmse']:.2f}" if v else f"{name.split('-')[0]} —"
                         for name, v in report["per_year"][key].items())
        print(f"{key}: {line}")

    per_fold = []
    for name in MODELS:
        folds, pooled = [], []
        for fold in range(K):
            held = set(manifest["folds"][fold]["counties"])
            data = {k: v for k, v in rows_by_model[name].items() if k[0] in held}
            r = [{"fips": f, "year": y} for f, y in sorted(data)]
            truth = np.array([data[k][0] for k in sorted(data)])
            pred = np.array([data[k][1] for k in sorted(data)])
            base = np.array([county_base(labels, x["fips"], x["year"]) for x in r])
            folds.append({"fold": fold, "counties": len(held), **metrics(pred, truth, base, cells_of(r))})
        per_fold.append({"model": name, "folds": folds})
    report["per_fold"] = per_fold

    by_state = []
    for name in MODELS:
        data = rows_by_model[name]
        states = np.array([f[:2] for f, _ in data])
        keys = sorted(data)
        truth = np.array([data[k][0] for k in keys])
        pred = np.array([data[k][1] for k in keys])
        states = np.array([k[0][:2] for k in keys])
        for s in sorted(set(states)):
            sel = states == s
            by_state.append({"model": name, "state": s, "n": int(sel.sum()),
                             "rmse": float(np.sqrt(((pred[sel] - truth[sel]) ** 2).mean())),
                             "bias": float((pred[sel] - truth[sel]).mean())})
    report["by_state"] = by_state

    OUT.mkdir(exist_ok=True)
    (OUT / "spatial_cv.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'spatial_cv.json'}")
    try:
        plot(report)
    except Exception as error:  # 画图失败不影响数值汇总
        print(f"figure skipped: {error}")


def plot(report):
    import sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from TFT_model.attention_analysis import _configure_chinese_font
    from TFT_model.figures_2022 import BLUE, BLUE_DARK, ORANGE, INK, MUTED, style

    _configure_chinese_font()
    colors = {"PA-TFT": BLUE_DARK, "UNet-ConvLSTM": BLUE, "GNN-RNN": ORANGE, "PhenoYieldNet": MUTED}
    names = {"PA-TFT": "PAF-Net", "UNet-ConvLSTM": "UNet-ConvLSTM",
             "GNN-RNN": "GNN-RNN", "PhenoYieldNet": "PhenoYieldNet"}
    years = [str(y) for y in YEARS]
    fig, ax = plt.subplots(figsize=(6.8, 3.5), constrained_layout=True)
    xs = np.arange(len(years))
    width = 0.2
    for j, model in enumerate(MODELS):
        values = [report["per_year"][y][model]["rmse"] if report["per_year"][y][model] else np.nan
                  for y in years]
        offset = (j - (len(MODELS) - 1) / 2) * width
        ax.bar(xs + offset, values, width, color=colors[model], zorder=3, label=names[model])
        for x, value in zip(xs + offset, values):
            if np.isfinite(value):
                ax.text(x, value + 0.4, f"{value:.1f}", ha="center", fontsize=7, color=INK,
                        rotation=90, va="bottom", zorder=5)
    ax.set_xticks(xs)
    ax.set_xticklabels(years, fontsize=9)
    ax.set_xlabel("测试年份（留出县的全部县年）", fontsize=10, color=INK)
    ax.set_ylabel("RMSE（bu/ac）", fontsize=10, color=INK)
    top = max(v["rmse"] for y in years for v in report["per_year"][y].values() if v)
    ax.set_ylim(0, top * 1.32)
    style(ax)
    ax.grid(False, axis="x")
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center")
    fig.savefig(OUT / "fig5_4_spatial.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
