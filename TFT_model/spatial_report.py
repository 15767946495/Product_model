"""旧版空间交叉验证汇总（2022 单年、全部年份训练、全样本统计量）。

前向滚动起点的多年度版本见 ``spatial_forward_report.py``；本脚本保留用于复现
论文表5-2/5-3的旧口径，不再作为新实验的入口。

口径（第5.2.2、5.2.3节）：741 个县按州分层分成 5 折，每折以留出的 1 折县为测试集、
其余县的全部年份为训练集。5 折的测试县互不重叠且合起来等于全部县，因此把 5 折的
预测拼起来，每个县年恰好对应一次预测。

**只报告 2022 年子集。** 县均值偏移与仅县固定效应必须只用目标年之前的单产（纯前向），
2022 年的偏移量是 2017—2021 年的均值；2017—2021 各行若要同样处理，需要更早年份的
单产，而 2017 年没有任何前史。留一年份法会让 2017—2021 的偏移量掺入其后的年份，
是前视，因此不报告这些年。

GNN-RNN 与 PhenoYieldNet 只能预测 2022（五年窗口与单测试年适配协议），本来也只覆盖 2022。

仅县固定效应基线：预测值为该县 2022 年之前各年对数单产的均值（与 PA-TFT 使用的
县均值偏移同一定义，目标年份不进入自己的基线）。

冷启动口径（表5-3）：完全不用留出县的历史单产，基线为"本折训练县同州同年均值"。

用法：
  python -I TFT_model/spatial_report.py --output paper_figures/spatial_cv.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FOLDS = Path("/data/raid0/hqx/train_dataset/spatial_folds/folds.json")
DEFAULT_DATASET = Path("/data/raid0/hqx/train_dataset/dataset.jsonl")
DEFAULT_RUNS = Path("/data/raid0/hqx/TFT_out")

FIPS = re.compile(r'"fips":"(\d+)"')
YEAR = re.compile(r'"year":(\d+)')
YIELD = re.compile(r'"yield_bu_ac":([\d.]+)')

# 模型 -> (相对 run 根目录的路径模板, 覆盖年份)
MODELS = {
    "PA-TFT": ("tft/spatial-improved-fold{fold}-s0", None),
    "PA-TFT（无县均值偏移）": ("tft/spatial-noffset-fold{fold}-s0", None),
    "UNet-ConvLSTM": ("spatial_baselines/unet_convlstm/spatial-unet-fold{fold}", None),
    "GNN-RNN": ("spatial_baselines/gnn_rnn/spatial-gnn-fold{fold}", (2022,)),
    "PhenoYieldNet": ("spatial_baselines/phenoyieldnet/spatial-pheno-fold{fold}", (2022,)),
}
COUNTY_FE = "仅县固定效应"
STATE_YEAR = "同州同年训练县均值"


def load_labels(path: Path) -> dict[tuple[str, int], float]:
    labels = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            head = line[:600]
            labels[(FIPS.search(head).group(1), int(YEAR.search(head).group(1)))] = float(
                YIELD.search(head).group(1))
    return labels


def load_predictions(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"missing predictions: {path}")
    return [json.loads(line) for line in path.open("r", encoding="utf-8")]


def county_history_log_mean(labels: dict[tuple[str, int], float]) -> dict[tuple[str, int], float]:
    """该县在目标年之前各年对数单产的均值（目标年及其后年份都不参与）。

    只对至少有一个先前年份的县年给出；2017 年没有前史，因此不在结果里。
    """
    by_county: dict[str, dict[int, float]] = {}
    for (fips, year), value in labels.items():
        by_county.setdefault(fips, {})[year] = np.log(float(value))
    out = {}
    for fips, per_year in by_county.items():
        for year in per_year:
            past = [v for y, v in per_year.items() if y < year]
            if past:
                out[(fips, year)] = float(np.mean(past))
    return out


def county_history_prediction(labels: dict[tuple[str, int], float]) -> dict[tuple[str, int], float]:
    """仅县固定效应：该县目标年之前各年对数单产均值还原到原始尺度。"""
    return {key: float(np.exp(value)) for key, value in county_history_log_mean(labels).items()}


def within_cell_demean(values: np.ndarray, cells: list[tuple[str, int]]) -> np.ndarray:
    """按 (州, 年份) 单元格去均值。"""
    result = values.copy()
    groups: dict[tuple[str, int], list[int]] = {}
    for index, cell in enumerate(cells):
        groups.setdefault(cell, []).append(index)
    for members in groups.values():
        result[members] -= result[members].mean()
    return result


def metrics(rows: list[dict], county_base: dict[tuple[str, int], float]) -> dict:
    """county_base 为留一年份的县均值（目标年份不进入自己的基线）。

    仅县固定效应的预测值恰等于该基线，其距平恒为 0，相关系数按定义记为 0。
    """
    truth = np.array([r["y_true"] for r in rows], dtype=np.float64)
    pred = np.array([r["y_pred"] for r in rows], dtype=np.float64)
    cells = [(r["fips"][:2], int(r["year"])) for r in rows]
    base = np.array([county_base[(r["fips"], int(r["year"]))] for r in rows], dtype=np.float64)
    anomaly_hat = np.log(np.maximum(pred, 1e-6)) - base
    anomaly = np.log(np.maximum(truth, 1e-6)) - base
    hat = within_cell_demean(anomaly_hat, cells)
    obs = within_cell_demean(anomaly, cells)
    corr = float(np.corrcoef(hat, obs)[0, 1]) if hat.std() > 1e-6 and obs.std() > 1e-6 else 0.0
    error = pred - truth
    return {
        "n": len(rows),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "mae": float(np.abs(error).mean()),
        "bias": float(error.mean()),
        "r2": float(1.0 - (error ** 2).sum() / ((truth - truth.mean()) ** 2).sum()),
        "r": float(np.corrcoef(pred, truth)[0, 1]),
        "anom_corr": corr,
        "pred_sd": float(pred.std()),
    }


def cold_start_rows(labels: dict[tuple[str, int], float], manifest: dict, fold: int) -> list[dict]:
    """冷启动基线：完全不用留出县的信息，只用本折训练县同州同年的平均单产。

    这是"对该县一无所知"时能给出的最强水平基准（比全国同年均值强得多）。
    """
    held = set(manifest["folds"][fold]["counties"])
    cells: dict[tuple[str, int], list[float]] = {}
    national: dict[int, list[float]] = {}
    for (fips, year), value in labels.items():
        if fips in held:
            continue
        cells.setdefault((fips[:2], year), []).append(np.log(value))
        national.setdefault(year, []).append(np.log(value))
    rows = []
    for (fips, year), value in labels.items():
        if fips not in held:
            continue
        pool = cells.get((fips[:2], year)) or national[year]
        rows.append({"fips": fips, "year": year, "y_true": value,
                     "y_pred": float(np.exp(np.mean(pool)))})
    return rows


def subset(rows: list[dict], years) -> list[dict]:
    return [r for r in rows if int(r["year"]) in years]


def main(args) -> None:
    runs = Path(args.runs_root).resolve()
    manifest = json.loads(Path(args.folds).read_text(encoding="utf-8"))
    labels = load_labels(Path(args.dataset_jsonl).resolve())
    n_folds = int(manifest["k"])

    county_base = county_history_log_mean(labels)
    fe_prediction = county_history_prediction(labels)
    fe_rows = [{"fips": f, "year": y, "y_true": labels[(f, y)], "y_pred": p}
               for (f, y), p in sorted(fe_prediction.items())]

    predictions: dict[str, list[dict]] = {COUNTY_FE: fe_rows}
    coverage: dict[str, tuple | None] = {COUNTY_FE: None}
    # 冷启动基线（完全不用留出县的历史单产），与模型行同一口径汇总
    predictions[STATE_YEAR] = sorted(
        (row for fold in range(n_folds) for row in cold_start_rows(labels, manifest, fold)),
        key=lambda r: (r["fips"], r["year"]))
    coverage[STATE_YEAR] = None
    for name, (template, years) in MODELS.items():
        rows, seen = [], set()
        for fold in range(n_folds):
            run_dir = runs / template.format(fold=fold)
            for row in load_predictions(run_dir / "county_predictions.jsonl"):
                key = (row["fips"], int(row["year"]))
                if key in seen:
                    raise ValueError(f"{name}: duplicate prediction for {key}")
                seen.add(key)
                rows.append({"fips": row["fips"], "year": int(row["year"]),
                             "y_true": float(row["y_true"]), "y_pred": float(row["y_pred"])})
        expected = sum(len(manifest["folds"][f]["counties"]) * (len(years) if years else 6)
                       for f in range(n_folds))
        if len(rows) != expected:
            raise ValueError(f"{name}: {len(rows)} predictions, expected {expected}")
        predictions[name] = sorted(rows, key=lambda r: (r["fips"], r["year"]))
        coverage[name] = years

    report = {"folds": {"k": n_folds, "seed": manifest["seed"],
                        "counties": [len(f["counties"]) for f in manifest["folds"]]},
              "reported_years": [2022],
              "note": "只报告2022年子集：县均值偏移与仅县固定效应只使用目标年之前"
                      "（2017—2021）的单产，2017—2021各行的前向偏移需要2017年以前的历史。",
              "models": {}}
    for name, rows in predictions.items():
        entry = {"coverage_years": coverage[name] and "2022" or "2017-2022"}
        entry["year_2022"] = metrics(subset(rows, {2022}), county_base)
        # 第5.2.2节"逐折看"的数是每折的 2022 年子集。
        entry["per_fold_2022"] = {
            str(fold): metrics([r for r in rows
                                if int(r["year"]) == 2022 and r["fips"] in set(manifest["folds"][fold]["counties"])],
                               county_base)
            for fold in range(n_folds)
        }
        report["models"][name] = entry
        if name == STATE_YEAR:
            # 该基线在同一 (州, 年) 内取常数，去均值后没有任何县间差异；按县历史构造的
            # 异常相关在这里退化为"县自身历史"的属性，不是模型的排序能力，因此不报。
            entry["year_2022"]["anom_corr"] = None
            for values in entry["per_fold_2022"].values():
                values["anom_corr"] = None
        print(f"{name:14s} 2022 RMSE {entry['year_2022']['rmse']:7.3f}")

    reference = report["models"][COUNTY_FE]
    report["differences"] = {
        "note": "模型减仅县固定效应；负值表示误差更低。点估计，不给区间。",
        "year_2022": {name: entry["year_2022"]["rmse"] - reference["year_2022"]["rmse"]
                      for name, entry in report["models"].items()
                      if name != COUNTY_FE},
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("county-grouped spatial CV report")
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--dataset-jsonl", default=str(DEFAULT_DATASET))
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "paper_figures" / "spatial_cv.json"))
    main(parser.parse_args())
