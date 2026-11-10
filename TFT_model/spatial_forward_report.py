"""前向滚动起点的按县五折空间交叉验证汇总：PA-TFT、县固定效应、线性面板（岭）。

对每个测试年 y ∈ {2018,...,2022}：
  - 训练县只用 y 之前的年份；留出县只在 y 年上测试；
  - 县均值偏移、遥感/特征县内中心、全局标准化都只用 y 之前的县年；
  - 留出县的偏移量来自它自己 y 之前的单产；训练行的偏移量是同年以前各年的留一均值。

2018 年只有 2017 一年前史：训练行的县内去均值使特征为 0，县均值偏移也退化为
跨县均值；因此 2018 行只作参考，不能和 2019--2022 同等解读。

用法：
  python -I TFT_model/spatial_forward_report.py --output paper_figures/spatial_forward.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TFT_model.rs_signal_test import (  # noqa: E402
    _ridge_by_year_cv,
    flatten,
    load,
    score,
)

DEFAULT_FEATURES = Path("/data/raid0/hqx/train_dataset/rs_signal_features_31states.npz")
DEFAULT_FOLDS = Path("/data/raid0/hqx/train_dataset/spatial_folds/folds.json")
DEFAULT_RUNS = Path("/data/raid0/hqx/TFT_out")
DEFAULT_ALPHA = 3162.0  # 单训练年（2018 行）无法做年份 CV 时的固定惩罚


def county_past_means(fips: np.ndarray, years: np.ndarray, z: np.ndarray) -> dict[str, list[float]]:
    """fips -> 按年份递增排列的 (year, log yield)。"""
    by_county: dict[str, dict[int, float]] = {}
    for f, year, value in zip(fips, years, z):
        by_county.setdefault(str(f), {})[int(year)] = float(value)
    return {f: sorted(per_year.items()) for f, per_year in by_county.items()}


def offsets(fips, years, z, test_year: int, *, leave_one_out: bool = True):
    """返回 (训练行偏移, 测试行前向偏移)。

    训练行：默认留一（该县 test_year 之前、且不是本行年份的各年均值）；若没有其它
    年份（2018 行的 2017 训练样本），用该年跨县均值兜底。``leave_one_out=False``
    时训练行也用 test_year 之前全部年份的均值，与测试行的偏移同尺度。
    测试行：该县 test_year 之前各年均值。
    """
    history = county_past_means(fips, years, z)
    fallback = float(np.mean([value for year, value in zip(years, z) if year == min(years)]))
    train_offset = np.full(len(z), np.nan)
    test_offset = np.full(len(z), np.nan)
    for f, per_year in history.items():
        past = [(year, value) for year, value in per_year if year < test_year]
        if not past:
            continue
        mean_past = float(np.mean([value for _, value in past]))
        for year, value in per_year:
            rows = (fips == f) & (years == year)
            if year < test_year:
                if leave_one_out:
                    others = [v for y, v in past if y != year]
                    train_offset[rows] = float(np.mean(others)) if others else fallback
                else:
                    train_offset[rows] = mean_past
            elif year == test_year:
                test_offset[rows] = mean_past
    return train_offset, test_offset


def metrics(z, pred, fips, years, base) -> dict:
    truth, hat = np.exp(z), np.exp(pred)
    error = hat - truth
    out = {
        "n": int(len(z)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
        "r2": float(1.0 - (error ** 2).sum() / ((truth - truth.mean()) ** 2).sum()),
        "r": float(np.corrcoef(hat, truth)[0, 1]) if len(z) > 2 else None,
        "pred_sd": float(hat.std()),
    }
    if base is not None:
        out.update(score(z, pred, fips, years, base))
        # score 也返回 n/rmse/bias，保留上面的 MAE/R2/r
        out["n"], out["rmse"], out["bias"] = int(len(z)), out["rmse"], float(np.mean(error))
    return out


def center_by_county(X, valid, fips, years, test_year):
    """每个县的 y 之前各年特征均值；留出县用它自己的历史（无标签）。"""
    centered = np.zeros_like(X, dtype=np.float64)
    for f in np.unique(fips):
        rows = fips == f
        past = rows & (years < test_year)
        weight = valid[past]
        mean = (X[past] * weight).sum(0) / np.maximum(weight.sum(0), 1)
        centered[rows] = np.where(valid[rows], X[rows] - mean, 0.0)
    return centered


def ridge_fold(data, fold_of, fold: int, test_year: int, *, leave_one_out: bool = False):
    """在训练县 y 之前年份上拟合县 FE + 天气 + 植被的岭回归，预测留出县 y 年。

    默认 ``leave_one_out=False``：训练行与测试行都用 y 之前全部年份的县均值，
    目标尺度一致；留一偏移会把目标放大 T/(T-1)，在 2019（T=2）尤其严重。
    """
    z, fips, years = data["z"], data["fips"], data["year"]
    always = np.ones(data["month_valid"].shape, dtype=bool)
    Xw = flatten(data["weather"], always)[0]
    Xv, Vv = flatten(data["veg"], data["month_valid"])
    X = np.concatenate([Xw, Xv], axis=1)
    valid = np.concatenate([np.ones_like(Xw, dtype=bool), Vv], axis=1)

    train = (fold_of != fold) & (years < test_year)
    test = (fold_of == fold) & (years == test_year)
    train_offset, test_offset = offsets(fips, years, z, test_year, leave_one_out=leave_one_out)

    # 训练行的偏移量在 y 之前有历史，测试行是留出县自己的前向均值
    train = train & np.isfinite(train_offset)
    test = test & np.isfinite(test_offset)
    centered = center_by_county(X, valid, fips, years, test_year)
    mu = centered[train].mean(0)
    sd = centered[train].std(0)
    keep = sd > 1e-10
    if not keep.any():
        # 只有一个训练年（2018 行）时县内去均值把全部特征变成 0，岭回归退化为县 FE。
        pred = test_offset[test]
        return test, pred, pred, None
    Xs = (centered[:, keep] - mu[keep]) / sd[keep]

    target = z - train_offset
    train_years = years[train]
    if len(np.unique(train_years)) >= 2:
        model, alpha = _ridge_by_year_cv(Xs[train], target[train], train_years)
    else:
        model, alpha = Ridge(alpha=DEFAULT_ALPHA).fit(Xs[train], target[train]), DEFAULT_ALPHA
    pred = test_offset[test] + model.predict(Xs[test])
    return test, pred, test_offset[test], alpha


def load_patft(runs: Path, prefix: str, fold: int, test_year: int):
    path = runs / "tft" / f"{prefix}-fold{fold}-y{test_year}-s0" / "county_predictions.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing PA-TFT predictions: {path}")
    rows = [json.loads(line) for line in path.open("r", encoding="utf-8")]
    return rows


def evaluate_patft(data, fold_of, manifest, runs, prefix, test_year):
    z, fips, years = data["z"], data["fips"], data["year"]
    _, test_offset = offsets(fips, years, z, test_year)
    index = {(str(f), int(y)): i for i, (f, y) in enumerate(zip(fips, years))}
    rows_idx, preds = [], []
    for fold in range(int(manifest["k"])):
        held = set(manifest["folds"][fold]["counties"])
        rows = load_patft(runs, prefix, fold, test_year)
        for row in rows:
            key = (str(row["fips"]), int(row["year"]))
            if key not in index:
                raise KeyError(f"prediction {key} not in feature panel")
            i = index[key]
            if str(row["fips"]) not in held or int(row["year"]) != test_year:
                raise ValueError(f"prediction {key} outside fold {fold} test set")
            rows_idx.append(i)
            preds.append(float(row["y_pred"]))
    idx = np.array(rows_idx)
    pred = np.log(np.maximum(np.array(preds, dtype=np.float64), 1e-6))
    if len(idx) != int((years == test_year).sum()):
        raise ValueError(f"PA-TFT {test_year}: {len(idx)} predictions, expected {(years == test_year).sum()}")
    return idx, pred, test_offset[idx]


def main(args) -> None:
    data = load(args.features)
    z, fips, years = data["z"], data["fips"], data["year"]
    manifest = json.loads(Path(args.folds).read_text(encoding="utf-8"))
    assign = manifest["assignments"]
    fold_of = np.array([assign[str(f)] for f in fips])
    runs = Path(args.runs_root).resolve()
    test_years = [int(y) for y in args.years.split(",")]

    report = {
        "protocol": "forward rolling origin, county-grouped 5-fold spatial CV",
        "years": test_years,
        "note": "each test year y uses only years < y for training, county offsets, "
                "RS centres and feature standardisation; 2018 has one prior year only",
        "models": {},
    }

    # 县固定效应：留出县在 y 年之前各年对数单产的均值（与 PA-TFT 的 offset 同定义）
    county_fe = {}
    for year in test_years:
        _, test_offset = offsets(fips, years, z, year)
        idx = np.where((years == year) & np.isfinite(test_offset))[0]
        county_fe[year] = (idx, test_offset[idx])
    report["models"]["county_fe"] = {
        "per_year": {
            str(year): metrics(z[idx], base, fips[idx], years[idx], base)
            for year, (idx, base) in county_fe.items()
        },
    }
    fe_idx = np.concatenate([idx for idx, _ in county_fe.values()])
    fe_base = np.concatenate([base for _, base in county_fe.values()])
    report["models"]["county_fe"]["pooled"] = metrics(
        z[fe_idx], fe_base, fips[fe_idx], years[fe_idx], fe_base)
    for cutoff in (2019, 2020):
        keep = years[fe_idx] >= cutoff
        report["models"]["county_fe"][f"pooled_{cutoff}"] = metrics(
            z[fe_idx[keep]], fe_base[keep], fips[fe_idx[keep]], years[fe_idx[keep]], fe_base[keep])

    # 线性面板（岭）：每折、每年单独拟合
    for name, prefix in (("ridge_panel", None), ("patft", args.patft_prefix)):
        if name == "patft" and args.skip_patft:
            continue
        per_year = {}
        pooled_idx, pooled_pred, pooled_base = [], [], []
        for year in test_years:
            if name == "ridge_panel":
                idxs, preds, bases, alphas = [], [], [], []
                for fold in range(int(manifest["k"])):
                    mask, pred, base, alpha = ridge_fold(data, fold_of, fold, year)
                    idxs.append(np.where(mask)[0])
                    preds.append(pred)
                    bases.append(base)
                    alphas.append(alpha)
                idx = np.concatenate(idxs)
                pred = np.concatenate(preds)
                base = np.concatenate(bases)
                per_year[str(year)] = metrics(z[idx], pred, fips[idx], years[idx], base)
                per_year[str(year)]["alpha"] = [None if a is None else float(a) for a in alphas]
            else:
                try:
                    idx, pred, base = evaluate_patft(data, fold_of, manifest, runs, prefix, year)
                except FileNotFoundError as error:
                    print(f"skip {name}: {error}")
                    per_year = None
                    break
                per_year[str(year)] = metrics(z[idx], pred, fips[idx], years[idx], base)
            pooled_idx.append(idx)
            pooled_pred.append(pred)
            pooled_base.append(base)
        if per_year is None:
            continue
        idx = np.concatenate(pooled_idx)
        pred = np.concatenate(pooled_pred)
        base = np.concatenate(pooled_base)
        report["models"][name] = {
            "per_year": per_year,
            "pooled": metrics(z[idx], pred, fips[idx], years[idx], base),
        }
        for cutoff in (2019, 2020):
            keep = years[idx] >= cutoff
            report["models"][name][f"pooled_{cutoff}"] = metrics(
                z[idx[keep]], pred[keep], fips[idx[keep]], years[idx[keep]], base[keep])

    # 逐折的 2022（与旧 spatial_report 对齐，便于看折间波动）
    year = 2022
    if year in test_years and "patft" in report["models"]:
        report["models"]["patft"]["per_fold_2022"] = {}
        idx_2022 = {}
        for fold in range(int(manifest["k"])):
            rows = load_patft(runs, args.patft_prefix, fold, year)
            held = set(manifest["folds"][fold]["counties"])
            zz = np.array([np.log(max(r["y_true"], 1e-6)) for r in rows])
            pp = np.array([np.log(max(r["y_pred"], 1e-6)) for r in rows])
            ff = np.array([r["fips"] for r in rows])
            _, test_offset = offsets(fips, years, z, year)
            index = {(str(f), int(y)): i for i, (f, y) in enumerate(zip(fips, years))}
            base = np.array([test_offset[index[(str(r["fips"]), year)]] for r in rows])
            assert set(ff) == held
            report["models"]["patft"]["per_fold_2022"][str(fold)] = metrics(
                zz, pp, ff, np.full(len(ff), year), base)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for name, entry in report["models"].items():
        print(f"{name:12s} " + "  ".join(
            f"{year}: {entry['per_year'][str(year)]['rmse']:.3f}" for year in test_years))
        print(f"{'':12s} pooled: {entry['pooled']['rmse']:.3f}")
    print(f"wrote {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("forward spatial CV report")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    parser.add_argument("--folds", default=str(DEFAULT_FOLDS))
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS))
    parser.add_argument("--patft-prefix", default="spatialfwd-improved")
    parser.add_argument("--skip-patft", action="store_true", default=False,
                        help="only compute county-FE and ridge baselines")
    parser.add_argument("--years", default="2018,2019,2020,2021,2022")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "paper_figures" / "spatial_forward.json"))
    main(parser.parse_args())
