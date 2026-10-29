"""遥感增量信号检验：冻结特征能否解释县固定效应之外的单产变化。

目标 z = log(yield)。所有特征与目标先做县内去均值（county FE / within 变换），
县均值仅由当期训练折计算，因此模型只能利用"同一县不同年份之间"的差异。
对每个留出年份 y（2017—2022 依次留出）：
  1. 训练折 = 其余年份中的已见县年；以训练折县均值去除目标与特征的县效应；
  2. 留出年份的预测 = 训练折县均值 + f(留出年份特征 - 训练折县特征均值)；
  3. 年份效应：同一年所有县共享的冲击无法由县内横截面识别，故另报"去除年份均值后"的
     县内州内异常相关（within-year 排序能力），与 RMSE 一起评价。
比较特征集：仅县均值（零模型）、天气月均值、植被指数、冻结 PVT（PCA 降维）、天气+植被、
天气+PVT；并用"在留出年份内跨县打乱遥感特征"的置换检验给出遥感增量的零分布。
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

from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

ALPHAS = np.logspace(-2, 5, 29)


def within(values: np.ndarray, groups: np.ndarray, train: np.ndarray, valid: np.ndarray | None = None):
    """县内去均值：values (N,K) 减去训练折该县均值；valid (N,K) 为 False 的位置
    不参与均值且去均值后置 0（缺图月份视为"与本县平常一样"，不注入伪信号）。"""
    valid = np.ones(values.shape, dtype=bool) if valid is None else valid
    centered = np.zeros_like(values, dtype=np.float64)
    for group in np.unique(groups):
        rows = groups == group
        fit = rows & train
        weight = valid[fit]
        mean = (values[fit] * weight).sum(0) / np.maximum(weight.sum(0), 1)
        centered[rows] = np.where(valid[rows], values[rows] - mean, 0.0)
    return centered


def _standardize(X, train):
    mu = X[train].mean(0)
    sd = X[train].std(0)
    keep = sd > 1e-10
    return (X[:, keep] - mu[keep]) / sd[keep]


ESTIMATOR = "ridge"   # "ridge"：按年份交叉验证选择惩罚强度；"ols"：普通最小二乘


class _LeastSquares:
    """带截距的普通最小二乘，用于检验估计器选择是否改变置换检验的结论。"""

    def __init__(self, X, y):
        self.coef, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)

    def predict(self, X):
        return self.coef[0] + X @ self.coef[1:]


def fit_model(X, y, years):
    if ESTIMATOR == "ols":
        return _LeastSquares(X, y), 0.0
    return _ridge_by_year_cv(X, y, years)


def _ridge_by_year_cv(X, y, years):
    """在训练折内按年份留一选择 alpha（年份内县间相关，逐行 LOO 会选过小的 alpha）。"""
    from sklearn.linear_model import Ridge

    scores = np.zeros(len(ALPHAS))
    for held in np.unique(years):
        fit, val = years != held, years == held
        for k, alpha in enumerate(ALPHAS):
            pred = Ridge(alpha=alpha).fit(X[fit], y[fit]).predict(X[val])
            scores[k] += ((pred - y[val]) ** 2).sum()
    alpha = float(ALPHAS[int(np.argmin(scores))])
    return Ridge(alpha=alpha).fit(X, y), alpha


def _prepare_block(X, valid, fips, train, n_pca, seed):
    """单个特征块：县内去均值 → 训练折标准化 →（可选）训练折 PCA 后再标准化。"""
    Xw = _standardize(within(X, fips, train, valid), train)
    if n_pca is not None and n_pca < Xw.shape[1]:
        Xw = PCA(n_components=n_pca, random_state=seed).fit(Xw[train]).transform(Xw)
        Xw = Xw / Xw[train].std(0).clip(1e-10)
    return Xw


def fold_predict(blocks, z, fips, years, test_year, *, permutation=None, seed=0, cache=None,
                 forward_only=False):
    """留出 test_year：县 FE（训练折县均值）+ 县内特征异常的岭回归。

    blocks: [(X (N,K), valid (N,K), is_rs, n_pca)]，空列表为只用县 FE 的零模型。
    permutation: 长度 N 的行索引（年份内跨县置换），只作用于遥感块的县内异常。
    forward_only: True 时训练折只含早于 test_year 的年份（滚动起点，无未来信息）；
    False 时为逐年留一（训练折含其他所有年份，仅用于信号检验）。
    返回 (test 掩码, 对数尺度预测, 选中的 alpha)。"""
    train = (years < test_year) if forward_only else (years != test_year)
    test = (years == test_year) & np.isin(fips, np.unique(fips[train]))
    z_mean = {f: z[train & (fips == f)].mean() for f in np.unique(fips[train])}
    base = np.array([z_mean.get(f, np.nan) for f in fips])
    if not blocks:
        return test, base[test], None
    parts = []
    for X, valid, is_rs, n_pca in blocks:
        key = (id(X), int(test_year), n_pca)
        if cache is not None and key in cache:
            Xw = cache[key]
        else:
            Xw = _prepare_block(X, valid, fips, train, n_pca, seed)
            if cache is not None:
                cache[key] = Xw
        parts.append(Xw[permutation] if is_rs and permutation is not None else Xw)
    Xw = np.concatenate(parts, axis=1)
    model, alpha = fit_model(Xw[train], (z - base)[train], years[train])
    return test, base[test] + model.predict(Xw[test]), alpha


def within_year_permutation(years, rng):
    """每个年份内部随机置换行（跨县），保留年份层面的遥感均值。"""
    perm = np.arange(len(years))
    for year in np.unique(years):
        idx = np.where(years == year)[0]
        perm[idx] = rng.permutation(idx)
    return perm


def _state_demean(values, states):
    out = values.astype(np.float64).copy()
    for state in np.unique(states):
        out[states == state] -= out[states == state].mean()
    return out


def score(z, pred, fips, years, base=None):
    """原尺度 RMSE / 偏差，以及州×年内的异常相关：corr(预测-县均值, 真值-县均值) 在
    去除州×年均值后计算，只衡量同一年同一州内"哪些县比自己平常好"的排序能力。"""
    groups = np.array([f"{f[:2]}_{y}" for f, y in zip(fips, years)])
    y, y_hat = np.exp(z), np.exp(pred)
    out = {
        "n": int(len(z)),
        "rmse": float(np.sqrt(np.mean((y_hat - y) ** 2))),
        "bias": float(np.mean(y_hat - y)),
    }
    if base is not None:
        anomaly_hat = _state_demean(pred - base, groups)
        anomaly = _state_demean(z - base, groups)
        out["within_state_year_anom_corr"] = (
            float(np.corrcoef(anomaly_hat, anomaly)[0, 1]) if anomaly_hat.std() > 1e-12 else 0.0
        )
    return out


def flatten(features: np.ndarray, valid: np.ndarray):
    """(N,6,K) + (N,6) → (N,6K) 与对应有效掩码。"""
    n, months, k = features.shape
    return features.reshape(n, months * k), np.repeat(valid[:, :, None], k, axis=2).reshape(n, months * k)


def evaluate(blocks, data, test_years, *, permutation=None, seed=0, cache=None, forward_only=False):
    """逐年份评价（默认留一，forward_only=True 为滚动起点）；返回逐年与合并指标。
    异常相关以同一折的县 FE 预测为基准。"""
    z, fips, years = data["z"], data["fips"], data["year"]
    rows, idx_all, pred_all, base_all = [], [], [], []
    for year in test_years:
        test, pred, alpha = fold_predict(
            blocks, z, fips, years, year, permutation=permutation, seed=seed, cache=cache,
            forward_only=forward_only,
        )
        _, base, _ = fold_predict([], z, fips, years, year, forward_only=forward_only)
        idx = np.where(test)[0]
        rows.append({"year": int(year), "alpha": alpha,
                     **score(z[idx], pred, fips[idx], years[idx], base)})
        idx_all.append(idx)
        pred_all.append(pred)
        base_all.append(base)
    idx = np.concatenate(idx_all)
    pooled = score(z[idx], np.concatenate(pred_all), fips[idx], years[idx], np.concatenate(base_all))
    return {"per_year": rows, "pooled": pooled}


def load(path):
    raw = np.load(path, allow_pickle=False)
    order = np.lexsort((raw["year"], raw["fips"]))
    data = {key: raw[key][order] for key in ("sample_id", "fips", "year", "label", "pvt", "veg",
                                              "month_valid", "weather")}
    data["z"] = np.log(data["label"].astype(np.float64))
    data["pvt"] = data["pvt"].astype(np.float64)
    return data


def feature_sets(data, n_pca):
    always = np.ones(data["month_valid"].shape, dtype=bool)
    weather = (*flatten(data["weather"], always), False, None)
    veg = (*flatten(data["veg"], data["month_valid"]), True, None)
    # PVT 6×512 维：训练折内 PCA 到 n_pca 维，避免岭回归在 1920 行上过拟合
    pvt = (*flatten(data["pvt"], data["month_valid"]), True, n_pca)
    return {
        "county_fe_only": [],
        "weather": [weather],
        "veg_indices": [veg],
        "pvt_frozen": [pvt],
        "weather+veg": [weather, veg],
        "weather+pvt": [weather, pvt],
    }


def run(data, test_years, *, n_pca, permutations, seed=0):
    sets = feature_sets(data, n_pca)
    cache = {}
    results = {}
    for name, blocks in sets.items():
        results[name] = evaluate(blocks, data, test_years, seed=seed, cache=cache)
        print(name, json.dumps(results[name]["pooled"]), flush=True)
    rng = np.random.default_rng(seed)
    for name in ("veg_indices", "pvt_frozen", "weather+veg", "weather+pvt") if permutations > 0 else ():
        nulls = []
        for _ in range(permutations):
            perm = within_year_permutation(data["year"], rng)
            nulls.append(evaluate(sets[name], data, test_years, permutation=perm, seed=seed, cache=cache)["pooled"])
        observed = results[name]["pooled"]
        rmse_null = np.array([row["rmse"] for row in nulls])
        corr_null = np.array([row["within_state_year_anom_corr"] for row in nulls])
        results[name]["permutation"] = {
            "n": permutations,
            "rmse_null_mean": float(rmse_null.mean()),
            "rmse_null_min": float(rmse_null.min()),
            # 单侧 p 值：置换后至少一样好的比例（+1 平滑）
            "p_rmse": float((1 + (rmse_null <= observed["rmse"]).sum()) / (permutations + 1)),
            "corr_null_mean": float(corr_null.mean()),
            "corr_null_max": float(corr_null.max()),
            "p_corr": float((1 + (corr_null >= observed["within_state_year_anom_corr"]).sum()) / (permutations + 1)),
        }
        print(name, "permutation", json.dumps(results[name]["permutation"]), flush=True)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Remote-sensing incremental signal test (county FE + ridge)")
    parser.add_argument("--features", required=True, help="npz from rs_signal_extract.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--test-years", default="2017,2018,2019,2020,2021,2022")
    parser.add_argument("--n-pca", type=int, default=32)
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    data = load(args.features)
    years = [int(y) for y in args.test_years.split(",")]
    results = run(data, years, n_pca=args.n_pca, permutations=args.permutations, seed=args.seed)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
