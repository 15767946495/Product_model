"""全样本信息检验（论文第5.8节所用设计）：每次置换都重新按年份交叉验证选择 lambda。

全部县年一起估计岭回归面板模型；统计量为全样本 RMSE；零分布为年份内跨县置换植被指数，
并对置换后的数据重新做年份分组交叉验证，使零分布复现完整估计流程。
输出 paper_figures/rs_info_test.json 所引用的结果文件。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/data/hqx/myself/Product_model")
from TFT_model.rs_signal_test import load, flatten, within, _standardize, _ridge_by_year_cv  # noqa: E402

NPZ = "/data/raid0/hqx/train_dataset/rs_signal_features_31states.npz"
OUT = Path("/data/raid0/hqx/train_dataset/rs_info_test_cv_31states.json")
N_PERM = 500

d = load(NPZ)
z, fips, years = d["z"], d["fips"], d["year"]
always = np.ones(d["month_valid"].shape, dtype=bool)
allrows = np.ones(len(z), dtype=bool)

Xw, _ = flatten(d["weather"], always)
Xv, _ = flatten(d["veg"], d["month_valid"])
Xw = _standardize(within(Xw, fips, allrows), allrows)
Xv = _standardize(within(Xv, fips, allrows), allrows)

zmean = {f: z[fips == f].mean() for f in np.unique(fips)}
base = np.array([zmean[f] for f in fips])
y = z - base


def rmse(pred_z):
    return float(np.sqrt(np.mean((np.exp(pred_z) - np.exp(z)) ** 2)))


rng = np.random.default_rng(0)
order = np.argsort(years)
bounds = np.split(order, np.flatnonzero(np.diff(years[order])) + 1)

result = {"n_county_years": int(len(z)), "n_counties": int(len(np.unique(fips))),
          "n_permutations": N_PERM, "design": "lambda re-selected by year-grouped CV in every permutation",
          "fits": {}, "tests": {}}

blocks = {"county_fe_only": np.zeros((len(z), 0)), "weather": Xw, "veg_indices": Xv,
          "weather+veg": np.concatenate([Xw, Xv], axis=1)}
for name, X in blocks.items():
    if X.shape[1] == 0:
        result["fits"][name] = {"rmse": rmse(base), "alpha": None}
        continue
    model, alpha = _ridge_by_year_cv(X, y, years)
    result["fits"][name] = {"rmse": rmse(base + model.predict(X)), "alpha": alpha}
    print(name, result["fits"][name], flush=True)


def perm_test(key, fixed, permuted):
    observed = result["fits"][key]["rmse"]
    null, alphas = [], []
    for _ in range(N_PERM):
        Xp = permuted.copy()
        for idx in bounds:
            Xp[idx] = permuted[rng.permutation(idx)]
        X = np.concatenate([fixed, Xp], axis=1) if fixed.shape[1] else Xp
        model, alpha = _ridge_by_year_cv(X, y, years)
        null.append(rmse(base + model.predict(X)))
        alphas.append(alpha)
    null = np.array(null)
    p = float((np.sum(null <= observed) + 1) / (N_PERM + 1))
    return {"observed": observed, "null_mean": float(null.mean()), "null_sd": float(null.std(ddof=1)),
            "null_q05": float(np.quantile(null, 0.05)), "p": p,
            "alpha_median": float(np.median(alphas)), "alpha_min": float(np.min(alphas)),
            "alpha_max": float(np.max(alphas))}


result["tests"]["weather+veg"] = perm_test("weather+veg", Xw, Xv)
print("weather+veg", result["tests"]["weather+veg"], flush=True)
result["tests"]["veg_indices"] = perm_test("veg_indices", np.zeros((len(z), 0)), Xv)
print("veg_indices", result["tests"]["veg_indices"], flush=True)

OUT.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("saved", OUT, flush=True)
