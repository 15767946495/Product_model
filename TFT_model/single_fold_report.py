"""2022 单折口径：各配置的预测、RMSE，以及与完整 PA-TFT 的 RMSE 差的按县/按州聚类自助区间（5000 次）。

论文只保留 2022 折、每个配置只保留种子 0 的单次运行（LOYO_SEEDS=0），不再作集成。
输出 paper_figures/single_fold_2022.json，供 docs/make_tables_tft.py 与正文引用。
用法：LOYO_SEEDS=0 python TFT_model/single_fold_report.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path("/data/raid0/hqx/TFT_out/tft")
OUT = ROOT / "paper_figures"
TAG = "loyo31-"
YEAR = 2022
# 种子列表可用 LOYO_SEEDS 覆盖（单种子复核时 LOYO_SEEDS=0）
SEEDS = tuple(int(s) for s in __import__("os").environ.get("LOYO_SEEDS", "42 1 2").split())
CONFIGS = {"improved": "完整PA-TFT", "noffset": "去掉县均值偏移",
           "nodemean": "去掉植被指数县内去均值", "nors": "去掉遥感输入"}


def load_labels():
    labels = {}
    for line in open(ROOT / "train_dataset" / "dataset.jsonl"):
        head = line[:600]
        fips = re.search(r'"fips":"(\d+)"', head).group(1)
        year = int(re.search(r'"year":(\d+)', head).group(1))
        labels[(fips, year)] = float(re.search(r'"yield_bu_ac":([\d.]+)', head).group(1))
    return labels


def load_pred(run_id):
    path = RUNS / run_id / "county_predictions.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in open(path)]
    return ({r["fips"]: r["y_pred"] for r in rows}, [r["fips"] for r in rows],
            np.array([r["y_true"] for r in rows]))


def demean(x, groups):
    x = x.copy()
    for g in set(groups):
        x[groups == g] -= x[groups == g].mean()
    return x


def anomaly_corr(pred, truth, base_log, fips, states):
    ah, a = np.log(pred) - base_log, np.log(truth) - base_log
    if ah.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(demean(ah, states), demean(a, states))[0, 1])


def cluster_bootstrap(e_main, e_ref, clusters, reps=5000, seed=0):
    """RMSE_主 − RMSE_对照 的聚类自助区间。"""
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    idx = {c: np.flatnonzero(clusters == c) for c in uniq}
    diffs = np.empty(reps)
    for i in range(reps):
        pick = rng.choice(uniq, len(uniq), replace=True)
        rows = np.concatenate([idx[c] for c in pick])
        diffs[i] = (np.sqrt((e_main[rows] ** 2).mean()) - np.sqrt((e_ref[rows] ** 2).mean()))
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))], float((diffs < 0).mean())


def main():
    labels = load_labels()
    improved = [load_pred(f"{TAG}improved-y{YEAR}-s{s}") for s in SEEDS]
    improved = [p for p in improved if p]
    assert improved, "缺少 PA-TFT 的 2022 折预测"
    fips, truth = improved[0][1], improved[0][2]
    ens_main = np.mean([np.array([p[0][f] for f in fips]) for p in improved], axis=0)
    base_log = np.array([np.mean([np.log(labels[(c, u)]) for u in range(2017, 2022)]) for c in fips])
    states = np.array([c[:2] for c in fips])

    out = {"year": YEAR, "n": int(len(fips)),
           "main": {"single_rmse": [float(np.sqrt(((np.array([p[0][f] for f in fips]) - truth) ** 2).mean())) for p in improved],
                    "rmse": float(np.sqrt(((ens_main - truth) ** 2).mean()))},
           "configs": {}}
    e_main = ens_main - truth
    for key, label in CONFIGS.items():
        preds = [load_pred(f"{TAG}{key}-y{YEAR}-s{s}") for s in SEEDS]
        preds = [p for p in preds if p]
        if not preds:
            continue
        ens = np.mean([np.array([p[0][f] for f in fips]) for p in preds], axis=0)
        e = ens - truth
        rmse = float(np.sqrt((e ** 2).mean()))
        row = {"label": label, "n_seeds": len(preds), "rmse": rmse,
               "single_rmse": [float(np.sqrt(((np.array([p[0][f] for f in fips]) - truth) ** 2).mean())) for p in preds],
               "bias": float(e.mean()),
               "anom_corr": anomaly_corr(ens, truth, base_log, fips, states),
               "pred_sd": float(ens.std()),
               "delta": rmse - out["main"]["rmse"]}
        if key != "improved":
            row["county_ci"], row["county_share"] = cluster_bootstrap(e_main, e, np.arange(len(fips)))
            row["state_ci"], row["state_share"] = cluster_bootstrap(e_main, e, states)
        out["configs"][key] = row
        print(f"{label:28s} rmse {rmse:6.2f}  delta {row['delta']:+.2f}  anom {row['anom_corr']:.3f}")

    fe = np.exp(base_log)
    e_fe = fe - truth
    row = {"label": "仅县固定效应", "rmse": float(np.sqrt((e_fe ** 2).mean())), "bias": float(e_fe.mean()),
           "anom_corr": 0.0, "pred_sd": float(fe.std()), "delta": None}
    row["rmse_delta"] = row["rmse"] - out["main"]["rmse"]
    row["county_ci"], row["county_share"] = cluster_bootstrap(e_main, e_fe, np.arange(len(fips)))
    row["state_ci"], row["state_share"] = cluster_bootstrap(e_main, e_fe, states)
    out["configs"]["county_fe"] = row
    print(f"{'仅县固定效应':28s} rmse {row['rmse']:6.2f}")
    (OUT / "single_fold_2022.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
