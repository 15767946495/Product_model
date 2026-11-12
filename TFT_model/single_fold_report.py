"""2022 单折口径：各配置的预测、RMSE，以及与主模型 PA-TFT 的 RMSE 差的按县/按州聚类自助区间（5000 次）。

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
TAG = __import__("os").environ.get("LOYO_TAG", "paper-")
YEAR = 2022
TRAIN_LAST_YEAR = 2021
# 种子列表可用 LOYO_SEEDS 覆盖（单种子复核时 LOYO_SEEDS=0）
SEEDS = tuple(int(s) for s in __import__("os").environ.get("LOYO_SEEDS", "0").split())
MAIN = "main"
CONFIGS = {"main": "PA-TFT（主模型）"}
# 移除式消融：把整条支路从结构中删掉后重新训练，运行名见 run_removal_ablation.sh
REMOVAL_CONFIGS = {
    "norsbranch": "删除遥感支路",
    "nosoilbranch": "删除土壤静态支路",
    "noconstructed": "去掉4个构造气象变量",
}


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


def param_count(run_id):
    """参数量：删除支路后应当少掉该支路自己的参数。"""
    import torch

    state = torch.load(RUNS / run_id / "checkpoint-final.pth", map_location="cpu",
                       weights_only=False)["model"]
    return int(sum(v.numel() for v in state.values()))


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
    mains = [load_pred(f"{TAG}{MAIN}-y{YEAR}-s{s}") for s in SEEDS]
    mains = [p for p in mains if p]
    assert mains, "缺少 PA-TFT 主模型的 2022 折预测"
    fips, truth = mains[0][1], mains[0][2]
    ens_main = np.mean([np.array([p[0][f] for f in fips]) for p in mains], axis=0)
    base_log = np.array([np.mean([np.log(labels[(c, u)]) for u in range(2017, TRAIN_LAST_YEAR + 1)]) for c in fips])
    states = np.array([c[:2] for c in fips])

    out = {"year": YEAR, "n": int(len(fips)),
           "main": {"single_rmse": [float(np.sqrt(((np.array([p[0][f] for f in fips]) - truth) ** 2).mean())) for p in mains],
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
        if key != MAIN:
            row["county_ci"], row["county_share"] = cluster_bootstrap(e_main, e, np.arange(len(fips)))
            row["state_ci"], row["state_share"] = cluster_bootstrap(e_main, e, states)
        out["configs"][key] = row
        print(f"{label:28s} rmse {rmse:6.2f}  delta {row['delta']:+.2f}  anom {row['anom_corr']:.3f}")

    # 移除式消融：整条支路从结构中删除后重新训练（2022 折、种子 0）
    out["removal_ablation"] = {}
    for key, label in REMOVAL_CONFIGS.items():
        run_id = f"ablate-{key}-y{YEAR}-s0"
        loaded = load_pred(run_id)
        if loaded is None:
            print(f"跳过 {label}：缺少 {run_id}")
            continue
        pred_map = loaded[0]
        ens = np.array([pred_map[f] for f in fips])
        e = ens - truth
        rmse = float(np.sqrt((e ** 2).mean()))
        row = {"label": label, "run_id": run_id, "rmse": rmse,
               "delta": rmse - out["main"]["rmse"],
               "bias": float(e.mean()),
               "anom_corr": anomaly_corr(ens, truth, base_log, fips, states),
               "pred_sd": float(ens.std()),
               "params": param_count(run_id)}
        row["state_ci"], row["state_share"] = cluster_bootstrap(e_main, e, states)
        out["removal_ablation"][key] = row
        print(f"{label:28s} rmse {rmse:6.2f}  delta {row['delta']:+.2f}  anom {row['anom_corr']:.3f}  "
              f"params {row['params']:,}  CI [{row['state_ci'][0]:+.2f},{row['state_ci'][1]:+.2f}]")

    out["main"]["params"] = param_count(f"{TAG}{MAIN}-y{YEAR}-s0")
    (OUT / "single_fold_2022.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
