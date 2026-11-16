"""2022 单折口径：各配置的预测、RMSE，以及与主模型 PAF-Net 的逐州配对比较。

逐州比较直接以州为单位计算两个模型的 RMSE 与差，报告改善/变差的州数与州内差的中位数，
不复采样州（单折只有 29 个州，按州自助重采样的区间不稳定）。
论文只保留 2022 折、每个配置只保留种子 0 的单次运行（LOYO_SEEDS=0），不再作集成。
输出 paper_figures/single_fold_2022.json，供 fig5_3 与正文引用。
用法：LOYO_SEEDS=0 python TFT_model/single_fold_report.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFT_model.paper_report import STATE_NAMES  # noqa: E402

RUNS = Path("/data/raid0/hqx/TFT_out/tft")
OUT = ROOT / "paper_figures"
TAG = __import__("os").environ.get("LOYO_TAG", "paper-")
YEAR = 2022
TRAIN_LAST_YEAR = 2021
# 种子列表可用 LOYO_SEEDS 覆盖（单种子复核时 LOYO_SEEDS=0）
SEEDS = tuple(int(s) for s in __import__("os").environ.get("LOYO_SEEDS", "0").split())
MAIN = "main"
CONFIGS = {"main": "PAF-Net（主模型）"}
# 移除式消融：把整条支路从结构中删掉后重新训练，运行名见 run_removal_ablation.sh
REMOVAL_CONFIGS = {
    "norsbranch": "删除遥感支路",
    "nosoilbranch": "删除土壤静态支路",
    "nocumgdd": "去掉累积生长度日",
}
# 单组件消融：每个配置只删除 PAF-Net 相对 TFT 补充的一处设计，其余结构与训练设置不变。
COMPONENT_CONFIGS = {
    "spatialmean": ("县内空间注意力→等权均值", "ablate-spatialmean-y2022-s0"),
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


def state_compare(e_main, e_ref, states, names):
    """逐州配对比较：对照配置减主模型的州内 RMSE 差（正值=对照更差）。

    州内 RMSE 只在该州的县上计算；不对州做重采样，也不给区间。
    """
    rows = []
    for s in np.unique(states):
        idx = np.flatnonzero(states == s)
        rm_main = float(np.sqrt((e_main[idx] ** 2).mean()))
        rm_ref = float(np.sqrt((e_ref[idx] ** 2).mean()))
        rows.append({"state": s, "name": names.get(s, s), "n": int(len(idx)),
                     "rmse_main": round(rm_main, 2), "rmse_cfg": round(rm_ref, 2),
                     "delta": round(rm_ref - rm_main, 2)})
    d = np.array([r["delta"] for r in rows])
    order = np.argsort(d)
    return {
        "n_states": int(len(rows)),
        "n_worse": int((d > 0).sum()),
        "n_better": int((d < 0).sum()),
        "n_tie": int((d == 0).sum()),
        "median_delta": round(float(np.median(d)), 2),
        "q25": round(float(np.percentile(d, 25)), 2),
        "q75": round(float(np.percentile(d, 75)), 2),
        "min": round(float(d.min()), 2),
        "max": round(float(d.max()), 2),
        "worst": [rows[i] for i in order[::-1][:3]],
        "best": [rows[i] for i in order[:3]],
        "by_state": [rows[i] for i in order[::-1]],
    }


def main():
    labels = load_labels()
    mains = [load_pred(f"{TAG}{MAIN}-y{YEAR}-s{s}") for s in SEEDS]
    mains = [p for p in mains if p]
    assert mains, "缺少 PAF-Net 主模型的 2022 折预测"
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
            row["state_compare"] = state_compare(e_main, e, states, STATE_NAMES)
        out["configs"][key] = row
        print(f"{label:28s} rmse {rmse:6.2f}  delta {row['delta']:+.2f}  anom {row['anom_corr']:.3f}")

    # 移除式消融 / 单组件消融（2022 折、种子 0）
    groups = (
        ("removal_ablation", {k: (v, f"ablate-{k}-y{YEAR}-s0") for k, v in REMOVAL_CONFIGS.items()}),
        ("component_ablation", COMPONENT_CONFIGS),
    )
    for group, configs in groups:
        out[group] = {}
        for key, (label, run_id) in configs.items():
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
            row["state_compare"] = state_compare(e_main, e, states, STATE_NAMES)
            out[group][key] = row
            sc = row["state_compare"]
            print(f"{label:34s} rmse {rmse:6.2f}  delta {row['delta']:+.2f}  anom {row['anom_corr']:.3f}  "
                  f"params {row['params']:,}  州: 变差 {sc['n_worse']}/{sc['n_states']}, "
                  f"中位 {sc['median_delta']:+.2f}")

    out["main"]["params"] = param_count(f"{TAG}{MAIN}-y{YEAR}-s0")

    (OUT / "single_fold_2022.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
