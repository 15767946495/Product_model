"""PAF-Net 在生长季内不同时间节点的预测（2022 折，种子 0）。

主模型按日粒度建模，时间注意力带未来屏蔽，因此可以在生长季内任意月末截断输入，
用截至该月的逐日气象与月度植被指数预测当年单产。本脚本对 2022 年的 741 个县分别
截断到 4、5、6、7、8、9 月末（气象序列为每月 28 天，共 168 天），重新前向，
报告各截点的 RMSE、MAE、偏差、$R^2$、州内异常相关，以及相邻截点预测的平均绝对变化。

只评价 PAF-Net 自身，不与比较模型对比。写入 paper_figures/inseason_2022.json。
用法：LOYO_SEEDS=0 python -I TFT_model/inseason_report.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from TFT_model import dataset_adapter as adapter  # noqa: E402
from TFT_model.model_adapter import build_model  # noqa: E402
from TFT_model.paper_report import STATE_NAMES  # noqa: E402
from train_dataset.dataset import DATA_ROOT  # noqa: E402

RUNS = Path(os.environ.get("LOYO_RUNS", "/data/raid0/hqx/TFT_out/tft"))
TAG = os.environ.get("LOYO_TAG", "paper-")
MAIN = os.environ.get("LOYO_MAIN", "main")
SEED = int(os.environ.get("LOYO_SEEDS", "0").split()[0])
JSONL = ROOT / "train_dataset" / "dataset.jsonl"
OUT = ROOT / "paper_figures"
TRAIN_LAST_YEAR = 2021
# 气象序列为 4—9 月每月前 28 天；月末截点即 28 的整数倍。
CUTOFFS = [(28 * k, f"{3 + k}月末") for k in range(1, 7)]
YEARS = (2017, 2018, 2019, 2020, 2021, 2022)


def load_labels():
    labels = {}
    for line in open(JSONL, encoding="utf-8"):
        rec = json.loads(line)
        labels[(rec["fips"], int(rec["year"]))] = float(rec["yield_bu_ac"])
    return labels


def demean(x, groups):
    x = x.copy()
    for g in set(groups):
        x[groups == g] -= x[groups == g].mean()
    return x


def metrics(pred, truth, base_log, fips):
    states = np.array([f[:2] for f in fips])
    e = pred - truth
    sst = ((truth - truth.mean()) ** 2).sum()
    a_hat = demean(np.log(pred) - base_log, states)
    a = demean(np.log(truth) - base_log, states)
    anom = float(np.corrcoef(a_hat, a)[0, 1]) if a_hat.std() > 1e-9 else 0.0
    return {
        "n": int(len(pred)),
        "rmse": float(np.sqrt((e ** 2).mean())),
        "mae": float(np.abs(e).mean()),
        "bias": float(e.mean()),
        "r2": float(1 - (e ** 2).sum() / sst),
        "r": float(np.corrcoef(truth, pred)[0, 1]),
        "pred_sd": float(pred.std()),
        "anom_corr": anom,
    }


@torch.no_grad()
def predict_cutoffs(run_dir: Path, device, labels):
    ck = torch.load(run_dir / "checkpoint-final.pth", map_location="cpu", weights_only=False)
    model = build_model(ck["model_config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    stats = adapter.stats_from_json(ck["stats"])
    store = adapter.load_rs_store(ck["model_config"])
    test_year = int(json.loads((run_dir / "config.json").read_text())["test_year"])
    train_years, test_years = adapter.split_years(test_year)

    def batches_of(years):
        dataset = adapter.build_dataset(JSONL, DATA_ROOT, years, load_images=False)
        loader = adapter.build_loader(dataset, stats, 4, False, use_constructed=ck["use_constructed"],
                                      num_workers=4, rs_store=store)
        return list(loader)

    test_batches = batches_of(test_years)
    fips = [f for b in test_batches for f in b["fips"]]
    truth = np.array([y for b in test_batches for y in b["labels"].tolist()])
    base_log = np.array([
        np.mean([np.log(labels[(c, y)]) for y in range(2017, TRAIN_LAST_YEAR + 1)])
        for c in fips
    ])

    def run(batches, cutoff):
        out = []
        for b in batches:
            inputs = adapter.model_inputs(b, device)
            inputs["seq_lens"] = torch.full_like(inputs["seq_lens"], cutoff)
            pred = model(**inputs)[0][:, 0].cpu()
            if b.get("county_offset") is not None:
                pred = pred + b["county_offset"].float()
            out.extend(torch.exp(pred).tolist())
        return np.array(out)

    preds = {}
    for cutoff, label in CUTOFFS:
        preds[cutoff] = run(test_batches, cutoff)
    return fips, truth, base_log, preds


def main():
    run_dir = RUNS / f"{TAG}{MAIN}-y2022-s{SEED}"
    labels = load_labels()
    device = torch.device("cuda")
    fips, truth, base_log, preds = predict_cutoffs(run_dir, device, labels)

    out = {"run": run_dir.name, "test_year": 2022, "n": int(len(fips)),
           "cutoffs": {}, "transition": {}}
    prev = None
    for cutoff, label in CUTOFFS:
        row = {"label": label, "seq_len": cutoff, **metrics(preds[cutoff], truth, base_log, fips)}
        if prev is not None:
            row["mean_abs_change_prev"] = float(np.abs(preds[cutoff] - preds[prev]).mean())
            row["max_abs_change_prev"] = float(np.abs(preds[cutoff] - preds[prev]).max())
        out["cutoffs"][str(cutoff)] = row
        if prev is not None:
            out["transition"][f"{prev}->{cutoff}"] = {
                "mean_abs_change": row["mean_abs_change_prev"],
                "max_abs_change": row["max_abs_change_prev"],
            }
        prev = cutoff
        print(f"{label:>8s} ({cutoff:3d}天): RMSE {row['rmse']:6.2f}  MAE {row['mae']:6.2f}  "
              f"bias {row['bias']:+6.2f}  R2 {row['r2']:.3f}  anom {row['anom_corr']:.3f}"
              + (f"  |Δprev| {row['mean_abs_change_prev']:.2f}" if "mean_abs_change_prev" in row else ""))

    (OUT / "inseason_2022.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
