"""PA-TFT的测试时遥感干预：模型在预测时是否真正使用了植被指数中的县年信息？

对指定年份的检查点（默认 2022 折、种子 0），先正常预测一次，再在保持天气、土壤、
坐标与县均值偏移不变的前提下替换遥感输入：
  - shuffle：同一测试年份、同一月份内，跨县随机置换各网格的（已去均值、已标准化）植被指数，重复3次；
  - zero   ：全部置零，即"每个网格都与本县往年一样"。
置换保留了每个月份遥感输入的整体分布，只破坏"哪个县对应哪组指数"。若模型利用了县年信息，置换后误差应变大。
写入 paper_figures/rs_intervention_improved.json。
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
from train_dataset.dataset import DATA_ROOT  # noqa: E402

RUNS = Path(os.environ.get("LOYO_RUNS", "/data/raid0/hqx/TFT_out/tft"))
TAG = os.environ.get("LOYO_TAG", "loyo31-")
JSONL = ROOT / "train_dataset" / "dataset.jsonl"
YEARS = tuple(int(y) for y in os.environ.get("LOYO_YEARS", "2022").split())
# 种子列表可用 LOYO_SEEDS 覆盖（单种子复核时 LOYO_SEEDS=0）
SEEDS = tuple(int(s) for s in os.environ.get("LOYO_SEEDS", "42 1 2").split())
N_SHUFFLE = 3


@torch.no_grad()
def predict_variants(run_dir, device, rng_seed=0):
    """返回 fips 顺序、真值、以及 {baseline, shuffle_k, zero} 的原尺度预测。"""
    ck = torch.load(run_dir / "checkpoint-final.pth", map_location="cpu", weights_only=False)
    model = build_model(ck["model_config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    stats = adapter.stats_from_json(ck["stats"])
    store = adapter.load_rs_store(ck["model_config"])
    test_year = int(json.loads((run_dir / "config.json").read_text())["test_year"])
    _, test_years = adapter.split_years(test_year)
    dataset = adapter.build_dataset(JSONL, DATA_ROOT, test_years, load_images=False)
    loader = adapter.build_loader(dataset, stats, 4, False, use_constructed=True, num_workers=4, rs_store=store)
    batches = list(loader)
    valid = [b["ag_grid_mask"].permute(0, 2, 1) & b["grid_mask"][:, :, None] for b in batches]   # (B,G,6)

    def run(features):
        out = []
        for b, f in zip(batches, features):
            inputs = adapter.model_inputs(b, device)
            inputs["rs_features"] = f.to(device)
            pred = model(**inputs)[0][:, 0].cpu() + b["county_offset"].float()
            out.extend(torch.exp(pred).tolist())
        return np.array(out)

    variants = {"baseline": run([b["rs_features"] for b in batches])}
    pools = {m: torch.cat([b["rs_features"][:, m][v[:, :, m]] for b, v in zip(batches, valid)]) for m in range(6)}
    for k in range(N_SHUFFLE):
        g = torch.Generator().manual_seed(rng_seed + k)
        perm = {m: pools[m][torch.randperm(pools[m].shape[0], generator=g)] for m in range(6)}
        cursor = {m: 0 for m in range(6)}
        feats = []
        for b, v in zip(batches, valid):
            f = torch.zeros_like(b["rs_features"])
            for m in range(6):
                n = int(v[:, :, m].sum())
                f[:, m][v[:, :, m]] = perm[m][cursor[m]:cursor[m] + n]
                cursor[m] += n
            feats.append(f)
        variants[f"shuffle_{k}"] = run(feats)
    variants["zero"] = run([torch.zeros_like(b["rs_features"]) for b in batches])
    fips = [f for b in batches for f in b["fips"]]
    truth = np.array([y for b in batches for y in b["labels"].tolist()])
    return fips, truth, variants


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def main():
    device = torch.device("cuda")
    out = {}
    for year in YEARS:
        per_seed, fips0, truth0 = {}, None, None
        for seed in SEEDS:
            fips, truth, v = predict_variants(RUNS / f"{TAG}improved-y{year}-s{seed}", device)
            if fips0 is None:
                fips0, truth0 = fips, truth
            assert fips == fips0 and np.allclose(truth, truth0)
            per_seed[seed] = v
        ens = {name: np.mean([per_seed[s][name] for s in SEEDS], axis=0) for name in per_seed[SEEDS[0]]}
        base = rmse(ens["baseline"], truth0)
        shuffle = [rmse(ens[f"shuffle_{k}"], truth0) for k in range(N_SHUFFLE)]
        out[str(year)] = {
            "rmse_baseline": base, "rmse_shuffle": shuffle, "rmse_shuffle_mean": float(np.mean(shuffle)),
            "rmse_zero": rmse(ens["zero"], truth0),
            "delta_shuffle": float(np.mean(shuffle) - base), "delta_zero": rmse(ens["zero"], truth0) - base,
            "mean_abs_change_shuffle": float(np.mean([np.abs(ens[f"shuffle_{k}"] - ens["baseline"]).mean() for k in range(N_SHUFFLE)])),
            "mean_abs_change_zero": float(np.abs(ens["zero"] - ens["baseline"]).mean()),
        }
        r = out[str(year)]
        print(f"{year}: baseline {base:.2f} | shuffle {r['rmse_shuffle_mean']:.2f} ({r['delta_shuffle']:+.2f}) | zero {r['rmse_zero']:.2f} "
              f"({r['delta_zero']:+.2f}) | mean|Δpred| shuffle {r['mean_abs_change_shuffle']:.2f} zero {r['mean_abs_change_zero']:.2f}", flush=True)
    ds = np.array([out[str(y)]["delta_shuffle"] for y in YEARS]); dz = np.array([out[str(y)]["delta_zero"] for y in YEARS])
    out["summary"] = {"delta_shuffle_mean": float(ds.mean()), "delta_zero_mean": float(dz.mean()),
                      "years_worse_shuffle": int((ds > 0).sum()), "years_worse_zero": int((dz > 0).sum())}
    if len(YEARS) > 1:  # 单折时没有折间检验
        from scipy import stats
        out["summary"].update({"t_p_shuffle": float(stats.ttest_1samp(ds, 0).pvalue),
                               "t_p_zero": float(stats.ttest_1samp(dz, 0).pvalue)})
    print(json.dumps(out["summary"]))
    (ROOT / "paper_figures" / "rs_intervention_improved.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
