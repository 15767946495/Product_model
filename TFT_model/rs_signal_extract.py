"""提取冻结特征，用于检验遥感是否在县固定效应之外携带单产信息。

对每个县年样本、每个有效网格-月份影像计算：
- 冻结 ImageNet 预训练 PVT-Tiny 的 CLS 表示（未经任何产量监督）；
- 可见光植被指数：ExG=2g-r-b、VARI=(G-R)/(G+R-B)、GCC=G/(R+G+B)、亮度，
  并给出像素中位数与"绿色像素占比"(ExG>0.05)。
再在县内有效网格上等权平均，得到 (月份, 特征) 的县年表示。只读不写数据集。

输出 npz：sample_id/fips/year/label、pvt (N,6,512)、veg (N,6,K)、month_valid (N,6)。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from TFT_model.pvt import DEFAULT_PVT_PRETRAINED_PATH, PVTTinyEncoder
from train_dataset.collate import collate_county_year
from train_dataset.dataset import (
    DATA_ROOT, DAYS_PER_MONTH, REMOTE_MONTHS, WEATHER_FEATURES, YEARS, CropNetDataset,
)

VEG_FEATURES = (
    "exg_mean", "exg_median", "green_frac", "vari_mean", "vari_median",
    "gcc_mean", "brightness_mean", "r_mean", "g_mean", "b_mean",
)


def vegetation_indices(images: torch.Tensor) -> torch.Tensor:
    """images: (N,3,H,W) in [0,1] → (N, len(VEG_FEATURES))。"""
    r, g, b = images[:, 0], images[:, 1], images[:, 2]
    total = (r + g + b).clamp_min(1e-6)
    rn, gn, bn = r / total, g / total, b / total
    exg = 2 * gn - rn - bn
    denominator = g + r - b
    # VARI 分母接近 0 时不稳定：只在 |分母|>0.02 的像素上统计
    stable = denominator.abs() > 0.02
    vari = torch.where(stable, (g - r) / torch.where(stable, denominator, torch.ones_like(denominator)),
                       torch.zeros_like(denominator)).clamp(-1, 1)
    flat = lambda x: x.flatten(1)
    vari_count = flat(stable).sum(1).clamp_min(1)
    vari_mean = flat(vari).sum(1) / vari_count
    vari_median = torch.stack([
        v[s].median() if s.any() else v.new_zeros(()) for v, s in zip(flat(vari), flat(stable))
    ])
    return torch.stack([
        flat(exg).mean(1), flat(exg).median(1).values, (flat(exg) > 0.05).float().mean(1),
        vari_mean, vari_median, flat(gn).mean(1), flat(total).mean(1) / 3,
        flat(r).mean(1), flat(g).mean(1), flat(b).mean(1),
    ], dim=1)


WEATHER_SUMMARY = tuple(WEATHER_FEATURES) + ("GDD", "KDD", "PET_deficit")


def county_monthly_weather(batch: dict) -> torch.Tensor:
    """县内有效网格平均后的月度天气：11 个原始变量月均值 + 月 GDD、KDD 与(降水-PET)月累计。

    返回 (B, 6, 14)。复用 TFT 的构造特征定义，累计量取每月增量，避免与月份强共线。
    """
    from TFT_model.dataset_adapter import append_constructed_features

    weather = batch["weather"]                            # (B,G,168,11) 原始单位
    mask = batch["grid_mask"].to(weather.dtype)           # (B,G)
    feats = append_constructed_features(weather, batch["coords"], ("CumGDD", "KDD", "CumDeficit"))
    B, G, T, F = feats.shape
    monthly = feats.reshape(B, G, len(REMOTE_MONTHS), DAYS_PER_MONTH, F)
    raw = monthly[..., :len(WEATHER_FEATURES)].mean(3)                  # 月均值
    cumulative = monthly[..., len(WEATHER_FEATURES):]
    end = cumulative[:, :, :, -1]
    start = torch.cat([torch.zeros_like(end[:, :, :1]), end[:, :, :-1]], dim=2)
    month_totals = end - start                                         # 每月增量
    per_grid = torch.cat([raw, month_totals], dim=-1)                  # (B,G,6,14)
    weights = mask[:, :, None, None]
    return (per_grid * weights).sum(1) / weights.sum(1).clamp_min(1.0)


def county_mean(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """values (B,6,G,K), valid (B,6,G) → (B,6,K)；无有效网格的月份为 0。"""
    weights = valid.to(values.dtype)[..., None]
    return (values * weights).sum(2) / weights.sum(2).clamp_min(1.0)


@torch.no_grad()
def extract(dataset, encoder, device, *, batch_size: int, num_workers: int, chunk: int = 256) -> dict:
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_county_year, persistent_workers=False,
    )
    out = {key: [] for key in ("sample_id", "fips", "year", "label", "pvt", "veg", "month_valid", "n_grids",
                               "weather")}
    for step, batch in enumerate(loader):
        images = batch["images"]                         # (B,6,G,3,S,S)
        valid = batch["image_mask"] & batch["grid_mask"][:, None, :]
        B, M, G = valid.shape
        flat_valid = valid.reshape(-1)
        picked = images.reshape(B * M * G, *images.shape[3:])[flat_valid]
        pvt = torch.zeros(B * M * G, encoder.norm.normalized_shape[0])
        veg = torch.zeros(B * M * G, len(VEG_FEATURES))
        pvt_parts, veg_parts = [], []
        for start in range(0, picked.shape[0], chunk):
            part = picked[start:start + chunk].to(device, non_blocking=True)
            pvt_parts.append(encoder(part).float().cpu())
            veg_parts.append(vegetation_indices(part).float().cpu())
        if pvt_parts:
            pvt[flat_valid] = torch.cat(pvt_parts)
            veg[flat_valid] = torch.cat(veg_parts)
        out["pvt"].append(county_mean(pvt.reshape(B, M, G, -1), valid).half())
        out["veg"].append(county_mean(veg.reshape(B, M, G, -1), valid))
        out["month_valid"].append(valid.any(dim=2))
        out["n_grids"].append(batch["grid_mask"].sum(1))
        out["weather"].append(county_monthly_weather(batch))
        out["sample_id"].extend(batch["sample_id"])
        out["fips"].extend(batch["fips"])
        out["year"].append(batch["year"])
        out["label"].append(batch["labels"])
        if step % 20 == 0:
            print(f"[{step}/{len(loader)}] {batch['sample_id'][0]}", flush=True)
    return {
        "sample_id": np.array(out["sample_id"]), "fips": np.array(out["fips"]),
        "year": torch.cat(out["year"]).numpy(), "label": torch.cat(out["label"]).numpy(),
        "pvt": torch.cat(out["pvt"]).numpy(), "veg": torch.cat(out["veg"]).numpy(),
        "month_valid": torch.cat(out["month_valid"]).numpy(), "n_grids": torch.cat(out["n_grids"]).numpy(),
        "weather": torch.cat(out["weather"]).numpy(),
        "veg_features": np.array(VEG_FEATURES),
        "weather_features": np.array(WEATHER_SUMMARY),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Extract frozen remote-sensing features for the signal test")
    parser.add_argument("--dataset-jsonl", default=str(_PROJECT_ROOT / "train_dataset" / "dataset.jsonl"))
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--pvt_pretrained_path", default=str(DEFAULT_PVT_PRETRAINED_PATH))
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--years", default=",".join(str(y) for y in YEARS))
    return parser


def main(args):
    years = tuple(int(y) for y in args.years.split(","))
    dataset = CropNetDataset(Path(args.dataset_jsonl), years, load_images=True, data_root=Path(args.data_root))
    device = torch.device(args.device)
    encoder = PVTTinyEncoder(out_dim=512, pretrained=True, pretrained_path=args.pvt_pretrained_path).to(device).eval()
    result = extract(dataset, encoder, device, batch_size=args.batch_size, num_workers=args.num_workers)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **result)
    print(f"saved {len(result['sample_id'])} samples to {args.output}")


if __name__ == "__main__":
    main(build_parser().parse_args())
