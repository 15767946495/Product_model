"""逐网格-月份的可见光植被指数缓存，作为 TFT 的数值化遥感输入（替代 PVT 图像编码）。

特征由 Sentinel-2 RGB 影像逐像素计算（见 rs_signal_extract.vegetation_indices），不使用任何标签，
也不含可训练参数。输出 npz，按 sample_id 索引：
  features (N, 6, Gmax, K) float32：网格 g 在第 m 个月的 K 个特征，超出该县网格数的位置为 0；
  grid_count (N,)：该样本真实网格数；sample_id (N,)；feature_names (K,)。
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

from TFT_model.rs_signal_extract import VEG_FEATURES, vegetation_indices
from train_dataset.collate import collate_county_year
from train_dataset.dataset import DATA_ROOT, REMOTE_MONTHS, YEARS, CropNetDataset

DEFAULT_PATH = Path("/data/raid0/hqx/train_dataset/rs_grid_features.npz")


@torch.no_grad()
def compute(dataset, device, *, batch_size: int, num_workers: int, chunk: int = 256) -> dict:
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_county_year,
    )
    ids, per_sample = [], []
    for step, batch in enumerate(loader):
        images = batch["images"]                                  # (B,6,G,3,S,S)
        valid = batch["image_mask"] & batch["grid_mask"][:, None, :]
        B, M, G = valid.shape
        flat_valid = valid.reshape(-1)
        picked = images.reshape(B * M * G, *images.shape[3:])[flat_valid]
        out = torch.zeros(B * M * G, len(VEG_FEATURES))
        parts = [
            vegetation_indices(picked[start:start + chunk].to(device)).float().cpu()
            for start in range(0, picked.shape[0], chunk)
        ]
        if parts:
            out[flat_valid] = torch.cat(parts)
        out = out.reshape(B, M, G, len(VEG_FEATURES))
        counts = batch["grid_mask"].sum(1).tolist()
        for b, count in enumerate(counts):
            if not bool(valid[b, :, :count].all()):
                raise ValueError(f"{batch['sample_id'][b]}: missing image for a real grid-month")
            per_sample.append(out[b, :, :count].numpy())
        ids.extend(batch["sample_id"])
        if step % 20 == 0:
            print(f"[{step}/{len(loader)}] {batch['sample_id'][0]}", flush=True)
    gmax = max(a.shape[1] for a in per_sample)
    features = np.zeros((len(per_sample), len(REMOTE_MONTHS), gmax, len(VEG_FEATURES)), dtype=np.float32)
    for i, array in enumerate(per_sample):
        features[i, :, :array.shape[1]] = array
    if not np.isfinite(features).all():
        raise ValueError("non-finite vegetation features")
    return {
        "sample_id": np.array(ids), "features": features,
        "grid_count": np.array([a.shape[1] for a in per_sample]),
        "feature_names": np.array(VEG_FEATURES),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Cache per-grid-month vegetation-index features")
    parser.add_argument("--dataset-jsonl", default=str(_PROJECT_ROOT / "train_dataset" / "dataset.jsonl"))
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--output", default=str(DEFAULT_PATH))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=12)
    return parser


def main(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"{output} already exists; remove it explicitly to recompute")
    dataset = CropNetDataset(Path(args.dataset_jsonl), YEARS, load_images=True, data_root=Path(args.data_root))
    result = compute(dataset, torch.device(args.device), batch_size=args.batch_size, num_workers=args.num_workers)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **result)
    print(f"saved {len(result['sample_id'])} samples, features {result['features'].shape} to {output}")


if __name__ == "__main__":
    main(build_parser().parse_args())
