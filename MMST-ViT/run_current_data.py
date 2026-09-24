"""Run the official MMST-ViT on the current CropNet data adapter.

The script keeps official model/loss code and only replaces the dataset path:
  pretrain: 2017-2020 short weather + Sentinel-2
  finetune: 2021 short weather + 2017-2020 long weather
  eval:     2022 short weather + 2017-2021 long weather
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from current_data import CurrentMMSTDataset, build_manifest


def build_loader(manifest: Path, batch_size: int, train: bool):
    records = json.loads(manifest.read_text(encoding="utf-8"))
    dataset = CurrentMMSTDataset(records, train=train)
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=0)


def inspect(args):
    loader = build_loader(Path(args.manifest), args.batch_size, train=False)
    images, short, long, labels, metadata, coords = next(iter(loader))
    print("official MMST-ViT adapter batch")
    print("manifest:", args.manifest)
    print("images:", tuple(images.shape))
    print("short_term:", tuple(short.shape))
    print("long_term:", tuple(long.shape))
    print("labels:", tuple(labels.shape))
    print("coords:", tuple(coords.shape))
    print("first sample:", metadata["FIPS"][0], metadata["year"][0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--build-years", nargs="+", type=int)
    args = parser.parse_args()
    if args.build_years:
        build_manifest(args.build_years, args.manifest)
    inspect(args)


if __name__ == "__main__":
    main()
