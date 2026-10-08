"""正式 TFT 训练入口：官方 MMST Corn 前端加 soil-conditioned TFT。"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from TFT_model.mmst_data import MMSTCornDataset, MMSTCountyLoader
from TFT_model.mmst_tft_model import MMSTTFTYieldModel


MMST_ROOT = ROOT / "MMST-ViT"
DEFAULT_DATA_ROOT = Path("/data/raid0/hqx/Product_model_runtime/DataSrc")
DEFAULT_TRAIN_CONFIG = MMST_ROOT / "data/corn_train.json"
DEFAULT_SOIL_FILE = DEFAULT_DATA_ROOT / "soil/county_soil.json"
DEFAULT_CHECKPOINT = MMST_ROOT / "output_dir/pvt_simclr/checkpoint-latest.pth"
DEFAULT_OUTPUT_DIR = Path("/data/raid0/hqx/TFT_train/mmst_frontend_2021")
FINAL_CHECKPOINT_NAME = "checkpoint-final.pth"

EPOCHS = 100
BASE_LR = 1e-3
WEIGHT_DECAY = 0.05
WARMUP_EPOCHS = 5
TRAIN_YEAR = 2021


def _load_official_data_wrapper():
    """隔离加载官方 DataWrapper，不保留顶层 dataset alias。"""
    sentinel_name = "_tft_mmst_official_sentinel_for_data_wrapper"
    wrapper_name = "_tft_mmst_official_data_wrapper"
    sentinel_path = MMST_ROOT / "dataset/sentinel_loader.py"
    wrapper_path = MMST_ROOT / "dataset/data_wrapper.py"

    sentinel_spec = importlib.util.spec_from_file_location(sentinel_name, sentinel_path)
    if sentinel_spec is None or sentinel_spec.loader is None:
        raise ImportError(f"无法加载官方 Sentinel 模块: {sentinel_path}")
    sentinel_module = importlib.util.module_from_spec(sentinel_spec)
    sentinel_spec.loader.exec_module(sentinel_module)

    package_before = sys.modules.get("dataset")
    sentinel_before = sys.modules.get("dataset.sentinel_loader")
    package = type(sys)("dataset")
    package.__path__ = [str(MMST_ROOT / "dataset")]
    sys.modules["dataset"] = package
    sys.modules["dataset.sentinel_loader"] = sentinel_module
    try:
        wrapper_spec = importlib.util.spec_from_file_location(wrapper_name, wrapper_path)
        if wrapper_spec is None or wrapper_spec.loader is None:
            raise ImportError(f"无法加载官方 DataWrapper 模块: {wrapper_path}")
        wrapper_module = importlib.util.module_from_spec(wrapper_spec)
        wrapper_spec.loader.exec_module(wrapper_module)
    finally:
        if package_before is None:
            sys.modules.pop("dataset", None)
        else:
            sys.modules["dataset"] = package_before
        if sentinel_before is None:
            sys.modules.pop("dataset.sentinel_loader", None)
        else:
            sys.modules["dataset.sentinel_loader"] = sentinel_before
    return wrapper_module.DataWrapper


DataWrapper = _load_official_data_wrapper()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="固定 2021 Corn MMST-TFT 正式训练")
    parser.add_argument("--root-dir", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--train-config", default=str(DEFAULT_TRAIN_CONFIG))
    parser.add_argument("--soil-file", default=str(DEFAULT_SOIL_FILE))
    parser.add_argument("--pretrained", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--checkpoint-name", default=FINAL_CHECKPOINT_NAME)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=BASE_LR)
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--outer-batch-size", type=int, default=1)
    parser.add_argument("--train-year", type=int, default=TRAIN_YEAR)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    return parser


def validate_args(args) -> None:
    expected = {
        "epochs": EPOCHS,
        "lr": BASE_LR,
        "weight_decay": WEIGHT_DECAY,
        "warmup_epochs": WARMUP_EPOCHS,
        "outer_batch_size": 1,
        "train_year": TRAIN_YEAR,
    }
    for name, value in expected.items():
        actual = getattr(args, name)
        if isinstance(value, float):
            valid = math.isclose(actual, value, rel_tol=0.0, abs_tol=1e-12)
        else:
            valid = actual == value
        if not valid:
            raise ValueError(f"固定协议要求 {name}={value}")
    if Path(args.train_config).resolve() != DEFAULT_TRAIN_CONFIG.resolve():
        raise ValueError("固定协议要求 MMST-ViT/data/corn_train.json")
    if Path(args.pretrained).resolve() != DEFAULT_CHECKPOINT.resolve():
        raise ValueError("固定协议严格要求 checkpoint-latest.pth")
    if args.checkpoint_name != FINAL_CHECKPOINT_NAME:
        raise ValueError(f"固定协议要求 {FINAL_CHECKPOINT_NAME}")
    if args.device != "cuda:0":
        raise ValueError("正式训练固定使用单卡 cuda:0")


def transform_training_images(images: torch.Tensor) -> torch.Tensor:
    """将官方 raw HWC Sentinel 张量应用官方训练增强并恢复县级形状。"""
    if images.ndim != 6 or images.shape[1] != 6 or images.shape[-1] != 3:
        raise ValueError("raw images must have shape (B, 6, G, H, W, 3)")
    batch, months, grids, height, width, channels = images.shape
    flat = images.reshape(batch * months * grids, height, width, channels)
    flat = flat.permute(0, 3, 1, 2).to(torch.float32).div(255.0)
    transformed, _ = DataWrapper(train=True)(flat)
    return transformed.reshape(batch, months, grids, 3, 224, 224)


def compute_soil_stats(dataset, train_year: int = TRAIN_YEAR, eps: float = 1e-6):
    fips = [
        code for code, year in zip(dataset.fips_codes, dataset.years)
        if int(year) == int(train_year)
    ]
    if not fips:
        raise ValueError(f"没有 {train_year} soil 训练县")
    values = torch.stack([dataset.soil_by_fips[code].to(torch.float32) for code in fips])
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False).clamp_min(eps)
    source = {"year": int(train_year), "county_count": len(fips)}
    return mean, std, source


def zscore_soil(soil: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return (soil.to(torch.float32) - mean.to(soil.device)) / std.to(soil.device)


def create_optimizer(model, lr=BASE_LR, weight_decay=WEIGHT_DECAY):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.999),
    )


def adjust_learning_rate(
    optimizer,
    progress,
    *,
    base_lr=BASE_LR,
    warmup_epochs=WARMUP_EPOCHS,
    total_epochs=EPOCHS,
):
    progress = max(0.0, min(float(progress), float(total_epochs)))
    if warmup_epochs > 0 and progress <= warmup_epochs:
        lr = base_lr * progress / warmup_epochs
    else:
        ratio = (progress - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        ratio = max(0.0, min(ratio, 1.0))
        lr = 0.5 * base_lr * (1.0 + math.cos(math.pi * ratio))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def atomic_save_checkpoint(
    path,
    *,
    model,
    optimizer,
    epoch,
    config,
    schedule,
    soil_stats,
    source_checkpoint,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "config": dict(config),
        "schedule": dict(schedule),
        "soil_stats": dict(soil_stats),
        "source_checkpoint": str(source_checkpoint),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args):
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("正式训练要求 CUDA cuda:0")
    if not Path(args.pretrained).is_file():
        raise FileNotFoundError(f"官方 checkpoint 不存在: {args.pretrained}")
    _set_seed(args.seed)
    device = torch.device(args.device)

    dataset = MMSTCornDataset(args.root_dir, args.train_config, args.soil_file)
    try:
        if len(dataset) != 195 or any(year != TRAIN_YEAR for year in dataset.years):
            raise ValueError("正式训练要求 corn_train.json 的 195 个 2021 县")
        loader = MMSTCountyLoader(dataset, shuffle=True, num_workers=0)
        soil_mean, soil_std, soil_source = compute_soil_stats(dataset)
        if soil_source["county_count"] != 195:
            raise ValueError("soil stats 必须仅由 195 个 2021 训练县计算")

        model = MMSTTFTYieldModel(
            args.pretrained,
            hidden_size=args.hidden_size,
            heads=args.heads,
            dropout=args.dropout,
        ).to(device)
        optimizer = create_optimizer(model, args.lr, args.weight_decay)
        criterion = nn.MSELoss()
        schedule = {
            "name": "linear-warmup-cosine",
            "base_lr": args.lr,
            "warmup_epochs": args.warmup_epochs,
            "total_epochs": args.epochs,
            "min_lr": 0.0,
        }

        for epoch in range(args.epochs):
            model.train()
            loss_sum = 0.0
            for step, batch in enumerate(loader):
                adjust_learning_rate(
                    optimizer,
                    epoch + step / max(len(loader), 1),
                    base_lr=args.lr,
                    warmup_epochs=args.warmup_epochs,
                    total_epochs=args.epochs,
                )
                images, weather, target, soil, _, _ = batch
                images = transform_training_images(images).to(device, non_blocking=True)
                weather = weather.to(device, non_blocking=True)
                target = target.reshape(-1, 1).to(device, non_blocking=True)
                soil = zscore_soil(soil, soil_mean, soil_std).to(device, non_blocking=True)

                prediction, _, _ = model(images, weather, soil)
                loss = criterion(prediction, target)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError(f"epoch={epoch + 1} step={step} loss 非有限")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradients = [
                    parameter.grad for parameter in model.parameters()
                    if parameter.grad is not None
                ]
                if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
                    raise FloatingPointError(f"epoch={epoch + 1} step={step} gradient 非有限")
                optimizer.step()
                loss_sum += loss.item()
            print(
                f"epoch {epoch + 1:03d}/{args.epochs} "
                f"loss={loss_sum / max(len(loader), 1):.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.8f}"
            )

        checkpoint_path = Path(args.output_dir) / args.checkpoint_name
        config = vars(args).copy()
        atomic_save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=args.epochs,
            config=config,
            schedule=schedule,
            soil_stats={
                "mean": soil_mean.tolist(),
                "std": soil_std.tolist(),
                "source": soil_source,
            },
            source_checkpoint=str(Path(args.pretrained).resolve()),
        )
        print(f"final checkpoint: {checkpoint_path}")
        return checkpoint_path
    finally:
        dataset.close()


def main():
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
