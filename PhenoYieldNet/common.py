"""Stage 1 / Stage 2 共用：参数、增强、学习率调度与 SimCLR 视图（逐图、与上游一致）。"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torchvision.transforms as transforms

import config as cfg
from train_dataset import run_output as ro


def add_args(parser, stage: dict):
    ro.add_common_args(
        parser,
        epochs=stage["epochs"],
        lr=None,
        output_root=Path(__file__).resolve().parent / "output_dir",
    )
    parser.add_argument("--blr", type=float, default=stage["blr"],
                        help="base lr: absolute lr = blr * batch_size / 256 (upstream rule)")
    parser.add_argument("--batch_size", type=int, default=stage["batch_size"],
                        help="grid-month patches per backbone chunk (upstream --batch_size)")
    parser.add_argument("--warmup_epochs", type=int, default=stage["warmup_epochs"])
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=stage["weight_decay"])
    parser.add_argument("--drop_path", type=float, default=stage["drop_path"])
    return parser


def resolve_lr(args) -> float:
    if args.lr is None:
        args.lr = args.blr * args.batch_size / 256
    return args.lr


def adjust_learning_rate(optimizer, epoch_float: float, args) -> float:
    """上游 util/lr_sched.py：线性 warmup + 半周期余弦。"""
    if epoch_float < args.warmup_epochs:
        lr = args.lr * epoch_float / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch_float - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1e-8)))
    for group in optimizer.param_groups:
        group["lr"] = lr * group.get("lr_scale", 1.0)
    return lr


def param_groups(model, weight_decay: float):
    """timm add_weight_decay：bias / 1 维参数不衰减。"""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if p.ndim <= 1 or name.endswith(".bias") else decay).append(p)
    return [{"params": no_decay, "weight_decay": 0.0}, {"params": decay, "weight_decay": weight_decay}]


def stage1_transform(img_size: int = cfg.IMG_SIZE) -> transforms.Compose:
    """上游 time_aware_wrapper.SimCLRDataTransform（kernel 3，antialias=False）。"""
    return transforms.Compose([
        transforms.RandomResizedCrop(size=img_size, antialias=False),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=90),
        transforms.GaussianBlur(kernel_size=3),
    ])


def stage2_transform(img_size: int = cfg.IMG_SIZE, train: bool = True) -> transforms.Compose:
    """上游 dataset/data_wrapper.DataWrapper：训练 RandomResizedCrop+Flip+Blur(9)，测试 CenterCrop。"""
    if not train:
        return transforms.Compose([transforms.CenterCrop(size=img_size)])
    return transforms.Compose([
        transforms.RandomResizedCrop(size=img_size, antialias=True),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=9),
    ])


def per_image(transform, x: torch.Tensor) -> torch.Tensor:
    """逐张采样增强参数（上游 Stage 1 逐图增强）。"""
    return torch.stack([transform(img) for img in x])
