"""PhenoYieldNet-unified Stage 1：2017--2020 时序掩码视图对比适配（不使用标签），SpectralGPT+ 初始化。

与上游 train_da.py 一致：每个县年先按 frame_drop_rate 随机把 (月, 网格) 图像置零，
展开为 (网格 x 月) 张图像，每张独立做两次 SimCLR 增强，配同网格同月 28 天 x 9 维天气上下文，
按 --batch_size 分块计算 ContrastiveLoss。checkpoint['model'] 即 Stage 2 使用的骨干 state_dict。
输出：output_root/phenoyieldnet_stage1/<run_id>/{config.json, checkpoint-latest.pth, checkpoint-final.pth,
train_metrics.jsonl, run_metadata.json}。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from einops import rearrange

import config as cfg
from backbone import build_backbone, load_pretrained
from common import add_args, adjust_learning_rate, param_groups, per_image, resolve_lr, stage1_transform
from dataset_adapter import build_dataset, build_loader, require_unpadded
from loss.contrastive_loss import ContrastiveLoss
from train_dataset import run_output as ro

STAGE1_MODEL_NAME = "phenoyieldnet_stage1"
STAGE1_ADAPTATION = "PhenoYieldNet-unified contrastive adaptation"


def build_parser():
    parser = argparse.ArgumentParser("PhenoYieldNet-unified Stage 1 (2017-2020 contrastive adaptation)")
    add_args(parser, cfg.STAGE1)
    parser.add_argument("--pretrain_model", default=cfg.SPECTRALGPT_PATH, help="SpectralGPT+.pth")
    parser.add_argument("--pretrain_sha256", default=cfg.SPECTRALGPT_SHA256,
                        help="expected SHA-256 of --pretrain_model; empty string skips the check")
    parser.add_argument("--no_pretrain", action="store_true", help="random init (tests only)")
    parser.add_argument("--frame_drop_rate", type=float, default=cfg.STAGE1["frame_drop_rate"])
    return parser


def contrastive_views(batch, drop_rate: float, transform, generator=None):
    """上游 Sentinel_Frame_Drop_Subset：(月,网格) 图像按概率置零，展开为 (b g t) 顺序后逐图增强。"""
    images, context = batch["images"], batch["weather_context"]
    keep = torch.rand(images.shape[:3], generator=generator) > drop_rate
    images = images * keep[..., None, None, None].to(images.dtype)
    flat = rearrange(images, 'b t g c h w -> (b g t) c h w')
    ys = rearrange(context, 'b t g n d -> (b g t) n d')
    return per_image(transform, flat), per_image(transform, flat), ys


def train_one_epoch(model, loader, optimizer, device, epoch, args, generator) -> dict:
    model.train()
    transform = stage1_transform()
    total, steps_done = 0.0, 0
    steps = len(loader)
    for step, batch in enumerate(loader):
        require_unpadded(batch)
        adjust_learning_rate(optimizer, step / steps + epoch, args)
        xi_all, xj_all, ys_all = contrastive_views(batch, args.frame_drop_rate, transform, generator)
        for start in range(0, xi_all.shape[0], args.batch_size):
            xi = xi_all[start:start + args.batch_size].to(device, non_blocking=True)
            xj = xj_all[start:start + args.batch_size].to(device, non_blocking=True)
            ys = ys_all[start:start + args.batch_size].to(device, non_blocking=True)
            zi, zj = model(xi, ys), model(xj, ys)
            loss = ContrastiveLoss(zi.shape[0], device)(zi, zj)
            if not math.isfinite(loss.item()):
                raise FloatingPointError(f"{batch['sample_id']}: loss {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
            steps_done += 1
        print(f"Epoch [{epoch}] [{step}/{steps - 1}] {batch['sample_id'][0]} patches {xi_all.shape[0]}")
    return {"train_loss": total / max(steps_done, 1), "lr": optimizer.param_groups[0]["lr"]}


def checkpoint_payload(model, optimizer, epoch, args) -> dict:
    return {
        **ro.checkpoint_metadata(
            model_name=STAGE1_MODEL_NAME,
            adaptation=STAGE1_ADAPTATION,
            dataset_jsonl=args.dataset_jsonl,
            train_years=cfg.PRETRAIN_YEARS,
            test_years=cfg.TEST_YEARS,
            model_config={"drop_path": args.drop_path, "num_classes": 1},
        ),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }


def main(args) -> Path:
    ro.seed_everything(args.seed)
    device = torch.device(args.device)
    run_dir = ro.make_run_dir(args.output_root, cfg.STAGE1_NAME, args.run_id)
    args.run_id = run_dir.name
    resolve_lr(args)
    ro.write_json(run_dir / "config.json", vars(args))

    net = build_backbone(args.drop_path, num_classes=1)  # 上游 Stage 1 num_classes=1
    load_report = None
    if not args.no_pretrain:
        load_report = load_pretrained(net, args.pretrain_model, expected_sha256=args.pretrain_sha256 or None)
        ro.write_json(run_dir / "pretrained_load_report.json", load_report)
    net.to(device)

    dataset = build_dataset(args.dataset_jsonl, args.data_root, cfg.PRETRAIN_YEARS)
    loader = build_loader(dataset, True, num_workers=args.num_workers,
                          generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.AdamW(param_groups(net, args.weight_decay), lr=args.lr, betas=(0.9, 0.95))
    ro.write_json(run_dir / "run_metadata.json", ro.base_metadata(
        model=cfg.STAGE1_NAME, adaptation=STAGE1_ADAPTATION, run_id=args.run_id,
        dataset_jsonl=args.dataset_jsonl,
        seed=args.seed, train_years=cfg.PRETRAIN_YEARS, test_years=cfg.TEST_YEARS,
        counties=len({s["fips"] for s in dataset.samples}), inputs=cfg.INPUTS, train_samples=len(dataset),
        labels_used=False, test_executed=False, epochs=args.epochs,
        optimizer="AdamW(0.9,0.95)", lr=args.lr, blr=args.blr, weight_decay=args.weight_decay,
        backbone=cfg.BACKBONE, pretrained=None if load_report is None else {
            "url": cfg.SPECTRALGPT_URL, "path": load_report["path"], "sha256": load_report.get("sha256"),
            "license": cfg.SPECTRALGPT_LICENSE, "loaded": len(load_report["loaded_keys"]),
            "missing": len(load_report["missing_keys"]), "unexpected": len(load_report["unexpected_keys"])},
        parameters=ro.count_parameters(net),
    ))

    generator = torch.Generator().manual_seed(args.seed + 1)
    epoch = -1
    for epoch in range(args.epochs):
        row = train_one_epoch(net, loader, optimizer, device, epoch, args, generator)
        ro.append_jsonl(run_dir / "train_metrics.jsonl", {"epoch": epoch, **row})
        if epoch % args.save_freq == 0 or epoch + 1 == args.epochs:
            latest = ro.save_checkpoint(
                run_dir / ro.LATEST_CHECKPOINT,
                checkpoint_payload(net, optimizer, epoch, args),
            )
            ro.update_json(
                run_dir / "run_metadata.json", epoch=epoch, latest_checkpoint=str(latest),
                latest_checkpoint_sha256=ro.sha256_file(latest),
            )
    final = ro.save_checkpoint(
        run_dir / ro.FINAL_CHECKPOINT,
        checkpoint_payload(net, optimizer, epoch, args),
    )
    ro.update_json(
        run_dir / "run_metadata.json", status="finished", epoch=epoch,
        final_checkpoint=str(final), final_checkpoint_sha256=ro.sha256_file(final),
        checkpoint_sha256=ro.sha256_file(final), finished=ro.now(),
    )
    return run_dir


if __name__ == "__main__":
    main(build_parser().parse_args())
