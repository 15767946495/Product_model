"""UNet-ConvLSTM 训练（2017--2021 各县年）+ 最终测试（2022，只评估 checkpoint-final.pth）。

固定 epoch，无验证集、无早停、无按指标选择 checkpoint。目标为 log(yield) 的训练集标准化值。
输出：output_root/unet_convlstm/<run_id>/{config.json, checkpoint-latest.pth, checkpoint-final.pth,
train_metrics.jsonl, test_metrics.json, county_predictions.jsonl, run_metadata.json}。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

import config as cfg
from dataset_adapter import (
    build_dataset, build_loader, fit_stats, model_inputs, stats_from_json, stats_to_json, target,
)
from model import UNetConvLSTM
from train_dataset import run_output as ro

MODEL_KEYS = ("base_channels", "weather_dim", "hidden_channels", "dropout", "grid_chunk")


def build_parser():
    d = cfg.DEFAULTS
    parser = argparse.ArgumentParser("UNet-ConvLSTM (unified CropNet, train 2017-2021 / test 2022)")
    ro.add_common_args(
        parser,
        epochs=d["epochs"],
        lr=d["lr"],
        output_root=Path(__file__).resolve().parent / "output_dir",
    )
    for key in ("image_size", "base_channels", "weather_dim", "hidden_channels", "county_batch_size", "grid_chunk",
                "warmup_epochs"):
        parser.add_argument(f"--{key}", type=int, default=d[key])
    parser.add_argument("--dropout", type=float, default=d["dropout"])
    parser.add_argument("--weight_decay", type=float, default=d["weight_decay"])
    parser.add_argument("--eval_checkpoint", default="", help="only run the final 2022 test on this checkpoint-final.pth")
    return parser


def build_model(model_config: dict) -> UNetConvLSTM:
    return UNetConvLSTM(**{k: model_config[k] for k in MODEL_KEYS})


def lr_at(epoch_float: float, args) -> float:
    """线性 warmup + 半周期余弦，按迭代调整。"""
    if epoch_float < args.warmup_epochs:
        return args.lr * epoch_float / max(args.warmup_epochs, 1e-8)
    progress = (epoch_float - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1e-8)
    return 0.5 * args.lr * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def train_one_epoch(model, loader, optimizer, stats, device, epoch, args) -> dict:
    model.train()
    total, seen = 0.0, 0
    steps = len(loader)
    for step, batch in enumerate(loader):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(epoch + step / steps, args)
        pred = model(**model_inputs(batch, stats, device))
        loss = torch.nn.functional.mse_loss(pred, target(batch, stats).to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError(f"epoch {epoch} step {step} {batch['sample_id']}: loss {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(batch["sample_id"])
        seen += len(batch["sample_id"])
        print(f"Epoch [{epoch}] [{step}/{steps - 1}] {batch['sample_id']} loss {loss.item():.4f}")
    return {"train_loss": total / max(seen, 1), "lr": optimizer.param_groups[0]["lr"],
            "optimizer_steps": steps}


@torch.no_grad()
def predict(model, loader, stats, device) -> list[dict]:
    model.eval()
    rows = []
    for batch in loader:
        pred_std = model(**model_inputs(batch, stats, device)).float().cpu()
        pred = torch.exp(pred_std * stats["y_std"] + stats["y_mean"])
        for sid, fips, year, y, p in zip(batch["sample_id"], batch["fips"], batch["year"].tolist(),
                                         batch["labels"].tolist(), pred.tolist()):
            rows.append({"sample_id": sid, "fips": fips, "year": year, "y_true": y, "y_pred": p})
    return rows


def evaluate_final_checkpoint(checkpoint_path, dataset_jsonl, data_root, output_dir, *, device="cuda",
                              num_workers=0) -> dict:
    """只加载 checkpoint-final.pth，只读 2022。"""
    checkpoint, _ = ro.validate_unified_run(
        checkpoint_path,
        output_dir,
        model_name=cfg.MODEL_NAME,
        adaptation=cfg.ADAPTATION,
        dataset_jsonl=dataset_jsonl,
        train_years=cfg.TRAIN_YEARS,
        test_years=cfg.TEST_YEARS,
        required_fields=tuple(f"model_config.{key}" for key in (*MODEL_KEYS, "image_size")) + tuple(
            f"stats.{key}" for key in ("weather_mean", "weather_std", "y_mean", "y_std")
        ),
        log_loaded=False,
    )
    image_size = checkpoint["model_config"].get("image_size")
    if type(image_size) is not int or image_size <= 0:
        raise ValueError("checkpoint.model_config.image_size: expected positive integer")
    print(f"Loaded final checkpoint: {Path(checkpoint_path).resolve()}")
    device = torch.device(device)
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    stats = stats_from_json(checkpoint["stats"])
    dataset = build_dataset(dataset_jsonl, data_root, cfg.TEST_YEARS, image_size=image_size)
    rows = predict(model, build_loader(dataset, 1, False, num_workers=num_workers), stats, device)
    return ro.finish_test(output_dir, rows, dataset.sample_ids, checkpoint_path=checkpoint_path,
                          years=cfg.TEST_YEARS)


def main(args) -> Path:
    if args.eval_checkpoint:
        run_dir = Path(args.eval_checkpoint).resolve().parent
        evaluate_final_checkpoint(args.eval_checkpoint, args.dataset_jsonl, args.data_root, run_dir,
                                  device=args.device, num_workers=args.num_workers)
        return run_dir

    ro.seed_everything(args.seed)
    device = torch.device(args.device)
    run_dir = ro.make_run_dir(args.output_root, cfg.MODEL_NAME, args.run_id)
    args.run_id = run_dir.name
    ro.write_json(run_dir / "config.json", vars(args))

    dataset = build_dataset(args.dataset_jsonl, args.data_root, cfg.TRAIN_YEARS, image_size=args.image_size)
    stats = fit_stats(dataset)
    loader = build_loader(dataset, args.county_batch_size, True, num_workers=args.num_workers,
                          generator=torch.Generator().manual_seed(args.seed))

    model_config = {**{k: getattr(args, k) for k in MODEL_KEYS}, "image_size": args.image_size}
    checkpoint_metadata = ro.checkpoint_metadata(
        model_name=cfg.MODEL_NAME,
        adaptation=cfg.ADAPTATION,
        dataset_jsonl=args.dataset_jsonl,
        train_years=cfg.TRAIN_YEARS,
        test_years=cfg.TEST_YEARS,
        model_config=model_config,
    )
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ro.write_json(run_dir / "run_metadata.json", ro.base_metadata(
        model=cfg.MODEL_NAME, adaptation=cfg.ADAPTATION, run_id=args.run_id, dataset_jsonl=args.dataset_jsonl,
        seed=args.seed, train_years=cfg.TRAIN_YEARS, test_years=cfg.TEST_YEARS,
        counties=len({s["fips"] for s in dataset.samples}), inputs=cfg.INPUTS, train_samples=len(dataset),
        epochs=args.epochs, optimizer="AdamW", lr=args.lr, weight_decay=args.weight_decay,
        scheduler=f"linear warmup {args.warmup_epochs} epochs + cosine", image_size=args.image_size,
        target="standardized log(yield_bu_ac)", parameters=ro.count_parameters(model),
        checkpoint_policy="final epoch only, no validation selection",
    ))

    epoch = -1
    for epoch in range(args.epochs):
        row = train_one_epoch(model, loader, optimizer, stats, device, epoch, args)
        ro.append_jsonl(run_dir / "train_metrics.jsonl", {"epoch": epoch, **row})
        if epoch % args.save_freq == 0 or epoch + 1 == args.epochs:
            latest = ro.save_checkpoint(run_dir / ro.LATEST_CHECKPOINT, {
                **checkpoint_metadata,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "stats": stats_to_json(stats)})
            ro.update_json(
                run_dir / "run_metadata.json", epoch=epoch, latest_checkpoint=str(latest),
                latest_checkpoint_sha256=ro.sha256_file(latest),
            )

    final = ro.save_checkpoint(run_dir / ro.FINAL_CHECKPOINT, {
        **checkpoint_metadata, "model": model.state_dict(), "epoch": epoch, "stats": stats_to_json(stats),
    })
    ro.update_json(
        run_dir / "run_metadata.json", epoch=epoch, final_checkpoint=str(final),
        final_checkpoint_sha256=ro.sha256_file(final), trained=ro.now(),
    )
    evaluate_final_checkpoint(final, args.dataset_jsonl, args.data_root, run_dir, device=args.device,
                              num_workers=args.num_workers)
    return run_dir


if __name__ == "__main__":
    main(build_parser().parse_args())
