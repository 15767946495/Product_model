"""UNet-ConvLSTM 训练 + 最终测试（2022，只评估 checkpoint-final.pth）。

协议（与 TFT 一致）：--valid-year 2021 时只用 2017--2020 训练，2021 早停选模（patience 3），
最后在 2022 测试；不传 --valid-year 则退回原来的 2017--2021 固定 epoch 训练。
目标为 log(yield) 的训练集标准化值。
输出：output_root/unet_convlstm/<run_id>/{config.json, checkpoint-latest.pth, checkpoint-final.pth,
train_metrics.jsonl, test_metrics.json, county_predictions.jsonl, run_metadata.json}。
"""

from __future__ import annotations

import argparse
import json
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
    parser.add_argument("--eval_jsonl", default="",
                        help="spatial CV: JSONL holding only the held-out counties for the final test")
    parser.add_argument("--valid-year", type=int, default=None,
                        help="hold-out validation year for early stopping (e.g. 2021); training then uses only years before it")
    parser.add_argument("--early-stop-patience", type=int, default=10,
                        help="stop after N epochs without validation improvement (default 3); -1 disables early stopping")
    parser.add_argument("--checkpoint-policy", choices=("best", "final"), default="best",
                        help="best: restore the lowest-validation epoch; final: keep the last epoch (fixed budget, no selection)")
    return parser


def _split(args):
    """返回 (训练年, 验证年或None, 测试年)。"""
    test_year = int(cfg.TEST_YEARS[-1])
    if args.valid_year is not None:
        valid_year = int(args.valid_year)
        if not (min(cfg.TRAIN_YEARS) < valid_year < test_year):
            raise ValueError(f"valid year {valid_year} must lie strictly between {min(cfg.TRAIN_YEARS)} and {test_year}")
        train_years = tuple(y for y in cfg.TRAIN_YEARS if y < valid_year)
        return train_years, (valid_year,), (test_year,)
    return tuple(cfg.TRAIN_YEARS), None, (test_year,)


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
def evaluate_loss(model, loader, stats, device) -> float:
    """验证集上的逐样本 MSE（与训练目标同口径：标准化后的 log yield）。"""
    was_training = model.training
    model.eval()
    total, seen = 0.0, 0
    for batch in loader:
        pred = model(**model_inputs(batch, stats, device))
        target_value = target(batch, stats).to(device)
        total += torch.nn.functional.mse_loss(pred, target_value, reduction="sum").item()
        seen += pred.numel()
    model.train(was_training)
    if seen == 0:
        raise RuntimeError("validation loader has no samples")
    return total / seen


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
                              num_workers=0, eval_jsonl=None) -> dict:
    """只加载 checkpoint-final.pth，在运行配置记录的测试年份上评估。

    ``eval_jsonl`` 用于空间交叉验证：校验仍对照训练数据，测试样本改由留出县文件给出。
    训练/测试年份优先读 run_dir/config.json（支持 --valid-year 的 2017--2020 训练协议），
    旧运行没有该字段时退回 cfg.TRAIN_YEARS / cfg.TEST_YEARS。
    """
    run_dir = Path(output_dir).resolve()
    config_path = run_dir / "config.json"
    run_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    valid_year = run_config.get("valid_year")
    test_years = tuple(cfg.TEST_YEARS)
    if valid_year is not None:
        train_years = tuple(y for y in cfg.TRAIN_YEARS if y < int(valid_year))
    else:
        train_years = tuple(cfg.TRAIN_YEARS)
    checkpoint, _ = ro.validate_unified_run(
        checkpoint_path,
        output_dir,
        model_name=cfg.MODEL_NAME,
        adaptation=cfg.ADAPTATION,
        dataset_jsonl=dataset_jsonl,
        train_years=train_years,
        test_years=test_years,
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
    dataset = build_dataset(eval_jsonl or dataset_jsonl, data_root, test_years, image_size=image_size)
    rows = predict(model, build_loader(dataset, 1, False, num_workers=num_workers), stats, device)
    return ro.finish_test(output_dir, rows, dataset.sample_ids, checkpoint_path=checkpoint_path,
                          years=test_years)


def main(args) -> Path:
    if args.eval_checkpoint:
        run_dir = Path(args.eval_checkpoint).resolve().parent
        evaluate_final_checkpoint(args.eval_checkpoint, args.dataset_jsonl, args.data_root, run_dir,
                                  device=args.device, num_workers=args.num_workers,
                                  eval_jsonl=args.eval_jsonl or None)
        return run_dir

    ro.seed_everything(args.seed)
    device = torch.device(args.device)
    run_dir = ro.make_run_dir(args.output_root, cfg.MODEL_NAME, args.run_id)
    args.run_id = run_dir.name
    ro.write_json(run_dir / "config.json", vars(args))

    train_years, valid_years, test_years = _split(args)
    dataset = build_dataset(args.dataset_jsonl, args.data_root, train_years, image_size=args.image_size)
    stats = fit_stats(dataset, train_years)
    loader = build_loader(dataset, args.county_batch_size, True, num_workers=args.num_workers,
                          generator=torch.Generator().manual_seed(args.seed))
    valid_loader = None
    if valid_years is not None:
        valid_dataset = build_dataset(args.dataset_jsonl, args.data_root, valid_years, image_size=args.image_size)
        if not len(valid_dataset):
            raise ValueError(f"validation years {valid_years} have no samples")
        valid_loader = build_loader(valid_dataset, args.county_batch_size, False,
                                    num_workers=args.num_workers)

    model_config = {**{k: getattr(args, k) for k in MODEL_KEYS}, "image_size": args.image_size}
    checkpoint_metadata = ro.checkpoint_metadata(
        model_name=cfg.MODEL_NAME,
        adaptation=cfg.ADAPTATION,
        dataset_jsonl=args.dataset_jsonl,
        train_years=train_years,
        test_years=test_years,
        model_config=model_config,
    )
    model = build_model(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ro.write_json(run_dir / "run_metadata.json", ro.base_metadata(
        model=cfg.MODEL_NAME, adaptation=cfg.ADAPTATION, run_id=args.run_id, dataset_jsonl=args.dataset_jsonl,
        seed=args.seed, train_years=train_years, test_years=test_years,
        counties=len({s["fips"] for s in dataset.samples}), inputs=cfg.INPUTS, train_samples=len(dataset),
        epochs=args.epochs, optimizer="AdamW", lr=args.lr, weight_decay=args.weight_decay,
        scheduler=f"linear warmup {args.warmup_epochs} epochs + cosine", image_size=args.image_size,
        target="standardized log(yield_bu_ac)", parameters=ro.count_parameters(model),
        checkpoint_policy=("final epoch only, fixed budget" if args.checkpoint_policy == "final"
                           else ("best validation epoch, early stopping" if valid_loader is not None
                                 else "final epoch only, no validation selection")),
        valid_years=list(valid_years) if valid_years is not None else None,
        valid_samples=len(valid_dataset) if valid_loader is not None else None,
        early_stop_patience=args.early_stop_patience if valid_loader is not None else None,
    ))

    best_state, best_epoch, best_valid, stale = None, None, math.inf, 0
    stopped_early = False
    epoch = -1
    for epoch in range(args.epochs):
        row = train_one_epoch(model, loader, optimizer, stats, device, epoch, args)
        if valid_loader is not None:
            valid_loss = evaluate_loss(model, valid_loader, stats, device)
            row["valid_loss"] = valid_loss
            if valid_loss < best_valid:
                best_valid, best_epoch, stale = valid_loss, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
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
        if valid_loader is not None and args.early_stop_patience >= 0 and stale >= args.early_stop_patience:
            stopped_early = True
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, valid loss {best_valid:.6f})")
            break

    if best_state is not None and args.checkpoint_policy == "best":
        model.load_state_dict(best_state)
        epoch = best_epoch
    final = ro.save_checkpoint(run_dir / ro.FINAL_CHECKPOINT, {
        **checkpoint_metadata, "model": model.state_dict(), "epoch": epoch, "stats": stats_to_json(stats),
    })
    ro.update_json(
        run_dir / "run_metadata.json", epoch=epoch, final_checkpoint=str(final),
        final_checkpoint_sha256=ro.sha256_file(final), trained=ro.now(),
        stopped_early=stopped_early,
        best_valid_loss=None if best_epoch is None else best_valid,
        best_valid_epoch=best_epoch,
    )
    evaluate_final_checkpoint(final, args.dataset_jsonl, args.data_root, run_dir, device=args.device,
                              num_workers=args.num_workers, eval_jsonl=args.eval_jsonl or None)
    return run_dir


if __name__ == "__main__":
    main(build_parser().parse_args())
