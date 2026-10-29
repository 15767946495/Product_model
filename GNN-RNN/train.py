"""GNN-RNN 训练（2017--2021 窗口预测 2021）+ 最终测试（2018--2022 窗口预测 2022，只评估 checkpoint-final.pth）。

固定 epoch，无验证集、无早停、无按指标选择 checkpoint。
输出：output_root/gnn_rnn/<run_id>/{config.json, checkpoint-latest.pth, checkpoint-final.pth,
train_metrics.jsonl, test_metrics.json, county_predictions.jsonl, run_metadata.json}。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

import config as cfg
from dataset_adapter import (
    TARGET_TEST_YEAR, TARGET_TRAIN_YEAR, TEST_YEARS, TRAIN_YEARS, fit_stats, load_windows, model_inputs,
    stats_from_json, stats_to_json,
)
from model import GNNRNN
from train_dataset import run_output as ro

MODEL_KEYS = ("aggregator", "n_layers", "z_dim", "dropout")


def build_parser():
    d = cfg.DEFAULTS
    parser = argparse.ArgumentParser("GNN-RNN (unified CropNet, train window 2017-2021, test window 2018-2022)")
    ro.add_common_args(
        parser,
        epochs=d["epochs"],
        lr=d["lr"],
        output_root=Path(__file__).resolve().parent / "output_dir",
    )
    parser.add_argument("--aggregator", choices=cfg.AGGREGATORS, default=d["aggregator"],
                        help=f"main config fixed to {cfg.MAIN_AGGREGATOR}; {cfg.ABLATION_AGGREGATOR} is the pre-declared ablation")
    parser.add_argument("--n_layers", type=int, default=d["n_layers"])
    parser.add_argument("--z_dim", type=int, default=d["z_dim"])
    parser.add_argument("--dropout", type=float, default=d["dropout"])
    parser.add_argument("--batch_size", type=int, default=d["batch_size"], help="target counties per optimizer step")
    parser.add_argument("--weight_decay", type=float, default=d["weight_decay"])
    parser.add_argument("--T0", type=int, default=d["T0"])
    parser.add_argument("--T_mult", type=int, default=d["T_mult"])
    parser.add_argument("--eta_min", type=float, default=d["eta_min"])
    parser.add_argument("--eval_checkpoint", default="", help="only run the final 2022 test on this checkpoint-final.pth")
    return parser


def logcosh(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """上游 loss_fn(mode="logcosh")，在标准化尺度上计算。"""
    return torch.mean(torch.log(torch.cosh(target - pred + 1e-12)))


def build_model(model_config: dict) -> GNNRNN:
    return GNNRNN(**{k: model_config[k] for k in MODEL_KEYS})


def train_one_epoch(model, inputs, y_std_target, optimizer, scheduler, epoch, args, generator) -> dict:
    model.train()
    n = y_std_target.shape[0]
    order = torch.randperm(n, generator=generator)
    steps = math.ceil(n / args.batch_size)
    total = 0.0
    for step in range(steps):
        idx = order[step * args.batch_size:(step + 1) * args.batch_size].to(y_std_target.device)
        # 全图前向；损失只作用于本步目标县的最后一年（上游 c1=0, c2=1）
        pred = model(inputs["weather"], inputs["soil"], inputs["edge_index"])[:, -1]
        loss = logcosh(pred[idx], y_std_target[idx])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"epoch {epoch} step {step}: loss is {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + (step + 1) / steps)
        total += loss.item()
    return {"train_loss": total / steps, "lr": optimizer.param_groups[0]["lr"],
            "optimizer_steps": steps}


@torch.no_grad()
def predict(model, batch, stats, device) -> torch.Tensor:
    """返回末年预测（bu/ac），按窗口节点（fips）顺序。"""
    model.eval()
    inputs = {k: v.to(device) for k, v in model_inputs(batch, stats).items()}
    pred_std = model(inputs["weather"], inputs["soil"], inputs["edge_index"])[:, -1].cpu()
    return pred_std * stats["y_std"] + stats["y_mean"]


def evaluate_final_checkpoint(checkpoint_path, dataset_jsonl, data_root, output_dir, *, device="cuda",
                              windows=None) -> dict:
    """只加载 checkpoint-final.pth，只用 2018--2022 窗口预测 2022。"""
    checkpoint, _ = ro.validate_unified_run(
        checkpoint_path,
        output_dir,
        model_name=cfg.MODEL_NAME,
        adaptation=cfg.ADAPTATION,
        dataset_jsonl=dataset_jsonl,
        train_years=TRAIN_YEARS,
        test_years=TEST_YEARS,
        required_fields=tuple(f"model_config.{key}" for key in MODEL_KEYS) + tuple(
            f"stats.{key}" for key in (
                "weather_mean", "weather_std", "soil_mean", "soil_std", "y_mean", "y_std",
            )
        ),
    )
    device = torch.device(device)
    model = build_model(checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    stats = stats_from_json(checkpoint["stats"])
    test = (windows or load_windows(dataset_jsonl, data_root))["test"]
    if test["years"].tolist() != list(TEST_YEARS):
        raise ValueError(f"test window years {test['years'].tolist()} != {list(TEST_YEARS)}")
    pred = predict(model, test, stats, device)
    rows = [{"sample_id": sid, "fips": fips, "year": TARGET_TEST_YEAR, "y_true": float(y), "y_pred": float(p)}
            for sid, fips, y, p in zip(test["target_sample_ids"], test["fips"], test["targets"], pred)]
    return ro.finish_test(output_dir, rows, test["target_sample_ids"], checkpoint_path=checkpoint_path,
                          years=(TARGET_TEST_YEAR,))


def main(args) -> Path:
    if args.eval_checkpoint:
        run_dir = Path(args.eval_checkpoint).resolve().parent
        evaluate_final_checkpoint(args.eval_checkpoint, args.dataset_jsonl, args.data_root, run_dir, device=args.device)
        return run_dir

    ro.seed_everything(args.seed)
    device = torch.device(args.device)
    run_dir = ro.make_run_dir(args.output_root, cfg.MODEL_NAME, args.run_id)
    args.run_id = run_dir.name
    ro.write_json(run_dir / "config.json", vars(args))

    windows = load_windows(args.dataset_jsonl, args.data_root)
    train = windows["train"]
    stats = fit_stats(train)
    inputs = {k: v.to(device) for k, v in model_inputs(train, stats).items()}
    y_std_target = ((train["targets"] - stats["y_mean"]) / stats["y_std"]).to(device)

    model_config = {k: getattr(args, k) for k in MODEL_KEYS}
    checkpoint_metadata = ro.checkpoint_metadata(
        model_name=cfg.MODEL_NAME,
        adaptation=cfg.ADAPTATION,
        dataset_jsonl=args.dataset_jsonl,
        train_years=TRAIN_YEARS,
        test_years=TEST_YEARS,
        model_config=model_config,
    )
    model = build_model(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T0, T_mult=args.T_mult,
                                                                     eta_min=args.eta_min)
    ro.write_json(run_dir / "run_metadata.json", ro.base_metadata(
        model=cfg.MODEL_NAME, adaptation=cfg.ADAPTATION, run_id=args.run_id, dataset_jsonl=args.dataset_jsonl,
        seed=args.seed, train_years=TRAIN_YEARS, test_years=TEST_YEARS, counties=len(train["fips"]),
        inputs=cfg.INPUTS, target_train_year=TARGET_TRAIN_YEAR, target_test_year=TARGET_TEST_YEAR,
        epochs=args.epochs, optimizer="Adam", lr=args.lr, weight_decay=args.weight_decay,
        scheduler=f"CosineAnnealingWarmRestarts(T0={args.T0}, T_mult={args.T_mult}, eta_min={args.eta_min})",
        aggregator=args.aggregator, main_aggregator=cfg.MAIN_AGGREGATOR,
        parameters=ro.count_parameters(model), edges=int(train["edge_index"].shape[1]),
        checkpoint_policy="final epoch only, no validation selection",
    ))

    generator = torch.Generator().manual_seed(args.seed)
    epoch = -1
    for epoch in range(args.epochs):
        row = train_one_epoch(model, inputs, y_std_target, optimizer, scheduler, epoch, args, generator)
        ro.append_jsonl(run_dir / "train_metrics.jsonl", {"epoch": epoch, **row})
        print(f"Epoch {epoch}: loss {row['train_loss']:.4f}  lr {row['lr']:.2e}")
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
        **checkpoint_metadata, "model": model.state_dict(), "epoch": epoch,
        "stats": stats_to_json(stats)})
    ro.update_json(
        run_dir / "run_metadata.json", epoch=epoch, final_checkpoint=str(final),
        final_checkpoint_sha256=ro.sha256_file(final), trained=ro.now(),
    )
    evaluate_final_checkpoint(final, args.dataset_jsonl, args.data_root, run_dir, device=args.device,
                              windows=windows)
    return run_dir


if __name__ == "__main__":
    main(build_parser().parse_args())
