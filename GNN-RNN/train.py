"""GNN-RNN 训练 + 最终测试（只评估 checkpoint-final.pth）。

默认协议：训练窗口 2017--2021 预测 2021，测试窗口 2018--2022 预测 2022。
统一协议（--valid-year 2021）：训练窗口 2017--2020 预测 2020，验证窗口 2018--2021 预测 2021
（早停选模，patience 3），测试窗口 2019--2022 预测 2022。
输出：output_root/gnn_rnn/<run_id>/{config.json, checkpoint-latest.pth, checkpoint-final.pth,
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
    TARGET_TEST_YEAR, TARGET_TRAIN_YEAR, TEST_YEARS, TRAIN_YEARS, fit_stats, load_windows, model_inputs,
    protocol_windows, stats_from_json, stats_to_json,
)
from model import GNNRNN
from train_dataset import run_output as ro
from train_dataset.dataset import GNN_WINDOWS, YEARS

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
    parser.add_argument("--fold-manifest", default="",
                        help="spatial CV: folds.json from train_dataset/make_spatial_folds.py")
    parser.add_argument("--county-fold", type=int, default=-1,
                        help="spatial CV: held-out fold index; loss and standardisation use the other counties")
    parser.add_argument("--rs-features", default="",
                        help="county-month vegetation-index npz (rs_signal_features_31states.npz); "
                             "enables the GNN-RNN+VI input-parity configuration")
    parser.add_argument("--valid-year", type=int, default=None,
                        help="hold-out validation year for early stopping (e.g. 2021); training window then ends before it")
    parser.add_argument("--early-stop-patience", type=int, default=10,
                        help="stop after N epochs without validation improvement (default 3); -1 disables early stopping")
    parser.add_argument("--checkpoint-policy", choices=("best", "final"), default="best",
                        help="best: restore the lowest-validation epoch; final: keep the last epoch (fixed budget, no selection)")
    return parser


def _run_windows(run_config: dict) -> dict:
    """按运行配置恢复窗口；未使用 --valid-year 的旧运行退回 GNN_WINDOWS。"""
    if run_config.get("valid_year") is not None:
        return protocol_windows(int(run_config["valid_year"]), int(run_config.get("test_year", YEARS[-1])))
    return GNN_WINDOWS


def logcosh(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """上游 loss_fn(mode="logcosh")，在标准化尺度上计算。"""
    return torch.mean(torch.log(torch.cosh(target - pred + 1e-12)))


def build_model(model_config: dict) -> GNNRNN:
    return GNNRNN(**{k: model_config[k] for k in MODEL_KEYS},
                  use_vi=bool(model_config.get("use_vi", False)))


def train_one_epoch(model, inputs, y_std_target, optimizer, scheduler, epoch, args, generator,
                    train_index=None) -> dict:
    model.train()
    pool = torch.arange(y_std_target.shape[0]) if train_index is None else train_index
    n = pool.numel()
    order = pool[torch.randperm(n, generator=generator)]
    steps = math.ceil(n / args.batch_size)
    total = 0.0
    for step in range(steps):
        idx = order[step * args.batch_size:(step + 1) * args.batch_size].to(y_std_target.device)
        # 全图前向；损失只作用于本步目标县的最后一年（上游 c1=0, c2=1）
        pred = model(**inputs)[:, -1]
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


def std_target(batch, stats, device):
    return ((batch["targets"] - stats["y_mean"]) / stats["y_std"]).to(device)


@torch.no_grad()
def evaluate_loss(model, batch, stats, device) -> float:
    """验证窗口上的逐县 MSE（标准化目标，与训练目标同口径）。"""
    was_training = model.training
    model.eval()
    inputs = {k: v.to(device) for k, v in model_inputs(batch, stats).items()}
    pred = model(**inputs)[:, -1]
    loss = torch.nn.functional.mse_loss(pred, std_target(batch, stats, device)).item()
    model.train(was_training)
    return loss


@torch.no_grad()
def predict(model, batch, stats, device) -> torch.Tensor:
    """返回末年预测（bu/ac），按窗口节点（fips）顺序。"""
    model.eval()
    inputs = {k: v.to(device) for k, v in model_inputs(batch, stats).items()}
    pred_std = model(**inputs)[:, -1].cpu()
    return pred_std * stats["y_std"] + stats["y_mean"]


def evaluate_final_checkpoint(checkpoint_path, dataset_jsonl, data_root, output_dir, *, device="cuda",
                              windows=None, test_index=None) -> dict:
    """只加载 checkpoint-final.pth，用运行配置记录的测试窗口预测测试年。

    ``test_index`` 用于空间交叉验证：只保留留出县的预测行。
    训练/测试窗口优先读 run_dir/config.json（支持 --valid-year 的 2017--2020 训练协议），
    旧运行没有该字段时退回 GNN_WINDOWS。
    """
    run_dir = Path(output_dir).resolve()
    config_path = run_dir / "config.json"
    run_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    window_spec = _run_windows(run_config)
    train_years, test_years = window_spec["train"], window_spec["test"]
    checkpoint, _ = ro.validate_unified_run(
        checkpoint_path,
        output_dir,
        model_name=cfg.MODEL_NAME,
        adaptation=cfg.ADAPTATION,
        dataset_jsonl=dataset_jsonl,
        train_years=train_years,
        test_years=test_years,
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
    rs_features = run_config.get("rs_features") or None
    test = (windows or load_windows(dataset_jsonl, data_root, windows=window_spec,
                                    rs_features=rs_features))["test"]
    if test["years"].tolist() != list(test_years):
        raise ValueError(f"test window years {test['years'].tolist()} != {list(test_years)}")
    target_test_year = int(test_years[-1])
    pred = predict(model, test, stats, device)
    keep = list(range(len(test["fips"]))) if test_index is None else test_index.tolist()
    rows = [{"sample_id": test["target_sample_ids"][i], "fips": test["fips"][i], "year": target_test_year,
             "y_true": float(test["targets"][i]), "y_pred": float(pred[i])} for i in keep]
    expected = [test["target_sample_ids"][i] for i in keep]
    return ro.finish_test(output_dir, rows, expected, checkpoint_path=checkpoint_path,
                          years=(target_test_year,))


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

    windows = load_windows(args.dataset_jsonl, args.data_root, windows=_run_windows(vars(args)),
                           rs_features=args.rs_features or None)
    train = windows["train"]
    valid = windows.get("valid")
    target_train_year = int(train["years"][-1])
    target_test_year = int(windows["test"]["years"][-1])
    if args.fold_manifest:
        manifest = json.loads(Path(args.fold_manifest).read_text(encoding="utf-8"))
        heldout = set(manifest["folds"][args.county_fold]["counties"])
        train_index = torch.tensor([i for i, f in enumerate(train["fips"]) if f not in heldout])
        test_index = torch.tensor([i for i, f in enumerate(train["fips"]) if f in heldout])
    else:
        train_index = test_index = None
    stats = fit_stats(train, mask=train_index)
    inputs = {k: v.to(device) for k, v in model_inputs(train, stats).items()}
    y_std_target = std_target(train, stats, device)

    model_config = {k: getattr(args, k) for k in MODEL_KEYS}
    model_config["use_vi"] = bool(args.rs_features)
    checkpoint_metadata = ro.checkpoint_metadata(
        model_name=cfg.MODEL_NAME,
        adaptation=cfg.ADAPTATION,
        dataset_jsonl=args.dataset_jsonl,
        train_years=windows["train"]["years"].tolist(),
        test_years=windows["test"]["years"].tolist(),
        model_config=model_config,
    )
    model = build_model(model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T0, T_mult=args.T_mult,
                                                                     eta_min=args.eta_min)
    ro.write_json(run_dir / "run_metadata.json", ro.base_metadata(
        model=cfg.MODEL_NAME, adaptation=cfg.ADAPTATION, run_id=args.run_id, dataset_jsonl=args.dataset_jsonl,
        seed=args.seed, train_years=windows["train"]["years"].tolist(),
        test_years=windows["test"]["years"].tolist(), counties=len(train["fips"]),
        inputs=cfg.INPUTS + (("vegetation_indices",) if args.rs_features else ()),
        rs_features=args.rs_features or None,
        target_train_year=target_train_year, target_test_year=target_test_year,
        epochs=args.epochs, optimizer="Adam", lr=args.lr, weight_decay=args.weight_decay,
        scheduler=f"CosineAnnealingWarmRestarts(T0={args.T0}, T_mult={args.T_mult}, eta_min={args.eta_min})",
        aggregator=args.aggregator, main_aggregator=cfg.MAIN_AGGREGATOR,
        parameters=ro.count_parameters(model), edges=int(train["edge_index"].shape[1]),
        checkpoint_policy=("final epoch only, fixed budget" if args.checkpoint_policy == "final"
                           else ("best validation epoch, early stopping" if valid is not None
                                 else "final epoch only, no validation selection")),
        valid_years=None if valid is None else valid["years"].tolist(),
        valid_samples=None if valid is None else len(valid["fips"]),
        early_stop_patience=args.early_stop_patience if valid is not None else None,
    ))

    generator = torch.Generator().manual_seed(args.seed)
    best_state, best_epoch, best_valid, stale = None, None, math.inf, 0
    stopped_early = False
    epoch = -1
    for epoch in range(args.epochs):
        row = train_one_epoch(model, inputs, y_std_target, optimizer, scheduler, epoch, args, generator,
                              train_index=train_index)
        if valid is not None:
            valid_loss = evaluate_loss(model, valid, stats, device)
            row["valid_loss"] = valid_loss
            if valid_loss < best_valid:
                best_valid, best_epoch, stale = valid_loss, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        ro.append_jsonl(run_dir / "train_metrics.jsonl", {"epoch": epoch, **row})
        print(f"Epoch {epoch}: loss {row['train_loss']:.4f}  lr {row['lr']:.2e}"
              + (f"  valid {row['valid_loss']:.6f}" if valid is not None else ""))
        if epoch % args.save_freq == 0 or epoch + 1 == args.epochs:
            latest = ro.save_checkpoint(run_dir / ro.LATEST_CHECKPOINT, {
                **checkpoint_metadata,
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                "stats": stats_to_json(stats)})
            ro.update_json(
                run_dir / "run_metadata.json", epoch=epoch, latest_checkpoint=str(latest),
                latest_checkpoint_sha256=ro.sha256_file(latest),
            )
        if valid is not None and args.early_stop_patience >= 0 and stale >= args.early_stop_patience:
            stopped_early = True
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, valid loss {best_valid:.6f})")
            break

    if best_state is not None and args.checkpoint_policy == "best":
        model.load_state_dict(best_state)
        epoch = best_epoch
    final = ro.save_checkpoint(run_dir / ro.FINAL_CHECKPOINT, {
        **checkpoint_metadata, "model": model.state_dict(), "epoch": epoch,
        "stats": stats_to_json(stats)})
    ro.update_json(
        run_dir / "run_metadata.json", epoch=epoch, final_checkpoint=str(final),
        final_checkpoint_sha256=ro.sha256_file(final), trained=ro.now(),
        stopped_early=stopped_early,
        best_valid_loss=None if best_epoch is None else best_valid,
        best_valid_epoch=best_epoch,
    )
    evaluate_final_checkpoint(final, args.dataset_jsonl, args.data_root, run_dir, device=args.device,
                              windows=windows, test_index=test_index)
    return run_dir


if __name__ == "__main__":
    main(build_parser().parse_args())
