"""PhenoYieldNet-unified Stage 2：单年监督微调（PhenoYieldNet-SC，只做 corn）+ 2022 最终测试。

与上游 train_ft.py / scripts/run_finetune.sh 一致：骨干 num_classes=2 后加载 Stage 1 checkpoint，
ViT_multimodal_multicrop 端到端训练（上游脚本不传 --frozen_fm，公开代码实际不冻结编码器），
MSE 作用于 log(yield)（--yield_only_loss 语义），AdamW(0.9,0.95)，每步一个县年。
只用 corn 的 crop query 与回归头；不使用长期气象。
默认协议：2021 微调、2022 测试。统一协议（--valid-year 2021）：2020 微调、2021 早停选模、2022 测试。
输出：output_root/phenoyieldnet/<run_id>/{config.json, checkpoint-latest.pth, checkpoint-final.pth,
train_metrics.jsonl, test_metrics.json, county_predictions.jsonl, run_metadata.json}。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from einops import rearrange

import config as cfg
from backbone import build_backbone, build_model, load_pretrained_state
from common import add_args, adjust_learning_rate, param_groups, per_image, resolve_lr, stage2_transform
from dataset_adapter import build_dataset, build_loader, require_unpadded
from train_dataset import run_output as ro

STAGE1_MODEL_NAME = "phenoyieldnet_stage1"
STAGE1_ADAPTATION = "PhenoYieldNet-unified contrastive adaptation"


def build_parser():
    parser = argparse.ArgumentParser("PhenoYieldNet-unified Stage 2 (finetune 2021 / final test 2022)")
    add_args(parser, cfg.STAGE2)
    parser.add_argument("--pretrain_model", default="", help="Stage 1 checkpoint-final.pth")
    parser.add_argument("--frozen_fm", action=argparse.BooleanOptionalAction, default=True,
                        help="冻结 Stage 1 骨干，只训解码器与回归头（论文 4.1.4；上游脚本不传，故默认按论文）")
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--eval_checkpoint", default="", help="only run the final 2022 test on this checkpoint-final.pth")
    parser.add_argument("--eval_jsonl", default="",
                        help="spatial CV: JSONL holding only the held-out counties for the final test")
    parser.add_argument("--valid-year", type=int, default=None,
                        help="hold-out validation year for early stopping (e.g. 2021); finetuning then uses only the year before it")
    parser.add_argument("--early-stop-patience", type=int, default=10,
                        help="stop after N epochs without validation improvement (default 3); -1 disables early stopping")
    parser.add_argument("--checkpoint-policy", choices=("best", "final"), default="best",
                        help="best: restore the lowest-validation epoch; final: keep the last epoch (fixed budget, no selection)")
    parser.add_argument("--stage1-own-years", action="store_true",
                        help="validate the Stage 1 checkpoint against the years recorded inside it "
                             "(spatial CV trains on all 2017-2022, which differs from cfg.PRETRAIN_YEARS)")
    parser.add_argument("--valid-county-frac", type=float, default=0.0,
                        help="按县划分验证集的比例，用于复现原论文的早停（0 = 不用；与 --valid-year 互斥）。"
                             "划分只取微调年内的县，不涉及测试年，也不涉及测试县。")
    return parser


def _split(args):
    """返回 (微调年, 验证年或None, 测试年)。微调沿用单年设计，取训练期中最后一年。"""
    test_year = tuple(cfg.TEST_YEARS)[-1]
    if args.valid_year is not None:
        valid_year = int(args.valid_year)
        if not (min(cfg.PRETRAIN_YEARS) < valid_year < test_year):
            raise ValueError(f"valid year {valid_year} must lie strictly between {min(cfg.PRETRAIN_YEARS)} and {test_year}")
        train_years = tuple(y for y in (*cfg.PRETRAIN_YEARS, *cfg.FINETUNE_YEARS) if y < valid_year)
        return (train_years[-1],), (valid_year,), (test_year,)
    return tuple(cfg.FINETUNE_YEARS), None, tuple(cfg.TEST_YEARS)


def _stage1_years(path):
    """Stage 1 checkpoint 自带 train/test years；空间 CV 微调年份与它不同时按自身记录校验。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return tuple(checkpoint["train_years"]), tuple(checkpoint["test_years"])


def model_config(args) -> dict:
    return {"batch_size": args.batch_size, "grad_checkpointing": args.grad_checkpointing,
            "frozen_fm": args.frozen_fm, "drop_path": args.drop_path}


def build_full_model(config: dict, stage1_state: dict | None = None):
    net = build_backbone(config["drop_path"], num_classes=2)  # 上游 Stage 2 num_classes=2
    report = load_pretrained_state(net, stage1_state) if stage1_state is not None else None
    return build_model(config, net), report


def checkpoint_payload(model, optimizer, epoch, args, config, train_years, test_years) -> dict:
    return {
        **ro.checkpoint_metadata(
            model_name=getattr(cfg, "MODEL_NAME", "phenoyieldnet"),
            adaptation=getattr(
                cfg,
                "ADAPTATION",
                "PhenoYieldNet-unified (PhenoYieldNet-SC, corn only, unified CropNet)",
            ),
            dataset_jsonl=args.dataset_jsonl,
            train_years=train_years,
            test_years=test_years,
            model_config=config,
        ),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }


def model_inputs(batch, device, transform):
    """图像 (b t g) 逐批增强（上游 DataWrapper 对整批同一组参数），天气上下文 9 维。"""
    require_unpadded(batch)
    b, t, g = batch["images"].shape[:3]
    x = rearrange(batch["images"], 'b t g c h w -> (b t g) c h w')
    x = rearrange(transform(x), '(b t g) c h w -> b t g c h w', b=b, t=t, g=g)
    crop_idx = torch.full((1,), cfg.CORN, dtype=torch.long, device=device)
    return x.to(device, non_blocking=True), batch["weather_context"].to(device, non_blocking=True), crop_idx


def train_one_epoch(model, loader, optimizer, device, epoch, args) -> dict:
    model.train()
    transform = stage2_transform(train=True)
    total, seen = 0.0, 0
    steps = len(loader)
    for step, batch in enumerate(loader):
        adjust_learning_rate(optimizer, step / steps + epoch, args)
        x, ys, crop_idx = model_inputs(batch, device, transform)
        z = torch.log(batch["labels"]).unsqueeze(-1).to(device)
        loss = torch.nn.functional.mse_loss(model(x, ys=ys, crop_idx=crop_idx), z)
        if not math.isfinite(loss.item()):
            raise FloatingPointError(f"{batch['sample_id']}: loss {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item()
        seen += 1
        print(f"Epoch [{epoch}] [{step}/{steps - 1}] {batch['sample_id'][0]} loss {loss.item():.4f}")
    return {"train_loss": total / max(seen, 1), "lr": optimizer.param_groups[0]["lr"]}


@torch.no_grad()
def evaluate_loss(model, loader, device) -> float:
    """验证年上的逐县 MSE(log yield)，与微调损失同口径。"""
    was_training = model.training
    model.eval()
    transform = stage2_transform(train=False)
    total, seen = 0.0, 0
    for batch in loader:
        x, ys, crop_idx = model_inputs(batch, device, transform)
        z = torch.log(batch["labels"]).unsqueeze(-1).to(device)
        loss = torch.nn.functional.mse_loss(model(x, ys=ys, crop_idx=crop_idx), z, reduction="sum").item()
        total += loss
        seen += z.numel()
    model.train(was_training)
    if seen == 0:
        raise RuntimeError("validation loader has no samples")
    return total / seen


@torch.no_grad()
def predict(model, loader, device) -> list[dict]:
    model.eval()
    transform = stage2_transform(train=False)
    rows = []
    for batch in loader:
        x, ys, crop_idx = model_inputs(batch, device, transform)
        pred = torch.exp(model(x, ys=ys, crop_idx=crop_idx)[:, -1].float()).cpu()
        for sid, fips, year, y, p in zip(batch["sample_id"], batch["fips"], batch["year"].tolist(),
                                         batch["labels"].tolist(), pred.tolist()):
            rows.append({"sample_id": sid, "fips": fips, "year": year, "y_true": y, "y_pred": p})
    return rows


def evaluate_final_checkpoint(checkpoint_path, dataset_jsonl, data_root, output_dir, *, device="cuda",
                               num_workers=0, eval_jsonl=None) -> dict:
    """只加载 checkpoint-final.pth，只读运行配置记录的测试年份。

    ``eval_jsonl`` 用于空间交叉验证：校验仍对照训练数据，测试样本改由留出县文件给出。
    微调/测试年份优先读 run_dir/config.json（支持 --valid-year 的统一协议），旧运行退回 cfg 默认值。
    """
    run_dir = Path(output_dir).resolve()
    config_path = run_dir / "config.json"
    run_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    if run_config.get("valid_year") is not None:
        finetune_years, _, test_years = _split(argparse.Namespace(valid_year=run_config["valid_year"]))
    else:
        finetune_years, test_years = tuple(cfg.FINETUNE_YEARS), tuple(cfg.TEST_YEARS)
    checkpoint, _ = ro.validate_unified_run(
        checkpoint_path,
        output_dir,
        model_name=cfg.MODEL_NAME,
        adaptation=cfg.ADAPTATION,
        dataset_jsonl=dataset_jsonl,
        train_years=finetune_years,
        test_years=test_years,
        required_fields=(
            "model_config.batch_size",
            "model_config.grad_checkpointing",
            "model_config.frozen_fm",
            "model_config.drop_path",
        ),
    )
    device = torch.device(device)
    model, _ = build_full_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    dataset = build_dataset(eval_jsonl or dataset_jsonl, data_root, test_years)
    rows = predict(model, build_loader(dataset, False, num_workers=num_workers), device)
    return ro.finish_test(output_dir, rows, dataset.sample_ids, checkpoint_path=checkpoint_path, years=test_years)


def main(args) -> Path:
    if args.eval_checkpoint:
        run_dir = Path(args.eval_checkpoint).resolve().parent
        evaluate_final_checkpoint(args.eval_checkpoint, args.dataset_jsonl, args.data_root, run_dir,
                                  device=args.device, num_workers=args.num_workers,
                                  eval_jsonl=args.eval_jsonl or None)
        return run_dir

    if not args.pretrain_model:
        raise ValueError("--pretrain_model requires Stage 1 checkpoint-final.pth")
    stage1_train_years, stage1_test_years = (
        _stage1_years(args.pretrain_model) if args.stage1_own_years
        else (cfg.PRETRAIN_YEARS, cfg.TEST_YEARS))
    stage1_checkpoint = ro.validate_final_checkpoint(
        args.pretrain_model,
        model_name=STAGE1_MODEL_NAME,
        adaptation=STAGE1_ADAPTATION,
        dataset_jsonl=args.dataset_jsonl,
        train_years=stage1_train_years,
        test_years=stage1_test_years,
    )

    ro.seed_everything(args.seed)
    device = torch.device(args.device)
    run_dir = ro.make_run_dir(args.output_root, cfg.MODEL_NAME, args.run_id)
    args.run_id = run_dir.name
    resolve_lr(args)
    ro.write_json(run_dir / "config.json", vars(args))

    config = model_config(args)
    model, load_report = build_full_model(config, stage1_checkpoint["model"])
    if load_report is not None:
        load_report["path"] = str(args.pretrain_model)
        ro.write_json(run_dir / "stage1_load_report.json", load_report)
    model.to(device)

    finetune_years, valid_years, test_years = _split(args)
    dataset = build_dataset(args.dataset_jsonl, args.data_root, finetune_years)
    train_set, valid_set, valid_counties = dataset, None, None
    if args.valid_county_frac:
        if valid_years is not None:
            raise ValueError("--valid-county-frac 与 --valid-year 互斥")
        if not 0 < args.valid_county_frac < 1:
            raise ValueError("--valid-county-frac must be in (0, 1)")
        from torch.utils.data import Subset
        fips = sorted({s["fips"] for s in dataset.samples})
        order = np.random.default_rng(args.seed).permutation(len(fips))
        n_valid = max(1, int(round(len(fips) * args.valid_county_frac)))
        valid_counties = sorted(fips[i] for i in order[:n_valid])
        held = set(valid_counties)
        mask = np.array([s["fips"] in held for s in dataset.samples])
        train_set = Subset(dataset, np.flatnonzero(~mask).tolist())
        valid_set = Subset(dataset, np.flatnonzero(mask).tolist())
    loader = build_loader(train_set, True, num_workers=args.num_workers,
                          generator=torch.Generator().manual_seed(args.seed))
    valid_loader = None
    if valid_years is not None:
        valid_set = build_dataset(args.dataset_jsonl, args.data_root, valid_years)
        if not len(valid_set):
            raise ValueError(f"validation years {valid_years} have no samples")
    if valid_set is not None:
        valid_loader = build_loader(valid_set, False, num_workers=args.num_workers)
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95))
    n_counties = len({s["fips"] for s in dataset.samples})
    n_valid_counties = len(valid_counties) if valid_counties else 0
    ro.write_json(run_dir / "run_metadata.json", ro.base_metadata(
        model=cfg.MODEL_NAME, adaptation=cfg.ADAPTATION, run_id=args.run_id, dataset_jsonl=args.dataset_jsonl,
        seed=args.seed, train_years=finetune_years, test_years=test_years,
        counties=n_counties, train_counties=n_counties - n_valid_counties,
        inputs=cfg.INPUTS, train_samples=len(train_set),
        pretrain_years=list(cfg.PRETRAIN_YEARS), stage1_checkpoint=args.pretrain_model or None,
        epochs=args.epochs, optimizer="AdamW(0.9,0.95)", lr=args.lr, blr=args.blr, weight_decay=args.weight_decay,
        frozen_backbone=args.frozen_fm, crop="corn", target="log(yield_bu_ac)",
        parameters=ro.count_parameters(model),
        checkpoint_policy=("final epoch only, fixed budget" if args.checkpoint_policy == "final"
                           else ("best validation epoch, early stopping" if valid_loader is not None
                                 else "final epoch only, no validation selection")),
        valid_years=list(valid_years) if valid_years is not None else None,
        valid_samples=len(valid_set) if valid_set is not None else None,
        valid_counties=list(valid_counties) if valid_counties else None,
        early_stop_patience=args.early_stop_patience if valid_loader is not None else None,
    ))

    best_state, best_epoch, best_valid, stale = None, None, math.inf, 0
    stopped_early = False
    epoch = -1
    for epoch in range(args.epochs):
        row = train_one_epoch(model, loader, optimizer, device, epoch, args)
        if valid_loader is not None:
            valid_loss = evaluate_loss(model, valid_loader, device)
            row["valid_loss"] = valid_loss
            if valid_loss < best_valid:
                best_valid, best_epoch, stale = valid_loss, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
        ro.append_jsonl(run_dir / "train_metrics.jsonl", {"epoch": epoch, **row})
        if epoch % args.save_freq == 0 or epoch + 1 == args.epochs:
            latest = ro.save_checkpoint(
                run_dir / ro.LATEST_CHECKPOINT,
                checkpoint_payload(model, optimizer, epoch, args, config, finetune_years, test_years),
            )
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
    final = ro.save_checkpoint(
        run_dir / ro.FINAL_CHECKPOINT,
        checkpoint_payload(model, optimizer, epoch, args, config, finetune_years, test_years),
    )
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
