"""PhenoYieldNet-unified Stage 2：2021 监督微调（PhenoYieldNet-SC，只做 corn）+ 2022 最终测试。

与上游 train_ft.py / scripts/run_finetune.sh 一致：骨干 num_classes=2 后加载 Stage 1 checkpoint，
ViT_multimodal_multicrop 端到端训练（上游脚本不传 --frozen_fm，公开代码实际不冻结编码器），
MSE 作用于 log(yield)（--yield_only_loss 语义），AdamW(0.9,0.95)，每步一个县年。
只用 corn 的 crop query 与回归头；不使用长期气象。固定 epoch，无验证集、无按指标选 checkpoint。
输出：output_root/phenoyieldnet/<run_id>/{config.json, checkpoint-latest.pth, checkpoint-final.pth,
train_metrics.jsonl, test_metrics.json, county_predictions.jsonl, run_metadata.json}。
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
    parser.add_argument("--frozen_fm", action="store_true",
                        help="freeze the backbone (paper description); upstream run_finetune.sh does not")
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--eval_checkpoint", default="", help="only run the final 2022 test on this checkpoint-final.pth")
    return parser


def model_config(args) -> dict:
    return {"batch_size": args.batch_size, "grad_checkpointing": args.grad_checkpointing,
            "frozen_fm": args.frozen_fm, "drop_path": args.drop_path}


def build_full_model(config: dict, stage1_state: dict | None = None):
    net = build_backbone(config["drop_path"], num_classes=2)  # 上游 Stage 2 num_classes=2
    report = load_pretrained_state(net, stage1_state) if stage1_state is not None else None
    return build_model(config, net), report


def checkpoint_payload(model, optimizer, epoch, args, config) -> dict:
    return {
        **ro.checkpoint_metadata(
            model_name=getattr(cfg, "MODEL_NAME", "phenoyieldnet"),
            adaptation=getattr(
                cfg,
                "ADAPTATION",
                "PhenoYieldNet-unified (PhenoYieldNet-SC, corn only, unified CropNet)",
            ),
            dataset_jsonl=args.dataset_jsonl,
            train_years=cfg.FINETUNE_YEARS,
            test_years=cfg.TEST_YEARS,
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
                               num_workers=0) -> dict:
    """只加载 checkpoint-final.pth，只读 2022。"""
    checkpoint, _ = ro.validate_unified_run(
        checkpoint_path,
        output_dir,
        model_name=cfg.MODEL_NAME,
        adaptation=cfg.ADAPTATION,
        dataset_jsonl=dataset_jsonl,
        train_years=cfg.FINETUNE_YEARS,
        test_years=cfg.TEST_YEARS,
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
    dataset = build_dataset(dataset_jsonl, data_root, cfg.TEST_YEARS)
    rows = predict(model, build_loader(dataset, False, num_workers=num_workers), device)
    return ro.finish_test(output_dir, rows, dataset.sample_ids, checkpoint_path=checkpoint_path, years=cfg.TEST_YEARS)


def main(args) -> Path:
    if args.eval_checkpoint:
        run_dir = Path(args.eval_checkpoint).resolve().parent
        evaluate_final_checkpoint(args.eval_checkpoint, args.dataset_jsonl, args.data_root, run_dir,
                                  device=args.device, num_workers=args.num_workers)
        return run_dir

    if not args.pretrain_model:
        raise ValueError("--pretrain_model requires Stage 1 checkpoint-final.pth")
    stage1_checkpoint = ro.validate_final_checkpoint(
        args.pretrain_model,
        model_name=STAGE1_MODEL_NAME,
        adaptation=STAGE1_ADAPTATION,
        dataset_jsonl=args.dataset_jsonl,
        train_years=cfg.PRETRAIN_YEARS,
        test_years=cfg.TEST_YEARS,
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

    dataset = build_dataset(args.dataset_jsonl, args.data_root, cfg.FINETUNE_YEARS)
    loader = build_loader(dataset, True, num_workers=args.num_workers,
                          generator=torch.Generator().manual_seed(args.seed))
    optimizer = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr, betas=(0.9, 0.95))
    ro.write_json(run_dir / "run_metadata.json", ro.base_metadata(
        model=cfg.MODEL_NAME, adaptation=cfg.ADAPTATION, run_id=args.run_id, dataset_jsonl=args.dataset_jsonl,
        seed=args.seed, train_years=cfg.FINETUNE_YEARS, test_years=cfg.TEST_YEARS,
        counties=len({s["fips"] for s in dataset.samples}), inputs=cfg.INPUTS, train_samples=len(dataset),
        pretrain_years=list(cfg.PRETRAIN_YEARS), stage1_checkpoint=args.pretrain_model or None,
        epochs=args.epochs, optimizer="AdamW(0.9,0.95)", lr=args.lr, blr=args.blr, weight_decay=args.weight_decay,
        frozen_backbone=args.frozen_fm, crop="corn", target="log(yield_bu_ac)",
        parameters=ro.count_parameters(model), checkpoint_policy="final epoch only, no validation selection",
    ))

    epoch = -1
    for epoch in range(args.epochs):
        row = train_one_epoch(model, loader, optimizer, device, epoch, args)
        ro.append_jsonl(run_dir / "train_metrics.jsonl", {"epoch": epoch, **row})
        if epoch % args.save_freq == 0 or epoch + 1 == args.epochs:
            latest = ro.save_checkpoint(
                run_dir / ro.LATEST_CHECKPOINT,
                checkpoint_payload(model, optimizer, epoch, args, config),
            )
            ro.update_json(
                run_dir / "run_metadata.json", epoch=epoch, latest_checkpoint=str(latest),
                latest_checkpoint_sha256=ro.sha256_file(latest),
            )
    final = ro.save_checkpoint(
        run_dir / ro.FINAL_CHECKPOINT,
        checkpoint_payload(model, optimizer, epoch, args, config),
    )
    ro.update_json(
        run_dir / "run_metadata.json", epoch=epoch, final_checkpoint=str(final),
        final_checkpoint_sha256=ro.sha256_file(final), trained=ro.now(),
    )
    evaluate_final_checkpoint(final, args.dataset_jsonl, args.data_root, run_dir, device=args.device,
                              num_workers=args.num_workers)
    return run_dir


if __name__ == "__main__":
    main(build_parser().parse_args())
