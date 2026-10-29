"""TFT 统一 CropNet 训练入口与最终 2022 评估调度。"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from TFT_model import dataset_adapter as adapter
from TFT_model.model_adapter import build_model
from TFT_model.pvt import DEFAULT_PVT_PRETRAINED_PATH
from train_dataset import run_output as ro

MODEL_NAME = "tft"
ADAPTATION = "TFT full multimodal CropNet adaptation"
INPUTS = ("weather", "soil", "remote_sensing", "grid_coordinates")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("TFT (unified CropNet, train 2017-2021 / test 2022)")
    ro.add_common_args(
        parser,
        epochs=20,
        lr=1e-3,
        seed=42,
        output_root=Path(__file__).resolve().parent / "output_dir",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_lstm_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument(
        "--use-constructed", dest="use_constructed",
        action=argparse.BooleanOptionalAction, default=True,
    )
    parser.add_argument(
        "--vit-pretrained", "--pvt_pretrained", dest="vit_pretrained", action="store_true", default=False,
        help="opt in to loading local PVT-Tiny weights; default is offline random initialization",
    )
    parser.add_argument("--pvt_pretrained_path", default=str(DEFAULT_PVT_PRETRAINED_PATH),
                        help="local PVT-Tiny checkpoint path")
    parser.add_argument(
        "--no-vit-pretrained", "--no-pvt_pretrained", dest="vit_pretrained", action="store_false",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--vit-freeze-backbone", dest="vit_freeze_backbone",
        action=argparse.BooleanOptionalAction, default=False,
    )
    parser.add_argument(
        "--remote-constant", action="store_true", default=False,
        help="replace PVT remote-sensing features with a trainable constant",
    )
    parser.add_argument(
        "--remote-constant-init", choices=("zeros", "pvt_mean"), default="zeros",
        help="initial value of the remote constant; pvt_mean uses the mean PVT token on a fixed train-year subset",
    )
    parser.add_argument("--pvt-lr-scale", type=float, default=1.0)
    parser.add_argument(
        "--test-year", type=int, default=2022,
        help="leave-one-year-out: this year is the test year, all other years train (default 2022)",
    )
    parser.add_argument(
        "--county-offset", action="store_true", default=False,
        help="predict log-yield as training-year county mean + network output (network learns the anomaly)",
    )
    parser.add_argument(
        "--demean-rs", action="store_true", default=False,
        help="centre numeric RS features on each grid-month's training-year mean before standardising",
    )
    parser.add_argument(
        "--grad-clip", type=float, default=0.0,
        help="clip the global gradient norm to this value (0 disables)",
    )
    parser.add_argument(
        "--rs-features", default="",
        help="path to rs_features.py npz; replaces the PVT image encoder with a per-feature linear map "
             "over numeric grid-month vegetation indices (no images are loaded)",
    )
    parser.add_argument(
        "--rs-as-variable", action="store_true", default=False,
        help="feed remote-sensing tokens to the grid VSN as an extra variable (content path)",
    )
    parser.add_argument(
        "--rs-aux-weight", type=float, default=0.0,
        help="weight of an auxiliary log-yield loss predicted from pooled remote-sensing tokens only",
    )
    parser.add_argument(
        "--monitor-pvt-every", type=int, default=0,
        help="log PVT gradient/drift/collapse stats every N optimizer steps (0 disables)",
    )
    parser.add_argument(
        "--monitor-probe-samples", type=int, default=32,
        help="train-year samples used by the per-epoch shuffle-sensitivity probe",
    )
    parser.add_argument(
        "--eval_checkpoint", default="",
        help="only run the final 2022 test on this checkpoint-final.pth",
    )
    return parser


def _learning_rate(progress: float, args) -> float:
    if args.warmup_epochs > 0 and progress < args.warmup_epochs:
        return args.lr * progress / args.warmup_epochs
    cosine_progress = (progress - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
    return 0.5 * args.lr * (1.0 + math.cos(math.pi * min(max(cosine_progress, 0.0), 1.0)))


def _optimizer(model, args):
    groups = {key: [] for key in ("main_decay", "main_no_decay", "pvt_decay", "pvt_no_decay")}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        part = "pvt" if name.startswith("vit_encoder.backbone.") else "main"
        decay = parameter.ndim > 1 and not name.endswith(".bias") and "norm" not in name.lower()
        groups[f"{part}_{'decay' if decay else 'no_decay'}"].append(parameter)
    return torch.optim.AdamW(
        [
            {"name": key, "params": parameters,
             "lr": args.lr * (args.pvt_lr_scale if key.startswith("pvt_") else 1.0),
             "weight_decay": args.weight_decay if key.endswith("_decay") and not key.endswith("_no_decay") else 0.0}
            for key, parameters in groups.items() if parameters
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
    )


def train_one_epoch(model, loader, optimizer, device, epoch, args, monitor=None) -> dict:
    model.train()
    total = 0.0
    aux_total = 0.0
    seen = 0
    optimizer_steps = 0
    steps = len(loader)
    for step, batch in enumerate(loader):
        lr = _learning_rate(epoch + step / max(steps, 1), args)
        for group in optimizer.param_groups:
            group["lr"] = lr * (args.pvt_lr_scale if group["name"].startswith("pvt_") else 1.0)

        inputs = adapter.model_inputs(batch, device)
        if monitor is not None:
            monitor.before_forward()
        pred, _, aux = model(**inputs)
        valid = aux["sample_valid"]
        if not bool(valid.any()):
            continue
        labels = batch["labels"].to(device)
        target = torch.log(labels)
        if batch.get("county_offset") is not None:
            target = target - batch["county_offset"].to(device)
        loss = torch.nn.functional.mse_loss(pred[valid, 0], target[valid])
        objective = loss
        aux_loss = None
        if args.rs_aux_weight > 0:
            aux_loss = torch.nn.functional.mse_loss(aux["rs_aux_pred"][valid, 0], torch.log(labels[valid]))
            objective = loss + args.rs_aux_weight * aux_loss
        if not bool(torch.isfinite(objective)):
            raise FloatingPointError(f"epoch {epoch} step {step}: non-finite loss {objective.item()}")

        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        gradients = [
            parameter.grad for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if any(not bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise FloatingPointError(f"epoch {epoch} step {step}: non-finite gradient")
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.grad is not None], args.grad_clip,
            )
        if monitor is not None:
            monitor.after_backward(
                epoch=epoch, lr=lr, loss=loss.item(), aux=aux, inputs=inputs,
                aux_loss=None if aux_loss is None else aux_loss.item(),
            )
        optimizer.step()
        optimizer_steps += 1

        count = int(valid.sum().item())
        total += loss.item() * count
        if aux_loss is not None:
            aux_total += aux_loss.item() * count
        seen += count
        print(f"Epoch [{epoch}] [{step}/{steps - 1}] {batch['sample_id']} loss {loss.item():.4f}")
    if optimizer_steps == 0:
        raise RuntimeError(f"epoch {epoch}: no optimizer steps")
    row = {
        "train_loss": total / seen,
        "lr": optimizer.param_groups[0]["lr"],
        "optimizer_steps": optimizer_steps,
        "valid_samples": seen,
    }
    if args.rs_aux_weight > 0:
        row["rs_aux_loss"] = aux_total / seen
    return row


@torch.no_grad()
def _pvt_mean_token(model, dataset, stats, device, args, samples: int = 64) -> torch.Tensor:
    """固定训练年份子集上有效 PVT token 的均值：与坍缩后 PVT 同量级、但不含影像内容的常数。"""
    from TFT_model.pvt_monitor import build_probe_loader
    from TFT_model.rs_intervention import valid_token_mask

    loader = build_probe_loader(
        dataset, stats, samples=samples, seed=0,
        use_constructed=args.use_constructed, batch_size=args.batch_size,
    )
    was_training = model.training
    model.eval()
    total = torch.zeros(model.hidden_size, dtype=torch.float64, device=device)
    count = 0
    try:
        for batch in loader:
            inputs = adapter.model_inputs(batch, device)
            tokens = model._forward_remote_sensing(
                inputs["ag_images"], inputs["grid_coords"], inputs["grid_mask"], device, inputs["ag_grid_mask"],
            )
            valid = valid_token_mask(inputs["grid_mask"], inputs["ag_grid_mask"])
            total += tokens[valid].double().sum(dim=0)
            count += int(valid.sum())
    finally:
        model.train(was_training)
    if count == 0:
        raise RuntimeError("no valid remote-sensing tokens to initialize the remote constant")
    return (total / count).float()


def _checkpoint_payload(model, optimizer, epoch, args, stats, model_config) -> dict:
    train_years, test_years = adapter.split_years(args.test_year)
    return {
        **ro.checkpoint_metadata(
            model_name=MODEL_NAME,
            adaptation=ADAPTATION,
            dataset_jsonl=args.dataset_jsonl,
            train_years=train_years,
            test_years=test_years,
            model_config=model_config,
        ),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "stats": adapter.stats_to_json(stats),
        "epoch": epoch,
        "use_constructed": bool(args.use_constructed),
    }


def main(args) -> Path:
    from TFT_model.infer import evaluate_final_checkpoint

    if args.eval_checkpoint:
        run_dir = Path(args.eval_checkpoint).resolve().parent
        evaluate_final_checkpoint(
            args.eval_checkpoint, args.dataset_jsonl, args.data_root, run_dir,
            device=args.device, batch_size=args.batch_size, num_workers=args.num_workers,
        )
        return run_dir

    if args.remote_constant_init != "zeros" and not args.remote_constant:
        raise ValueError("--remote-constant-init requires --remote-constant")
    if args.rs_features and (args.remote_constant or args.rs_aux_weight > 0 or args.monitor_pvt_every > 0):
        raise ValueError("--rs-features has no PVT: it excludes --remote-constant, --rs-aux-weight and --monitor-pvt-every")
    if args.demean_rs and not args.rs_features:
        raise ValueError("--demean-rs requires --rs-features")
    train_years, test_years = adapter.split_years(args.test_year)

    ro.seed_everything(args.seed)
    device = torch.device(args.device)
    run_dir = ro.make_run_dir(args.output_root, MODEL_NAME, args.run_id)
    args.run_id = run_dir.name
    ro.write_json(run_dir / "config.json", vars(args))
    ro.write_json(run_dir / "run_metadata.json", ro.base_metadata(
        model=MODEL_NAME,
        adaptation=ADAPTATION,
        run_id=args.run_id,
        dataset_jsonl=args.dataset_jsonl,
        seed=args.seed,
        train_years=train_years,
        test_years=test_years,
        counties=0,
        inputs=INPUTS,
        train_samples=0,
        epochs=args.epochs,
        optimizer="AdamW",
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=f"linear warmup {args.warmup_epochs} epochs + cosine",
        target="log(yield_bu_ac) - county_mean" if args.county_offset else "log(yield_bu_ac)",
        checkpoint_policy="final epoch only, no validation selection",
        pvt_pretrained_path=str(Path(args.pvt_pretrained_path).expanduser().resolve()),
        pvt_pretrained_sha256=ro.sha256_file(args.pvt_pretrained_path) if args.vit_pretrained else None,
        remote_constant=bool(args.remote_constant),
    ))

    epoch = -1
    try:
        rs_store = adapter.RSFeatureStore(args.rs_features) if args.rs_features else None
        dataset = adapter.build_dataset(
            args.dataset_jsonl, args.data_root, train_years, load_images=rs_store is None,
        )
        stats = adapter.fit_stats(
            dataset, use_constructed=args.use_constructed, rs_store=rs_store, train_years=train_years,
            county_offset=args.county_offset, demean_rs=args.demean_rs,
        )
        loader = adapter.build_loader(
            dataset, stats, args.batch_size, True,
            use_constructed=args.use_constructed,
            num_workers=args.num_workers,
            generator=torch.Generator().manual_seed(args.seed),
            rs_store=rs_store,
        )
        model_config = {
            "hidden_size": args.hidden_size,
            "num_heads": args.num_heads,
            "num_lstm_layers": args.num_lstm_layers,
            "dropout": args.dropout,
            "dynamic_feature_names": adapter.dynamic_feature_names(args.use_constructed),
            "vit_pretrained": bool(args.vit_pretrained),
            "vit_freeze_backbone": bool(args.vit_freeze_backbone),
            "pvt_pretrained_path": str(Path(args.pvt_pretrained_path).expanduser().resolve()),
            "pvt_pretrained_sha256": ro.sha256_file(args.pvt_pretrained_path) if args.vit_pretrained else None,
            "remote_constant": bool(args.remote_constant),
            "rs_as_variable": bool(args.rs_as_variable),
            "rs_aux": args.rs_aux_weight > 0,
            "county_offset": bool(args.county_offset),
            "demean_rs": bool(args.demean_rs),
            "test_year": int(args.test_year),
        }
        if rs_store is not None:
            model_config.update(
                rs_feature_dim=len(rs_store.names),
                rs_feature_names=list(rs_store.names),
                rs_features_path=str(rs_store.path.resolve()),
                rs_features_sha256=ro.sha256_file(rs_store.path),
                vit_pretrained=False,
            )
        model = build_model(model_config).to(device)
        if args.remote_constant_init == "pvt_mean":
            init = _pvt_mean_token(model, dataset, stats, device, args)
            model.remote_constant.data.copy_(init)
            ro.update_json(
                run_dir / "run_metadata.json",
                remote_constant_init="pvt_mean",
                remote_constant_init_norm=float(init.norm()),
            )
        optimizer = _optimizer(model, args)
        ro.update_json(
            run_dir / "run_metadata.json",
            counties=len({sample["fips"] for sample in dataset.samples}),
            train_samples=len(dataset),
            parameters=ro.count_parameters(model),
        )
        monitor = None
        if args.monitor_pvt_every > 0:
            from TFT_model.pvt_monitor import PVTMonitor, build_probe_loader

            probe_loader = build_probe_loader(
                dataset, stats, samples=args.monitor_probe_samples, seed=args.seed,
                use_constructed=args.use_constructed, batch_size=args.batch_size,
            ) if args.monitor_probe_samples > 0 else None
            monitor = PVTMonitor(
                model, run_dir, every=args.monitor_pvt_every, probe_loader=probe_loader, device=device,
            )
            monitor.probe(-1)

        for epoch in range(args.epochs):
            row = train_one_epoch(model, loader, optimizer, device, epoch, args, monitor)
            ro.append_jsonl(run_dir / "train_metrics.jsonl", {"epoch": epoch, **row})
            if monitor is not None:
                monitor.probe(epoch)
            if epoch % args.save_freq == 0 or epoch + 1 == args.epochs:
                latest = ro.save_checkpoint(
                    run_dir / ro.LATEST_CHECKPOINT,
                    _checkpoint_payload(model, optimizer, epoch, args, stats, model_config),
                )
                ro.update_json(
                    run_dir / "run_metadata.json",
                    epoch=epoch,
                    latest_checkpoint=str(latest),
                    latest_checkpoint_sha256=ro.sha256_file(latest),
                )

        if monitor is not None:
            monitor.close()
        final = ro.save_checkpoint(
            run_dir / ro.FINAL_CHECKPOINT,
            _checkpoint_payload(model, optimizer, epoch, args, stats, model_config),
        )
        ro.update_json(
            run_dir / "run_metadata.json",
            epoch=epoch,
            final_checkpoint=str(final),
            final_checkpoint_sha256=ro.sha256_file(final),
            trained=ro.now(),
        )
        evaluate_final_checkpoint(
            final, args.dataset_jsonl, args.data_root, run_dir,
            device=args.device, batch_size=args.batch_size, num_workers=args.num_workers,
        )
    except Exception as error:
        ro.update_json(
            run_dir / "run_metadata.json",
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            failed=ro.now(),
        )
        raise
    return run_dir


if __name__ == "__main__":
    main(build_parser().parse_args())
