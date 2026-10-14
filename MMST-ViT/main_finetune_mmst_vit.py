"""MMST-ViT 监督微调（统一 CropNet，只用 2021）与最终测试（只用 2022 + checkpoint-final.pth）。

没有验证集，不按任何指标选择 checkpoint；训练结束无条件写 checkpoint-final.pth，再对其做一次 2022 测试。
2022 标签只进入 test_metrics.json 与 county_predictions.jsonl。
输出：output_root/mmst_vit/<run_id>/{config.json, checkpoint-latest.pth, checkpoint-final.pth,
train_metrics.jsonl, test_metrics.json, county_predictions.jsonl, run_metadata.json}。
"""

import argparse
import datetime
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter

import timm.optim.optim_factory as optim_factory

import util.lr_sched as lr_sched
import util.misc as misc
from dataset.data_wrapper import DataWrapper
from dataset_adapter import MMST_YEARS_FINETUNE, MMST_YEARS_TEST, build_mmst_dataloader, build_mmst_dataset
from models_mmst_vit import MMST_ViT
from models_pvt_simclr import PVTSimCLR
from train_dataset.metrics import regression_metrics
from util import run_io
from util.misc import NativeScalerWithGradNormCount as NativeScaler

FINETUNE_YEARS = MMST_YEARS_FINETUNE
TEST_YEARS = MMST_YEARS_TEST
MODEL_NAME = "mmst_vit"


def build_parser():
    parser = argparse.ArgumentParser('MMST-ViT fine-tuning (unified CropNet, train 2021 / test 2022)')
    run_io.add_common_args(parser, model_name=MODEL_NAME)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='grid-month crops per PVT forward chunk (memory bound, not the county batch)')
    parser.add_argument('--county_batch_size', default=1, type=int, help='county-years per optimizer step')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N', help='epochs to warmup LR')
    parser.add_argument('--depth', default=4, type=int, help='MMST spatial / temporal transformer depth')
    parser.add_argument('--heads', default=3, type=int, help='MMST spatial / temporal transformer heads')
    parser.add_argument('--dim_head', default=64, type=int)
    parser.add_argument('--pvt_simclr', default='', help='pretrained PVT-SimCLR checkpoint-final.pth')
    parser.add_argument('--eval_checkpoint', default='',
                        help='only run the final 2022 test on this checkpoint-final.pth')
    return parser


def model_config(args):
    keys = ("model_pvt", "embed_dim", "num_heads", "mm_depth", "depth", "heads", "dim_head", "batch_size")
    return {k: getattr(args, k) for k in keys}


def build_model(config, pretrained_pvt=False):
    pvt = PVTSimCLR(config["model_pvt"], out_dim=config["embed_dim"], context_dim=9,
                    num_head=config["num_heads"], mm_depth=config["mm_depth"], pretrained=pretrained_pvt)
    return MMST_ViT(out_dim=1, pvt_backbone=pvt, context_dim=9, dim=config["embed_dim"],
                    batch_size=config["batch_size"], depth=config["depth"], heads=config["heads"],
                    dim_head=config["dim_head"])


def capture_evaluation_years(checkpoint_path):
    """最终测试协议：只接受 checkpoint-final.pth，只评估 2022。"""
    if Path(checkpoint_path).name != run_io.FINAL_CHECKPOINT:
        raise ValueError(f"final test only evaluates {run_io.FINAL_CHECKPOINT}, got {checkpoint_path}")
    return TEST_YEARS


def model_inputs(batch, device, augment=None):
    """MMST batch -> 模型输入；padding 网格图像保持 0，增强只作用于真实 (月, 网格)。"""
    x = batch["images"]
    if augment is not None:
        b, t, g = x.shape[:3]
        valid = rearrange(batch["image_mask"] & batch["grid_mask"][:, None], 'b t g -> (b t g)')
        flat = rearrange(x, 'b t g c h w -> (b t g) c h w').clone()
        flat[valid] = augment(flat[valid])[0]
        x = rearrange(flat, '(b t g) c h w -> b t g c h w', b=b, t=t, g=g)
    return (x.to(device, non_blocking=True),
            batch["weather_context"].to(device, non_blocking=True),
            batch["long_term"].to(device, non_blocking=True),
            batch["grid_mask"].to(device, non_blocking=True))


def main(args):
    if args.eval_checkpoint:
        run_dir = Path(args.eval_checkpoint).resolve().parent
        return evaluate_final_checkpoint(args.eval_checkpoint, args.dataset_jsonl, args.data_root, run_dir,
                                         device=args.device, num_workers=args.num_workers)

    misc.init_distributed_mode(args)
    run_dir = run_io.prepare_run(args)
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset = build_mmst_dataset(args.dataset_jsonl, args.data_root, FINETUNE_YEARS, load_images=True)
    sampler = None
    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=misc.get_world_size(),
                                                      rank=misc.get_rank(), shuffle=True)
    data_loader = build_mmst_dataloader(None, None, FINETUNE_YEARS, batch_size=args.county_batch_size,
                                        shuffle=sampler is None, sampler=sampler, num_workers=args.num_workers,
                                        pin_memory=args.pin_mem, dataset=dataset)

    log_writer = SummaryWriter(log_dir=str(run_dir / "tensorboard")) if misc.is_main_process() else None

    config = model_config(args)
    model = build_model(config, pretrained_pvt=not args.no_pvt_pretrained and not args.pvt_simclr)
    if args.pvt_simclr:
        # 只加载 PVT-SimCLR 主干；MMST 的 80 网格位置编码等参数保持新初始化
        checkpoint = torch.load(args.pvt_simclr, map_location='cpu', weights_only=False)
        model.pvt_backbone.load_state_dict(checkpoint['model'])
        print("Loaded PVT-SimCLR from %s" % args.pvt_simclr)
    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.county_batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("actual lr: %.2e, effective county batch size: %d" % (args.lr, eff_batch_size))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.999))
    loss_scaler = NativeScaler()
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    if misc.is_main_process():
        run_io.write_json(run_dir / "run_metadata.json", {
            "model": MODEL_NAME, "run_id": args.run_id, "dataset_jsonl": args.dataset_jsonl,
            "data_root": args.data_root, "train_years": list(FINETUNE_YEARS), "train_samples": len(dataset),
            "test_years": list(TEST_YEARS), "pvt_simclr": args.pvt_simclr,
            "target": "log(yield_bu_ac)", "checkpoint_policy": "final epoch only, no validation selection",
            "world_size": misc.get_world_size(),
            "started": datetime.datetime.now().isoformat(timespec="seconds"), "status": "running",
        })

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    epoch = args.start_epoch - 1
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(model, data_loader, optimizer, device, epoch, loss_scaler,
                                      log_writer=log_writer, args=args)

        if epoch % args.save_freq == 0 or epoch + 1 == args.epochs:
            misc.save_model(args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                            loss_scaler=loss_scaler, epoch=epoch)
        if misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            run_io.append_jsonl(run_dir / "train_metrics.jsonl",
                                {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}})
    print('Training time {}'.format(datetime.timedelta(seconds=int(time.time() - start_time))))

    if not misc.is_main_process():
        return run_dir
    final = run_io.save_final_checkpoint(run_dir, model_without_ddp, epoch=epoch, model_config=config,
                                         train_years=FINETUNE_YEARS)
    run_io.update_json(run_dir / "run_metadata.json", final_checkpoint=str(final),
                       trained=datetime.datetime.now().isoformat(timespec="seconds"))
    if log_writer is not None:
        log_writer.close()
    return evaluate_final_checkpoint(final, args.dataset_jsonl, args.data_root, run_dir,
                                     device=args.device, num_workers=args.num_workers)


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    accum_iter = args.accum_iter
    # data augmentation by following SimCLR
    data_wrapper = DataWrapper(img_size=args.input_size)
    criterion = torch.nn.MSELoss().to(device)
    optimizer.zero_grad()

    total_step = len(data_loader)
    for data_iter_step, batch in enumerate(data_loader):
        print("Epoch: [{}]  [ {} / {}]  Samples: {}  Grids: {}".format(
            epoch, data_iter_step, total_step - 1, batch["sample_id"], batch["grid_mask"].sum(dim=1).tolist()))

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / total_step + epoch, args)

        x, ys, yl, grid_mask = model_inputs(batch, device, augment=data_wrapper)
        # log_scale target，与原 MMST-ViT 一致
        z = torch.log(batch["labels"]).unsqueeze(-1).to(device, non_blocking=True)

        z_hat = model(x, ys=ys, yl=yl, grid_mask=grid_mask)
        loss = criterion(z_hat, z)
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()
        if device.type == "cuda":
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)
        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            epoch_1000x = int((data_iter_step / total_step + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def predict(model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    """返回按 DataLoader 顺序的逐县预测（bu/ac）。"""
    model.eval()
    rows = []
    for batch in data_loader:
        x, ys, yl, grid_mask = model_inputs(batch, device)
        z_hat = model(x, ys=ys, yl=yl, grid_mask=grid_mask)
        pred = torch.exp(z_hat[:, -1].float()).cpu().tolist()
        for sid, fips, year, y_true, y_pred in zip(batch["sample_id"], batch["fips"], batch["year"].tolist(),
                                                   batch["labels"].tolist(), pred):
            rows.append({"sample_id": sid, "fips": fips, "year": year, "y_true": y_true, "y_pred": y_pred})
        print(" Test {}".format(batch["sample_id"]))
    return rows


def evaluate_final_checkpoint(checkpoint_path, dataset_jsonl, data_root, output_dir, *, device="cuda",
                              num_workers=0):
    """唯一的测试入口：加载 checkpoint-final.pth，只读 2022，写 test_metrics.json 与 county_predictions.jsonl。"""
    years = capture_evaluation_years(checkpoint_path)
    output_dir = Path(output_dir)
    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model = build_model(checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    loader = build_mmst_dataloader(dataset_jsonl, data_root, years, batch_size=1, shuffle=False,
                                   load_images=True, num_workers=num_workers)
    rows = predict(model, loader, device)
    expected = loader.dataset.sample_ids
    if [r["sample_id"] for r in rows] != expected:
        raise RuntimeError("county predictions do not follow unified Dataset sample order")

    metrics = regression_metrics([r["y_true"] for r in rows], [r["y_pred"] for r in rows])
    print("Final test {}: RMSE {:.2f}  MAE {:.2f}  R2 {:.3f}  Corr {:.3f}".format(
        list(years), metrics["rmse"], metrics["mae"], metrics["r2"], metrics["corr"]))
    run_io.write_json(output_dir / "test_metrics.json",
                      {**metrics, "years": list(years), "checkpoint": str(checkpoint_path)})
    tmp = output_dir / "county_predictions.jsonl.tmp"
    tmp.unlink(missing_ok=True)
    for row in rows:
        run_io.append_jsonl(tmp, row)
    tmp.replace(output_dir / "county_predictions.jsonl")
    run_io.update_json(output_dir / "run_metadata.json", status="finished", test_years=list(years),
                       test_checkpoint=str(checkpoint_path), test_samples=len(rows),
                       finished=datetime.datetime.now().isoformat(timespec="seconds"))
    return output_dir


if __name__ == '__main__':
    main(build_parser().parse_args())
