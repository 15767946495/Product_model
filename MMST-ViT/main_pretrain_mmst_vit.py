"""PVT-SimCLR 无监督预训练：统一 CropNet JSONL，固定 2017--2020。

每个县年 batch 展开为 (月 x 有效网格) 张图像及对应 28 天 9 维天气上下文，按 --batch_size 分块做 SimCLR。
输出：output_root/pvt_simclr/<run_id>/{config.json, checkpoint-latest.pth, checkpoint-final.pth,
train_metrics.jsonl, run_metadata.json}。
"""

import argparse
import datetime
import math
import sys
import time
from typing import Iterable

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from einops import rearrange
from torch.utils.tensorboard import SummaryWriter

import timm.optim.optim_factory as optim_factory

import util.lr_sched as lr_sched
import util.misc as misc
from dataset.data_wrapper import SimCLRViews
from dataset_adapter import MMST_YEARS_PRETRAIN, build_mmst_dataloader, build_mmst_dataset
from loss.contrastive_loss import ContrastiveLoss
from models_pvt_simclr import PVTSimCLR
from util import run_io
from util.misc import NativeScalerWithGradNormCount as NativeScaler

PRETRAIN_YEARS = MMST_YEARS_PRETRAIN
MODEL_NAME = "pvt_simclr"


def build_parser():
    parser = argparse.ArgumentParser('PVT SimCLR pre-training (unified CropNet, 2017--2020)')
    run_io.add_common_args(parser, model_name=MODEL_NAME)
    parser.add_argument('--batch_size', default=32, type=int,
                        help='images per SimCLR step (grid-month crops of one county-year)')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--warmup_epochs', type=int, default=20, metavar='N', help='epochs to warmup LR')
    return parser


def build_model(args, pretrained=None):
    pretrained = not args.no_pvt_pretrained if pretrained is None else pretrained
    return PVTSimCLR(args.model_pvt, out_dim=args.embed_dim, context_dim=9, num_head=args.num_heads,
                     mm_depth=args.mm_depth, pretrained=pretrained)


def model_config(args):
    return {k: getattr(args, k) for k in ("model_pvt", "embed_dim", "num_heads", "mm_depth")}


def main(args):
    misc.init_distributed_mode(args)
    run_dir = run_io.prepare_run(args)
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    dataset = build_mmst_dataset(args.dataset_jsonl, args.data_root, PRETRAIN_YEARS, load_images=True)
    sampler = None
    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=misc.get_world_size(),
                                                      rank=misc.get_rank(), shuffle=True)
    # 每步一个县年；网格数不一，网格级分块在 train_one_epoch 内完成
    data_loader = build_mmst_dataloader(None, None, PRETRAIN_YEARS, batch_size=1, shuffle=sampler is None,
                                        sampler=sampler, num_workers=args.num_workers,
                                        pin_memory=args.pin_mem, dataset=dataset)

    log_writer = SummaryWriter(log_dir=str(run_dir / "tensorboard")) if misc.is_main_process() else None

    model = build_model(args)
    model.to(device)
    model_without_ddp = model

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("actual lr: %.2e, effective batch size: %d" % (args.lr, eff_batch_size))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    loss_scaler = NativeScaler()
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    started = datetime.datetime.now().isoformat(timespec="seconds")
    if misc.is_main_process():
        run_io.write_json(run_dir / "run_metadata.json", {
            "model": MODEL_NAME, "run_id": args.run_id, "dataset_jsonl": args.dataset_jsonl,
            "data_root": args.data_root, "train_years": list(PRETRAIN_YEARS), "train_samples": len(dataset),
            "world_size": misc.get_world_size(), "started": started, "status": "running",
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

    if misc.is_main_process():
        final = run_io.save_final_checkpoint(run_dir, model_without_ddp, epoch=epoch,
                                             model_config=model_config(args), train_years=PRETRAIN_YEARS)
        run_io.update_json(run_dir / "run_metadata.json", status="finished", final_checkpoint=str(final),
                           finished=datetime.datetime.now().isoformat(timespec="seconds"))
        if log_writer is not None:
            log_writer.close()
    print('Training time {}'.format(datetime.timedelta(seconds=int(time.time() - start_time))))
    return run_dir


def grid_month_views(batch):
    """MMST batch -> 有效 (月, 网格) 的图像 (N,3,H,W) 与天气上下文 (N,28,9)；padding 网格不参与。"""
    valid = rearrange(batch["image_mask"] & batch["grid_mask"][:, None], 'b t g -> (b t g)')
    images = rearrange(batch["images"], 'b t g c h w -> (b t g) c h w')[valid]
    context = rearrange(batch["weather_context"], 'b t g n d -> (b t g) n d')[valid]
    return images, context


def train_one_epoch(model: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, log_writer=None, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    augment = SimCLRViews(img_size=args.input_size)
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    total_step = len(data_loader)
    for data_iter_step, batch in enumerate(data_loader):
        images, context = grid_month_views(batch)
        print("Epoch: [{}]  [ {} / {}]  Sample: {}  Grid-months: {}"
              .format(epoch, data_iter_step, total_step - 1, batch["sample_id"][0], images.shape[0]))

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / total_step + epoch, args)

        # prevent the number of grids from being too large to cause out of memory
        for start in range(0, images.shape[0], args.batch_size):
            xi, xj = augment(images[start:start + args.batch_size])
            xi = xi.to(device, non_blocking=True)
            xj = xj.to(device, non_blocking=True)
            ys = context[start:start + args.batch_size].to(device, non_blocking=True)

            zi = model(xi, ys)
            zj = model(xj, ys)

            criterion = ContrastiveLoss(zi.shape[0], device)
            loss = criterion(zi, zj)

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


if __name__ == '__main__':
    main(build_parser().parse_args())
