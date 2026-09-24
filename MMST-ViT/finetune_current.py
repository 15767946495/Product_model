"""Official MMST-ViT fine-tuning on 2021 and evaluation on 2022."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from current_data import CurrentMMSTDataset
from models_mmst_vit import MMST_ViT
from models_pvt_simclr import PVTSimCLR
from util.metrics import evaluate


def iterate(dataset, model, device, indices, optimizer=None):
    predictions, labels = [], []
    model.train(optimizer is not None)
    for index in indices:
        images, short, long, label, _, _ = dataset[index]
        images = images.unsqueeze(0).to(device)
        short = short.unsqueeze(0).to(device)
        long = long.unsqueeze(0).to(device)
        label = label.to(device)
        prediction = model(images, ys=short, yl=long)
        loss = torch.nn.functional.mse_loss(prediction[:, -1], label)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        predictions.append(float(torch.exp(prediction[:, -1]).item()))
        labels.append(float(torch.exp(label).item()))
    return np.asarray(predictions), np.asarray(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-manifest", default="data/finetune_2021.json")
    parser.add_argument("--eval-manifest", default="data/eval_2022.json")
    parser.add_argument("--pretrained", default="output_current/pretrain/checkpoint-latest.pth")
    parser.add_argument("--output-dir", default="output_current/finetune")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    rank = dist.get_rank() if distributed else 0

    train_records = json.loads(Path(args.train_manifest).read_text(encoding="utf-8"))
    eval_records = json.loads(Path(args.eval_manifest).read_text(encoding="utf-8"))
    train_data = CurrentMMSTDataset(train_records, train=True)
    eval_data = CurrentMMSTDataset(eval_records, train=False)
    pvt = PVTSimCLR("pvt_tiny", out_dim=512, context_dim=9, pretrained=True)
    checkpoint = torch.load(args.pretrained, map_location="cpu", weights_only=False)
    pvt.load_state_dict(checkpoint["model"])
    model = MMST_ViT(
        out_dim=1,
        num_grid=160,
        num_year=5,
        pvt_backbone=pvt,
        context_dim=9,
        dim=512,
        batch_size=32,
    ).to(args.device)
    if distributed:
        model = DDP(model, device_ids=[torch.cuda.current_device()])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    best_rmse = float("inf")
    train_sampler = DistributedSampler(train_data, shuffle=True) if distributed else None
    eval_indices = list(range(len(eval_data)))
    if distributed:
        eval_sampler = DistributedSampler(eval_data, shuffle=False)
    else:
        eval_sampler = None
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            train_indices = list(iter(train_sampler))
        else:
            train_indices = list(range(len(train_data)))
        iterate(train_data, model, args.device, train_indices, optimizer)
        if eval_sampler is not None:
            eval_indices = list(iter(eval_sampler))
        pred, label = iterate(eval_data, model, args.device, eval_indices)
        if distributed:
            pred_tensor = torch.tensor(pred, device=args.device, dtype=torch.float64)
            label_tensor = torch.tensor(label, device=args.device, dtype=torch.float64)
            sums = torch.tensor([
                len(pred),
                ((pred_tensor - label_tensor) ** 2).sum(),
                pred_tensor.sum(),
                label_tensor.sum(),
            ], device=args.device)
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            # RMSE is globally exact; R2/Corr remain rank-local in this lightweight runner.
            rmse = float(torch.sqrt(sums[1] / sums[0].clamp_min(1)).item())
            r2 = float("nan")
            corr = float("nan")
        else:
            rmse, r2, corr = evaluate(label, pred)
        if rank == 0 and rmse < best_rmse:
            best_rmse = rmse
            state = model.module.state_dict() if distributed else model.state_dict()
            torch.save({"model": state, "epoch": epoch}, output / "best.pth")
        if rank == 0:
            print(f"epoch={epoch + 1} rmse={rmse:.4f} r2={r2:.4f} corr={corr:.4f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
