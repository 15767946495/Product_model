"""Official MMST-ViT SimCLR pretraining on 2017-2020 current CropNet data."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from current_data import CurrentMMSTPretrainDataset
from dataset import sentinel_wrapper
from loss.contrastive_loss import ContrastiveLoss
from models_pvt_simclr import PVTSimCLR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/pretrain_2017_2020.json")
    parser.add_argument("--output-dir", default="output_current/pretrain")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    records = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if not records:
        raise ValueError(f"Pretraining manifest is empty: {args.manifest}")
    states = {str(record["state"]).strip().lower() for record in records}
    if len(states) < 6:
        raise ValueError(
            "MMST-ViT pretraining must use all available states, not the five-state "
            f"fine-tuning subset; manifest contains only {sorted(states)}"
        )
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if not distributed or int(os.environ.get("RANK", "0")) == 0:
        print(
            f"Pretraining data: {len(records)} county-years across {len(states)} states "
            f"({', '.join(sorted(states))})"
        )
    dataset = CurrentMMSTPretrainDataset(records, train=False)
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    rank = dist.get_rank() if distributed else 0
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
    )
    model = PVTSimCLR("pvt_tiny", out_dim=512, context_dim=9, pretrained=True).to(args.device)
    if distributed:
        # The official PVTSimCLR uses backbone.forward_features(), so the
        # classification head is intentionally unused during pretraining.
        model = DDP(
            model,
            device_ids=[torch.cuda.current_device()],
            find_unused_parameters=True,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.05)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        total_loss = 0.0
        steps = 0
        image_pairs = 0
        county_progress = tqdm(
            loader,
            desc=f"Epoch {epoch + 1}/{args.epochs}",
            disable=rank != 0,
            dynamic_ncols=True,
        )
        for county_step, (images, short) in enumerate(county_progress):
            # Match the official pipeline: B,T,G,H,W,C -> wrapper ->
            # flattened augmented image batches and weather contexts.
            official_images = images
            inner_loader = sentinel_wrapper.get_data_loader(
                official_images,
                short,
                batch_size=args.batch_size,
                num_workers=0,
            )
            inner_iterator = iter(inner_loader)
            first_batch = next(inner_iterator)
            local_steps = len(inner_loader)
            if distributed:
                max_steps_tensor = torch.tensor(
                    local_steps, device=args.device, dtype=torch.long
                )
                dist.all_reduce(max_steps_tensor, op=dist.ReduceOp.MAX)
                synchronized_steps = int(max_steps_tensor.item())
            else:
                synchronized_steps = local_steps

            cached_batch = first_batch
            for inner_step in range(synchronized_steps):
                if inner_step == 0:
                    first, second, weather_batch = first_batch
                    has_real_batch = first.shape[0] >= 2
                elif inner_step < local_steps:
                    first, second, weather_batch = next(inner_iterator)
                    has_real_batch = first.shape[0] >= 2
                else:
                    # All ranks must execute the same number of forward/backward
                    # calls. Reuse one local batch with a zero loss while ranks
                    # that own larger counties finish their real inner batches.
                    first, second, weather_batch = cached_batch
                    has_real_batch = False

                first = first.to(args.device, non_blocking=True)
                second = second.to(args.device, non_blocking=True)
                weather_batch = weather_batch.to(args.device, non_blocking=True)
                first_embedding = model(first, weather_batch)
                second_embedding = model(second, weather_batch)
                active_ranks = torch.tensor(
                    1 if has_real_batch else 0,
                    device=args.device,
                    dtype=torch.float32,
                )
                if distributed:
                    dist.all_reduce(active_ranks, op=dist.ReduceOp.SUM)
                if has_real_batch:
                    loss = ContrastiveLoss(first_embedding.shape[0], args.device)(
                        first_embedding, second_embedding
                    )
                    # DDP averages gradients across every rank. Compensate for
                    # ranks participating with zero-loss synchronization steps.
                    if distributed:
                        loss = loss * (dist.get_world_size() / active_ranks.clamp_min(1.0))
                else:
                    loss = (first_embedding.sum() + second_embedding.sum()) * 0.0
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if has_real_batch:
                    total_loss += float(loss.detach().item())
                    steps += 1
                    image_pairs += int(first.shape[0])
            if rank == 0:
                county_progress.set_postfix(
                    county_rank=f"{county_step + 1}/{len(loader)}",
                    county_global=f"~{min((county_step + 1) * dist.get_world_size() if distributed else county_step + 1, len(dataset))}/{len(dataset)}",
                    inner_max=synchronized_steps,
                    pairs=image_pairs,
                    loss=f"{total_loss / max(steps, 1):.4f}",
                )
        if distributed:
            totals = torch.tensor([total_loss, steps], device=args.device, dtype=torch.float64)
            dist.all_reduce(totals, op=dist.ReduceOp.SUM)
            total_loss, steps = totals.tolist()
        if rank == 0:
            state = model.module.state_dict() if distributed else model.state_dict()
            checkpoint = {"model": state, "epoch": epoch}
            torch.save(checkpoint, output / "checkpoint-latest.pth")
            print(f"epoch={epoch + 1} loss={total_loss / max(steps, 1):.6f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
