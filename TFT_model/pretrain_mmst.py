"""MMST-style self-supervised pretraining for the TFT remote-sensing branch.

Data contract matches the CropNet five-state subset of
``MMST-ViT/data/pretrain_2017_2020.json``: 2017-2020, six monthly Sentinel-2
dates and 28 daily weather records per date. Weather tokens are Q; PVT patch
tokens are K/V.
"""

import argparse
import json
import math
import sys
import hashlib
import subprocess
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

try:
    from .models import VariableSelectionNetwork, WeatherRemotePatchPretrain
    from .pvt import PVTTinyEncoder
except ImportError:
    from models import VariableSelectionNetwork, WeatherRemotePatchPretrain
    from pvt import PVTTinyEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MMST_ROOT = PROJECT_ROOT / "MMST-ViT"
sys.path.insert(0, str(MMST_ROOT))
from current_data import CurrentMMSTPretrainDataset


FIVE_STATES = frozenset({
    "illinois", "iowa", "louisiana", "mississippi", "new_york",
})
PRETRAIN_YEARS = frozenset({2017, 2018, 2019, 2020})


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _state_name(record):
    if "State" not in record and "state" not in record:
        raise ValueError("Pretraining manifest record is missing State/state")
    return str(record.get("State", record.get("state"))).strip().lower().replace(" ", "_")


def _target_year(record):
    if "Year" not in record and "year" not in record:
        raise ValueError("Pretraining manifest record is missing Year/year")
    try:
        return int(record.get("Year", record.get("year")))
    except (TypeError, ValueError) as error:
        raise ValueError("Pretraining manifest record has an invalid Year/year") from error


def select_five_state_records(records: Sequence[dict]):
    """Keep only protocol states before validating the pretraining manifest."""
    selected = []
    for record in records:
        state = _state_name(record)
        if state in FIVE_STATES:
            selected.append(record)
    return selected


def validate_pretrain_records(records: Sequence[dict]):
    if not records:
        raise ValueError("Pretraining manifest is empty")
    states = {_state_name(record) for record in records}
    invalid_states = states - FIVE_STATES
    if invalid_states:
        raise ValueError(
            "Pretraining manifest contains states outside the five-state protocol: "
            + ", ".join(sorted(invalid_states))
        )
    invalid_years = {_target_year(record) for record in records} - PRETRAIN_YEARS
    if invalid_years:
        raise ValueError(
            "Pretraining manifest contains years outside pretraining years: "
            + ", ".join(map(str, sorted(invalid_years)))
        )
    if states != FIVE_STATES:
        missing = ", ".join(sorted(FIVE_STATES - states))
        raise ValueError("Manifest must contain exactly the five-state protocol; missing: " + missing)
    counts = {state: sum(_state_name(record) == state for record in records) for state in sorted(FIVE_STATES)}
    missing_years = {
        state: sorted(PRETRAIN_YEARS - {
            _target_year(record) for record in records if _state_name(record) == state
        })
        for state in sorted(FIVE_STATES)
    }
    incomplete = [f"{state}: {years}" for state, years in missing_years.items() if years]
    if incomplete:
        raise ValueError("Manifest must cover 2017-2020 for every state; missing " + "; ".join(incomplete))
    return counts


def adjust_learning_rate(
    optimizer, progress, *, base_lr, min_lr, warmup_epochs, total_epochs
):
    """Apply MMST-ViT's warmup and half-cycle cosine schedule."""
    if total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if warmup_epochs < 0 or warmup_epochs >= total_epochs:
        raise ValueError("warmup_epochs must satisfy 0 <= warmup_epochs < total_epochs")
    if warmup_epochs > 0 and progress < warmup_epochs:
        lr = base_lr * progress / warmup_epochs
    else:
        lr = min_lr + (base_lr - min_lr) * 0.5 * (
            1.0 + math.cos(
                math.pi * (progress - warmup_epochs) / (total_epochs - warmup_epochs)
            )
        )
    for param_group in optimizer.param_groups:
        scale = param_group.get("lr_scale", 1.0)
        param_group["lr"] = lr * scale
    return lr


def county_progress(epoch, batch_index, loader_length):
    if loader_length <= 0:
        raise ValueError("loader_length must be positive")
    return epoch + batch_index / loader_length


def schedule_county_iteration(optimizer, *, epoch, batch_index, loader_length, scheduler_config):
    """Update LR once per outer county loader iteration, like MMST-ViT."""
    return adjust_learning_rate(
        optimizer,
        county_progress(epoch, batch_index, loader_length),
        **scheduler_config,
    )


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(PROJECT_ROOT / "MMST-ViT/data/pretrain_2017_2020.json"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--blr", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "MMST-ViT/output_current/tft_pretrain.pth"))
    parser.add_argument("--resume", default=None)
    return parser


def resolve_learning_rate(args):
    if args.lr is not None:
        return args.lr
    return args.blr * args.batch_size / 256


def create_optimizer(model, learning_rate):
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 1 or name.endswith(".bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": 0.05},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
    )


def manifest_sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_pretrain_batch(images: torch.Tensor, weather: torch.Tensor):
    """Flatten county tensors from batched HWC or unbatched CHW layout."""
    if images.ndim == 6 and images.shape[-1] == 3:
        batch, dates, grids, height, width, channels = images.shape
        flat_images = images.permute(0, 1, 2, 5, 3, 4).reshape(
            batch * dates * grids, channels, height, width
        ).contiguous()
        if weather.ndim != 5 or weather.shape[:3] != images.shape[:3]:
            raise ValueError("weather must align with image batch dimensions")
        count = batch * dates * grids
    elif images.ndim == 5 and images.shape[2] == 3:
        dates, grids, channels, height, width = images.shape
        flat_images = images.reshape(dates * grids, channels, height, width).contiguous()
        if weather.ndim != 4 or weather.shape[:2] != images.shape[:2]:
            raise ValueError("weather must align with image county dimensions")
        count = dates * grids
    else:
        raise ValueError(f"unsupported pretraining image shape: {tuple(images.shape)}")
    flat_weather = weather.reshape(count, weather.shape[-2], weather.shape[-1])
    return flat_images, flat_weather


def save_checkpoint(
    path, *, model_state, optimizer_state, epoch, scheduler_config,
    manifest_path=None, manifest_hash=None, config=None,
    experiment_id="default_pure_cross_attention", git_commit_value=None,
    total_parameters=None, trainable_parameters=None,
):
    torch.save({
        "model": model_state,
        "optimizer": optimizer_state,
        "epoch": epoch,
        "scheduler": scheduler_config,
        "manifest_path": manifest_path,
        "manifest_hash": manifest_hash,
        "config": config or {},
        "protocol": {"years": [2017, 2018, 2019, 2020], "epochs": 200},
        "experiment_id": experiment_id,
        "git_commit": git_commit_value or git_commit(),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
    }, path)


def load_resume_checkpoint(
    path, model, optimizer=None, *, expected_manifest_hash=None,
    expected_scheduler=None, expected_config=None,
):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if expected_manifest_hash is not None and payload.get("manifest_hash") != expected_manifest_hash:
        raise ValueError("resume manifest hash mismatch")
    for name, expected, actual in (
        ("scheduler", expected_scheduler, payload.get("scheduler", {})),
        ("config", expected_config, payload.get("config", {})),
    ):
        if expected is not None and any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError(f"resume {name} mismatch")
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    return int(payload.get("epoch", -1)) + 1, payload


class MMSTWeatherPretrain(torch.nn.Module):
    def __init__(self, hidden_size=512, num_heads=4):
        super().__init__()
        self.pvt = PVTTinyEncoder(out_dim=hidden_size, drop=0.1)
        self.feature_projection = torch.nn.ModuleList(
            [torch.nn.Linear(1, hidden_size) for _ in range(9)]
        )
        self.vsn = VariableSelectionNetwork(
            {str(i): hidden_size for i in range(9)}, hidden_size, dropout=0.1
        )
        self.cross_modal = WeatherRemotePatchPretrain(hidden_size, num_heads, dropout=0.1)

    def forward(self, images, weather):
        # images: (B,3,H,W), weather: (B,28,9)
        patches = self.pvt.forward_tokens(images, include_cls=False)
        projected = {
            str(i): self.feature_projection[i](weather[..., i:i + 1])
            for i in range(9)
        }
        weather_tokens, _ = self.vsn(projected, torch.full(
            (weather.shape[0],), weather.shape[1], device=weather.device, dtype=torch.long
        ))
        return self.cross_modal(weather_tokens, patches)


def nt_xent(first, second, temperature=0.5):
    """Contrast corresponding flattened (batch, token) positions across views."""
    first = F.normalize(first.reshape(-1, first.shape[-1]), dim=-1)
    second = F.normalize(second.reshape(-1, second.shape[-1]), dim=-1)
    representations = torch.cat([first, second], dim=0)
    similarity = representations @ representations.transpose(0, 1) / temperature
    mask = ~torch.eye(similarity.shape[0], device=similarity.device, dtype=torch.bool)
    logits = similarity.masked_fill(~mask, -1e9)
    labels = torch.arange(similarity.shape[0], device=similarity.device)
    labels = (labels + first.shape[0]) % similarity.shape[0]
    return F.cross_entropy(logits, labels)


def main():
    args = build_parser().parse_args()
    records = json.loads(Path(args.manifest).resolve().read_text())
    records = select_five_state_records(records)
    validate_pretrain_records(records)
    dataset = CurrentMMSTPretrainDataset(records, train=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    model = MMSTWeatherPretrain().to(args.device)
    actual_lr = resolve_learning_rate(args)
    optimizer = create_optimizer(model, actual_lr)
    start_epoch = 0
    scheduler_config = {
        "base_lr": actual_lr,
        "min_lr": args.min_lr,
        "warmup_epochs": args.warmup_epochs,
        "total_epochs": args.epochs,
    }
    checkpoint_config = {
        **scheduler_config,
        "blr": args.blr,
        "batch_size": args.batch_size,
        "actual_lr": actual_lr,
    }
    augment = transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2), transforms.GaussianBlur(9),
    ])
    manifest_hash = manifest_sha256(args.manifest)
    if args.resume:
        start_epoch, _ = load_resume_checkpoint(
            args.resume, model, optimizer,
            expected_manifest_hash=manifest_hash,
            expected_scheduler=scheduler_config,
            expected_config={"epochs": args.epochs, "warmup_epochs": args.warmup_epochs, "blr": args.blr},
        )
    for epoch in range(start_epoch, args.epochs):
        model.train(); total = 0.0; steps = 0
        for batch_index, (images, weather) in enumerate(loader):
            schedule_county_iteration(
                optimizer,
                epoch=epoch,
                batch_index=batch_index,
                loader_length=len(loader),
                scheduler_config=scheduler_config,
            )
            flat_images, flat_weather = prepare_pretrain_batch(images, weather)
            flat_images = flat_images.float() / 255.0
            flat_images = flat_images.to(args.device)
            flat_weather = flat_weather.to(args.device)
            for start in range(0, flat_images.shape[0], args.batch_size):
                image_batch = flat_images[start:start + args.batch_size]
                weather_batch = flat_weather[start:start + args.batch_size]
                if image_batch.shape[0] < 2: continue
                first = augment(image_batch); second = augment(image_batch)
                z1, _ = model(first, weather_batch); z2, _ = model(second, weather_batch)
                loss = nt_xent(z1, z2)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total += float(loss.item()); steps += 1
        print(f"epoch={epoch + 1} loss={total / max(steps, 1):.6f}")
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            output,
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            epoch=epoch,
            scheduler_config=checkpoint_config,
            manifest_path=str(Path(args.manifest).resolve()),
            manifest_hash=manifest_hash,
            config=vars(args),
            total_parameters=sum(parameter.numel() for parameter in model.parameters()),
            trainable_parameters=sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        )


if __name__ == "__main__":
    main()
