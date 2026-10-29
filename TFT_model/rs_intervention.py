"""TFT 遥感分支的测试时干预：检验 PVT 输出是否真正影响预测。

先正常前向一次并缓存每个有效网格-月份的 PVT token，再用全局均值、月份均值、
同月跨县打乱和全零替换 PVT 输出重放天气/土壤输入（重放不再读取影像）。
若替换后预测不变，则遥感内容在推理时没有被模型使用。
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from TFT_model import dataset_adapter as adapter
from TFT_model.infer import ADAPTATION, MODEL_NAME
from TFT_model.model_adapter import build_model
from TFT_model.models import AG_DATE_PAIRS, RS_VARIABLE_NAME
from train_dataset import run_output as ro
from train_dataset.dataset import DATA_ROOT
from train_dataset.metrics import regression_metrics

N_MONTHS = len(AG_DATE_PAIRS)


def valid_token_mask(grid_mask: torch.Tensor, ag_grid_mask: torch.Tensor) -> torch.Tensor:
    """(B, G, 6) bool：PVT 实际编码的网格-月份位置。"""
    return ag_grid_mask.bool().permute(0, 2, 1) & grid_mask.bool()[:, :, None]


def token_statistics(tokens: torch.Tensor, months: torch.Tensor | None = None,
                     pairs: int = 20000, seed: int = 0) -> dict:
    """tokens: (N, H) 有效 PVT token。返回坍缩程度统计。"""
    tokens = tokens.double()
    mean = tokens.mean(dim=0)
    deviation = tokens - mean
    total_var = float(deviation.pow(2).sum(dim=1).mean())
    eig = torch.linalg.svdvals(deviation).pow(2)
    eig = eig / eig.sum().clamp_min(1e-30)
    stats = {
        "n_tokens": int(tokens.shape[0]),
        "mean_norm": float(mean.norm()),
        "rms_deviation_norm": total_var ** 0.5,
        "deviation_to_mean_ratio": total_var ** 0.5 / max(float(mean.norm()), 1e-12),
        "effective_rank": float(torch.exp(-(eig * torch.log(eig.clamp_min(1e-30))).sum())),
        "pca_top1_share": float(eig[0]),
    }
    generator = torch.Generator().manual_seed(seed)
    left = torch.randint(0, tokens.shape[0], (pairs,), generator=generator)
    right = torch.randint(0, tokens.shape[0], (pairs,), generator=generator)
    stats["pairwise_cos_mean"] = float(
        torch.nn.functional.cosine_similarity(tokens[left], tokens[right]).mean()
    )
    if months is not None:
        between = 0.0
        for month in months.unique():
            selected = months == month
            between += int(selected.sum()) * float((tokens[selected].mean(dim=0) - mean).pow(2).sum())
        stats["between_month_var_share"] = between / tokens.shape[0] / max(total_var, 1e-30)
    return stats


@contextlib.contextmanager
def patched_remote_sensing(model, *, capture=None, replace=None):
    """临时替换实例上的 _forward_remote_sensing。

    capture(tokens, valid)：正常编码后回调；replace(grid_mask, ag_grid_mask, device)：
    直接返回 (B, G, 6, H) token，跳过 PVT。退出时恢复原方法。
    """
    original = model._forward_remote_sensing
    shadowed = model.__dict__.get("_forward_remote_sensing")

    def wrapper(ag_images, grid_coords, grid_mask, device, ag_grid_mask=None):
        if replace is not None:
            return replace(grid_mask, ag_grid_mask, device)
        tokens = original(ag_images, grid_coords, grid_mask, device, ag_grid_mask)
        if capture is not None:
            capture(tokens, valid_token_mask(grid_mask.to(device), ag_grid_mask.to(device)))
        return tokens

    model._forward_remote_sensing = wrapper
    try:
        yield
    finally:
        if shadowed is None:
            del model._forward_remote_sensing
        else:
            model._forward_remote_sensing = shadowed


@dataclass
class Collected:
    preds: list = field(default_factory=list)       # 原尺度预测，逐样本
    labels: list = field(default_factory=list)
    fips: list = field(default_factory=list)
    batches: list = field(default_factory=list)     # 不含影像的 CPU 输入，供重放
    tokens: list = field(default_factory=list)      # (B, G, 6, H) CPU
    valid: list = field(default_factory=list)       # (B, G, 6) CPU
    rs_vsn_weight: list = field(default_factory=list)  # rs_as_variable 时遥感变量的平均 VSN 权重


def _replay_inputs(batch: dict) -> dict:
    inputs = adapter.model_inputs(batch, "cpu")
    inputs.pop("ag_images")
    return inputs


@torch.no_grad()
def collect(model, batches, device) -> Collected:
    """正常前向并缓存 PVT token；model 需处于 eval 模式。"""
    out = Collected()
    pending = []

    def capture(tokens, valid):
        pending.append((tokens.detach().float().cpu(), valid.cpu()))

    with patched_remote_sensing(model, capture=capture):
        for batch in batches:
            pending.clear()
            pred, _, aux = model(**adapter.model_inputs(batch, device))
            tokens, valid = pending[0]
            out.tokens.append(tokens)
            out.valid.append(valid)
            out.preds.extend(torch.exp(pred[:, 0]).cpu().tolist())
            out.labels.extend(batch["labels"].float().tolist())
            out.fips.extend(batch["fips"])
            out.batches.append(_replay_inputs(batch))
            if getattr(model, "rs_as_variable", False):
                out.rs_vsn_weight.append(_rs_vsn_weight(model, aux, batch))
    return out


def _rs_vsn_weight(model, aux, batch) -> float:
    weights = aux["grid_vsn_weights"]  # (B*G, T, 1, P+1)
    B, G = batch["grid_mask"].shape
    index = model.grid_vsn.var_names.index(RS_VARIABLE_NAME)
    rs_weight = weights[..., 0, index].reshape(B, G, -1)
    mask = batch["grid_mask"].to(rs_weight.device)[:, :, None].expand_as(rs_weight)
    return float(rs_weight[mask].mean())


@torch.no_grad()
def replay(model, collected: Collected, replacements, device) -> np.ndarray:
    """用 replacements[b] (B, G, 6, H) 代替 PVT 输出重放缓存输入，返回原尺度预测。"""
    state = {"batch": 0}

    def replace(grid_mask, ag_grid_mask, run_device):
        tokens = replacements[state["batch"]].to(run_device)
        valid = valid_token_mask(grid_mask.to(run_device), ag_grid_mask.to(run_device))
        return tokens * valid[..., None].to(tokens.dtype)

    preds = []
    with patched_remote_sensing(model, replace=replace):
        for inputs in collected.batches:
            run_inputs = {key: value.to(device) for key, value in inputs.items()}
            B, G = inputs["grid_mask"].shape
            run_inputs["ag_images"] = torch.zeros(B, N_MONTHS, G, 1, 1, 1, device=device)
            pred, _, _ = model(**run_inputs)
            preds.extend(torch.exp(pred[:, 0]).cpu().tolist())
            state["batch"] += 1
    return np.asarray(preds)


def flatten_tokens(collected: Collected) -> tuple[torch.Tensor, torch.Tensor]:
    tokens, months = [], []
    for batch_tokens, batch_valid in zip(collected.tokens, collected.valid):
        for month in range(N_MONTHS):
            selected = batch_tokens[:, :, month][batch_valid[:, :, month]]
            tokens.append(selected)
            months.append(torch.full((selected.shape[0],), month))
    return torch.cat(tokens), torch.cat(months)


def month_means(collected: Collected) -> list[torch.Tensor]:
    tokens, months = flatten_tokens(collected)
    return [
        tokens[months == month].mean(dim=0) if bool((months == month).any())
        else torch.zeros(tokens.shape[1])
        for month in range(N_MONTHS)
    ]


def fill_month_vectors(collected: Collected, vectors) -> list[torch.Tensor]:
    out = []
    for tokens in collected.tokens:
        filled = torch.zeros_like(tokens)
        for month in range(N_MONTHS):
            filled[:, :, month] = vectors[month].to(tokens.dtype)
        out.append(filled)
    return out


def shuffle_within_month(collected: Collected, seed: int) -> list[torch.Tensor]:
    """在全部样本的同一月份有效位置间随机置换 token（跨县、跨网格）。"""
    generator = torch.Generator().manual_seed(seed)
    pools = {}
    for month in range(N_MONTHS):
        pool = torch.cat([t[:, :, month][v[:, :, month]] for t, v in zip(collected.tokens, collected.valid)])
        pools[month] = pool[torch.randperm(pool.shape[0], generator=generator)]
    cursor = dict.fromkeys(range(N_MONTHS), 0)
    out = []
    for tokens, valid in zip(collected.tokens, collected.valid):
        shuffled = torch.zeros_like(tokens)
        for month in range(N_MONTHS):
            count = int(valid[:, :, month].sum())
            shuffled[:, :, month][valid[:, :, month]] = pools[month][cursor[month]:cursor[month] + count]
            cursor[month] += count
        out.append(shuffled)
    return out


def _delta_summary(labels, base, pred) -> dict:
    delta = pred - base
    return {
        **regression_metrics(labels, pred),
        "mean_abs_delta_bu": float(np.abs(delta).mean()),
        "max_abs_delta_bu": float(np.abs(delta).max()),
        "mean_delta_bu": float(delta.mean()),
    }


def run_interventions(model, batches, device, shuffle_seeds=(0, 1, 2)) -> dict:
    collected = collect(model, batches, device)
    labels, base = np.asarray(collected.labels), np.asarray(collected.preds)
    tokens, months = flatten_tokens(collected)
    means = month_means(collected)
    global_mean = tokens.mean(dim=0)
    results = {
        "baseline": regression_metrics(labels, base),
        "token_stats": token_statistics(tokens, months),
    }
    if collected.rs_vsn_weight:
        results["rs_vsn_weight_mean"] = float(np.mean(collected.rs_vsn_weight))
    plans = {
        "identity_replay": lambda: collected.tokens,
        "global_mean": lambda: fill_month_vectors(collected, [global_mean] * N_MONTHS),
        "month_mean": lambda: fill_month_vectors(collected, means),
        "month_mean_reversed": lambda: fill_month_vectors(collected, means[::-1]),
        **{f"shuffle_within_month_s{seed}": (lambda s=seed: shuffle_within_month(collected, s))
           for seed in shuffle_seeds},
        "zeros": lambda: [torch.zeros_like(t) for t in collected.tokens],
    }
    for name, build in plans.items():
        results[name] = _delta_summary(labels, base, replay(model, collected, build(), device))
    results["baseline_predictions"] = dict(zip(collected.fips, base.tolist()))
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("TFT remote-sensing test-time intervention")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-jsonl", default=str(ro.PROJECT_ROOT / "train_dataset" / "dataset.jsonl"))
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--output", default="", help="defaults to <run_dir>/rs_intervention.json")
    return parser


def main(args) -> dict:
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint, _ = ro.validate_unified_run(
        checkpoint_path, checkpoint_path.parent,
        model_name=MODEL_NAME, adaptation=ADAPTATION, dataset_jsonl=args.dataset_jsonl,
        train_years=adapter.TRAIN_YEARS, test_years=adapter.TEST_YEARS,
    )
    config = dict(checkpoint["model_config"])
    if config.get("remote_constant", False) or config.get("rs_feature_dim"):
        raise ValueError("this checkpoint has no PVT branch to intervene on (remote_constant or numeric rs features)")
    config["vit_pretrained"] = False
    device = torch.device(args.device)
    model = build_model(config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    stats = adapter.stats_from_json(checkpoint["stats"])
    dataset = adapter.build_dataset(args.dataset_jsonl, args.data_root, adapter.TEST_YEARS)
    loader = adapter.build_loader(
        dataset, stats, args.batch_size, False,
        use_constructed=bool(checkpoint.get("use_constructed", True)),
        num_workers=args.num_workers,
    )
    results = {"checkpoint": str(checkpoint_path), **run_interventions(model, loader, device)}
    output = Path(args.output) if args.output else checkpoint_path.parent / "rs_intervention.json"
    ro.write_json(output, results)
    for name, value in results.items():
        if isinstance(value, dict) and "rmse" in value:
            print(name, {k: round(v, 4) for k, v in value.items() if isinstance(v, float)})
    print("token_stats", results["token_stats"])
    return results


if __name__ == "__main__":
    main(build_parser().parse_args())
