"""导出正式 TFT checkpoint 在 2022 测试集上的内部权重统计。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from TFT_model import dataset_adapter as adapter
from TFT_model.infer import ADAPTATION, MODEL_NAME
from TFT_model.model_adapter import build_model
from train_dataset import run_output as ro
from train_dataset.dataset import DATA_ROOT

MONTHS = tuple(range(4, 10))
MONTH_LABELS = [f"{month}月" for month in MONTHS]


def monthly_vsn_summary(
    weights: torch.Tensor,
    month_ids: torch.Tensor,
    grid_mask: torch.Tensor,
    seq_lens: torch.Tensor,
) -> torch.Tensor:
    """将 (B*G,T,1,P) VSN 权重按有效网格和日期汇总为 (B,6,P)。"""
    if weights.ndim != 4 or weights.shape[2] != 1:
        raise ValueError("weights must have shape (B*G,T,1,P)")
    batch_size, grid_count = grid_mask.shape
    time_count = month_ids.shape[1]
    if weights.shape[:2] != (batch_size * grid_count, time_count):
        raise ValueError("weights shape is inconsistent with grid_mask and month_ids")
    reshaped = weights[:, :, 0, :].reshape(
        batch_size, grid_count, time_count, weights.shape[-1]
    )
    date_valid = torch.arange(time_count, device=seq_lens.device)[None, :] < seq_lens[:, None]
    outputs = []
    for month in MONTHS:
        valid = (
            grid_mask.to(dtype=torch.bool)[:, :, None]
            & date_valid[:, None, :]
            & (month_ids[:, None, :] == month)
        )
        count = valid.sum(dim=(1, 2))
        total = (reshaped * valid.unsqueeze(-1).to(reshaped.dtype)).sum(dim=(1, 2))
        value = total / count.clamp_min(1).unsqueeze(-1)
        value = value.masked_fill(count.unsqueeze(-1) == 0, float("nan"))
        outputs.append(value)
    return torch.stack(outputs, dim=1)


def spatial_concentration(
    spatial_weights: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """计算每个样本月份的空间注意力熵、最大权重和有效网格数。"""
    if spatial_weights.shape != valid_mask.shape:
        raise ValueError("spatial_weights and valid_mask must have identical shapes")
    valid = valid_mask.to(dtype=torch.bool)
    positive = valid & (spatial_weights > 0)
    probabilities = torch.where(positive, spatial_weights, torch.zeros_like(spatial_weights))
    entropy = torch.where(
        positive,
        -probabilities * torch.log(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)),
        torch.zeros_like(probabilities),
    ).sum(dim=-1)
    valid_count = valid.sum(dim=-1)
    normalized = torch.where(
        valid_count > 1,
        entropy / torch.log(valid_count.to(entropy.dtype)),
        torch.zeros_like(entropy),
    )
    max_weight = spatial_weights.masked_fill(~valid, float("-inf")).max(dim=-1).values
    no_valid = valid_count == 0
    nan = torch.full_like(entropy, float("nan"))
    return {
        "entropy": torch.where(no_valid, nan, entropy),
        "normalized_entropy": torch.where(no_valid, nan, normalized),
        "max_weight": torch.where(no_valid, nan, max_weight),
        "effective_grids": torch.where(no_valid, nan, torch.exp(entropy)),
    }


def masked_mean_temporal(
    attention: torch.Tensor,
    month_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """对 query/key 都有效的因果时间注意力求样本均值。"""
    if attention.ndim != 3 or attention.shape[0] != month_valid.shape[0]:
        raise ValueError("attention must have shape (B,M,M)")
    if attention.shape[1:] != (month_valid.shape[1], month_valid.shape[1]):
        raise ValueError("attention and month_valid month dimensions differ")
    month_count = month_valid.shape[1]
    causal = torch.tril(torch.ones(month_count, month_count, dtype=torch.bool, device=attention.device))
    pair_valid = month_valid[:, :, None] & month_valid[:, None, :] & causal[None, :, :]
    count = pair_valid.sum(dim=0)
    total = (attention * pair_valid.to(attention.dtype)).sum(dim=0)
    mean = total / count.clamp_min(1).to(attention.dtype)
    valid = count > 0
    return mean.masked_fill(~valid, float("nan")), valid


def _configure_chinese_font() -> str | None:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    candidates = (
        "/usr/share/fonts/google-noto-cjk/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJKsc-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            font_manager.fontManager.addfont(candidate)
            name = font_manager.FontProperties(fname=candidate).get_name()
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            plt.rcParams["svg.fonttype"] = "none"
            return candidate
    return None


def _save_figure(figure, output_dir: Path, stem: str) -> None:
    import matplotlib.pyplot as plt

    for suffix in ("png", "svg"):
        figure.savefig(output_dir / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def _summary_rows(frame: pd.DataFrame, groups: list[str], values: list[str]) -> pd.DataFrame:
    rows = []
    for keys, group in frame.groupby(groups, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(groups, keys))
        for value in values:
            series = group[value].dropna()
            rows.append({
                **base,
                "metric": value,
                "mean": series.mean(),
                "sd": series.std(ddof=1),
                "q25": series.quantile(0.25),
                "q50": series.quantile(0.50),
                "q75": series.quantile(0.75),
                "count": int(series.size),
            })
    return pd.DataFrame(rows)


def _plot_outputs(
    county_vsn: pd.DataFrame,
    monthly_vsn: pd.DataFrame,
    county_spatial: pd.DataFrame,
    temporal_matrix: pd.DataFrame,
    feature_names: list[str],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    _configure_chinese_font()

    heatmap_data = monthly_vsn.pivot(index="feature", columns="month", values="mean").loc[feature_names]
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(heatmap_data.to_numpy(), aspect="auto", cmap="YlGnBu")
    axis.set_xticks(range(6), MONTH_LABELS)
    axis.set_yticks(range(len(feature_names)), feature_names)
    for row in range(len(feature_names)):
        for column in range(6):
            axis.text(column, row, f"{heatmap_data.iloc[row, column]:.3f}", ha="center", va="center", fontsize=7)
    figure.colorbar(image, ax=axis, label="平均权重")
    axis.set(title="2022年动态变量选择权重月度均值", xlabel="月份", ylabel="动态变量")
    _save_figure(figure, output_dir, "monthly_vsn_heatmap")

    feature_means = county_vsn[feature_names].mean().sort_values(ascending=False)
    top_features = feature_means.head(5).index.tolist()
    long_vsn = county_vsn.melt(
        id_vars=["sample_id", "fips", "year", "month"],
        value_vars=top_features,
        var_name="feature",
        value_name="weight",
    )
    figure, axes = plt.subplots(len(top_features), 1, figsize=(10, 2.3 * len(top_features)), sharex=True)
    for axis, feature in zip(np.atleast_1d(axes), top_features):
        feature_data = long_vsn[long_vsn["feature"] == feature]
        values = [feature_data.loc[feature_data["month"] == month, "weight"].dropna() for month in MONTHS]
        boxes = axis.boxplot(values, tick_labels=MONTH_LABELS, showfliers=False, patch_artist=True)
        for box in boxes["boxes"]:
            box.set_facecolor("#4C78A8")
        axis.set(ylabel=feature, xlabel="")
    axes[-1].set_xlabel("月份")
    figure.suptitle("全期平均权重前5个变量的县级月度分布", y=1.0)
    figure.tight_layout()
    _save_figure(figure, output_dir, "vsn_top_features_boxplot")

    matrix = temporal_matrix.pivot(index="query_month", columns="key_month", values="mean_attention")
    figure, axis = plt.subplots(figsize=(7, 6))
    masked_matrix = np.ma.masked_invalid(matrix.to_numpy())
    image = axis.imshow(masked_matrix, cmap="viridis", vmin=0.0, vmax=np.nanmax(matrix.to_numpy()))
    axis.set_xticks(range(6), MONTH_LABELS)
    axis.set_yticks(range(6), MONTH_LABELS)
    for query in range(6):
        for key in range(6):
            value = matrix.iloc[query, key]
            if not pd.isna(value):
                axis.text(key, query, f"{value:.3f}", ha="center", va="center", color="white", fontsize=9)
    figure.colorbar(image, ax=axis, label="平均注意力")
    axis.set(title="2022年时间注意力均值", xlabel="被关注月份", ylabel="查询月份")
    _save_figure(figure, output_dir, "temporal_attention_heatmap")

    plot_data = county_spatial.melt(
        id_vars=["sample_id", "fips", "year", "month"],
        value_vars=["normalized_entropy", "max_weight", "effective_grids"],
        var_name="metric",
        value_name="value",
    )
    labels = {
        "normalized_entropy": "归一化熵",
        "max_weight": "最大空间权重",
        "effective_grids": "有效网格数 exp(H)",
    }
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for axis, metric in zip(axes, labels):
        metric_data = plot_data[plot_data["metric"] == metric]
        values = [metric_data.loc[metric_data["month"] == month, "value"].dropna() for month in MONTHS]
        boxes = axis.boxplot(values, tick_labels=MONTH_LABELS, showfliers=False, patch_artist=True)
        for box in boxes["boxes"]:
            box.set_facecolor("#59A14F")
        axis.set(title=labels[metric], xlabel="月份", ylabel="")
    figure.suptitle("2022年县内空间注意力集中度", y=1.02)
    figure.tight_layout()
    _save_figure(figure, output_dir, "spatial_attention_concentration")


@torch.no_grad()
def export_attention_analysis(
    checkpoint_path,
    dataset_jsonl,
    data_root,
    output_dir,
    *,
    device="cuda:0",
    batch_size=4,
) -> dict:
    checkpoint_path = Path(checkpoint_path).resolve()
    run_dir = checkpoint_path.parent
    output_dir = Path(output_dir).resolve()
    checkpoint, _ = ro.validate_unified_run(
        checkpoint_path,
        run_dir,
        model_name=MODEL_NAME,
        adaptation=ADAPTATION,
        dataset_jsonl=dataset_jsonl,
        train_years=adapter.TRAIN_YEARS,
        test_years=adapter.TEST_YEARS,
        required_fields=(
            "model_config.dynamic_feature_names",
            "model_config.hidden_size",
            "stats.dynamic_mean",
            "stats.dynamic_std",
            "stats.soil_mean",
            "stats.soil_std",
            "use_constructed",
        ),
    )
    feature_names = list(checkpoint["model_config"]["dynamic_feature_names"])
    if len(feature_names) != 15:
        raise ValueError(f"expected 15 dynamic features, got {len(feature_names)}")
    torch_device = torch.device(device)
    config = dict(checkpoint["model_config"])
    config["vit_pretrained"] = False
    model = build_model(config).to(torch_device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    stats = adapter.stats_from_json(checkpoint["stats"])
    use_constructed = bool(checkpoint.get("use_constructed", True))
    rs_store = adapter.load_rs_store(checkpoint["model_config"])
    dataset = adapter.build_dataset(dataset_jsonl, data_root, adapter.TEST_YEARS, load_images=rs_store is None)
    loader = adapter.build_loader(
        dataset, stats, batch_size, False, use_constructed=use_constructed, num_workers=0, rs_store=rs_store,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    county_vsn_rows = []
    county_spatial_rows = []
    temporal_total = torch.zeros(6, 6, dtype=torch.float64)
    temporal_count = torch.zeros(6, 6, dtype=torch.long)
    diagnostics = {
        "sample_count": 0,
        "invalid_sample_ids": [],
        "vsn_nan_count": 0,
        "spatial_nan_count": 0,
        "temporal_nan_count": 0,
        "vsn_weight_sum_max_abs_error": 0.0,
        "spatial_weight_sum_max_abs_error": 0.0,
        "temporal_weight_sum_max_abs_error": 0.0,
    }

    for batch in loader:
        _, attention, aux = model(**adapter.model_inputs(batch, torch_device))
        sample_valid = aux["sample_valid"].cpu()
        diagnostics["invalid_sample_ids"].extend(
            sid for sid, valid in zip(batch["sample_id"], sample_valid.tolist()) if not valid
        )
        vsn = monthly_vsn_summary(
            aux["grid_vsn_weights"],
            batch["month_ids"].to(torch_device),
            batch["grid_mask"].to(torch_device),
            batch["seq_lens"].to(torch_device),
        ).cpu()
        month_valid = aux["month_valid"].cpu()
        spatial_weights = aux["spatial_weights"].cpu()
        spatial_valid = (
            month_valid[:, :, None]
            & batch["ag_grid_mask"].bool()
            & batch["grid_mask"].bool()[:, None, :]
        )
        concentration = spatial_concentration(spatial_weights, spatial_valid)

        raw_vsn = aux["grid_vsn_weights"][:, :, 0, :].detach().cpu()
        raw_vsn_valid = (
            batch["grid_mask"].reshape(-1, 1)
            & (torch.arange(raw_vsn.shape[1])[None, :] < batch["seq_lens"].repeat_interleave(batch["grid_mask"].shape[1])[:, None])
        )
        if raw_vsn_valid.any():
            error = (raw_vsn.sum(dim=-1)[raw_vsn_valid] - 1.0).abs().max().item()
            diagnostics["vsn_weight_sum_max_abs_error"] = max(
                diagnostics["vsn_weight_sum_max_abs_error"], error
            )
        if spatial_valid.any():
            error = (spatial_weights.sum(dim=-1)[month_valid] - 1.0).abs().max().item()
            diagnostics["spatial_weight_sum_max_abs_error"] = max(
                diagnostics["spatial_weight_sum_max_abs_error"], error
            )
        if month_valid.any():
            error = (attention.detach().cpu().sum(dim=-1)[month_valid] - 1.0).abs().max().item()
            diagnostics["temporal_weight_sum_max_abs_error"] = max(
                diagnostics["temporal_weight_sum_max_abs_error"], error
            )
        diagnostics["vsn_nan_count"] += int(torch.isnan(vsn).sum().item())
        diagnostics["spatial_nan_count"] += int(
            sum(torch.isnan(value).sum().item() for value in concentration.values())
        )
        diagnostics["temporal_nan_count"] += int(torch.isnan(attention).sum().item())

        attention_cpu = attention.detach().cpu().double()
        causal = torch.tril(torch.ones(6, 6, dtype=torch.bool))
        pair_valid = month_valid[:, :, None] & month_valid[:, None, :] & causal[None]
        temporal_total += (attention_cpu * pair_valid).sum(dim=0)
        temporal_count += pair_valid.sum(dim=0)

        for index, (sample_id, fips, year) in enumerate(
            zip(batch["sample_id"], batch["fips"], batch["year"].tolist())
        ):
            for month_index, month in enumerate(MONTHS):
                vsn_row = {
                    "sample_id": sample_id,
                    "fips": fips,
                    "year": int(year),
                    "month": month,
                    "month_valid": bool(month_valid[index, month_index]),
                }
                vsn_row.update({
                    name: float(vsn[index, month_index, feature_index])
                    for feature_index, name in enumerate(feature_names)
                })
                county_vsn_rows.append(vsn_row)
                spatial_row = {
                    "sample_id": sample_id,
                    "fips": fips,
                    "year": int(year),
                    "month": month,
                    "month_valid": bool(month_valid[index, month_index]),
                    "valid_grid_count": int(spatial_valid[index, month_index].sum()),
                }
                spatial_row.update({
                    name: float(value[index, month_index])
                    for name, value in concentration.items()
                })
                county_spatial_rows.append(spatial_row)
        diagnostics["sample_count"] += len(batch["sample_id"])

    county_vsn = pd.DataFrame(county_vsn_rows)
    county_spatial = pd.DataFrame(county_spatial_rows)
    long_vsn = county_vsn.melt(
        id_vars=["sample_id", "fips", "year", "month", "month_valid"],
        value_vars=feature_names,
        var_name="feature",
        value_name="weight",
    )
    monthly_vsn = _summary_rows(long_vsn, ["month", "feature"], ["weight"])
    monthly_vsn = monthly_vsn.drop(columns="metric")
    spatial_summary = _summary_rows(
        county_spatial,
        ["month"],
        ["entropy", "normalized_entropy", "max_weight", "effective_grids", "valid_grid_count"],
    )
    temporal_mean = temporal_total / temporal_count.clamp_min(1)
    temporal_mean[temporal_count == 0] = float("nan")
    temporal_rows = [
        {
            "query_month": MONTHS[query],
            "key_month": MONTHS[key],
            "mean_attention": float(temporal_mean[query, key]),
            "valid_pair_count": int(temporal_count[query, key]),
            "is_causal": key <= query,
        }
        for query in range(6)
        for key in range(6)
    ]
    temporal_frame = pd.DataFrame(temporal_rows)

    county_vsn.to_csv(output_dir / "county_month_vsn.csv", index=False)
    monthly_vsn.to_csv(output_dir / "monthly_vsn_summary.csv", index=False)
    county_spatial.to_csv(output_dir / "county_month_spatial.csv", index=False)
    spatial_summary.to_csv(output_dir / "spatial_attention_summary.csv", index=False)
    temporal_frame.to_csv(output_dir / "temporal_attention_matrix.csv", index=False)

    top_features = (
        long_vsn.groupby("feature")["weight"].mean().sort_values(ascending=False).head(5)
    )
    diagnostics.update({
        "checkpoint": str(checkpoint_path),
        "dataset_jsonl": str(Path(dataset_jsonl).resolve()),
        "data_root": str(Path(data_root).resolve()),
        "device": str(torch_device),
        "feature_names": feature_names,
        "county_month_rows": len(county_vsn),
        "top_features_full_season": {
            name: float(value) for name, value in top_features.items()
        },
        "font_file": _configure_chinese_font(),
    })
    with (output_dir / "attention_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")

    _plot_outputs(
        county_vsn, monthly_vsn, county_spatial, temporal_frame, feature_names, output_dir
    )
    return diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("TFT 2022 attention analysis")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset-jsonl",
        default=str(ro.PROJECT_ROOT / "train_dataset" / "dataset.jsonl"),
    )
    parser.add_argument("--data-root", default=str(DATA_ROOT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    export_attention_analysis(
        args.checkpoint,
        args.dataset_jsonl,
        args.data_root,
        args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
    )
