"""TFT 统一 final checkpoint 的独立 2022 推理与逐月评估。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from TFT_model import dataset_adapter as adapter
from TFT_model.model_adapter import build_model
from train_dataset import run_output as ro
from train_dataset.dataset import DATA_ROOT
from train_dataset.metrics import regression_metrics

MODEL_NAME = "tft"
ADAPTATION = "TFT full multimodal CropNet adaptation"
DEFAULT_CUTOFFS = "04-28,05-28,06-28,07-28,08-28,09-28"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("TFT unified final 2022 test")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--dataset-jsonl", dest="dataset_jsonl",
        default=str(ro.PROJECT_ROOT / "train_dataset" / "dataset.jsonl"),
    )
    parser.add_argument("--data-root", dest="data_root", default=str(DATA_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cutoffs", default=DEFAULT_CUTOFFS)
    return parser


def parse_cutoffs(value: str) -> list[tuple[int, int]]:
    cutoffs = []
    for raw in value.split(","):
        text = raw.strip()
        if not text:
            continue
        try:
            month_text, day_text = text.split("-")
            cutoff = (int(month_text), int(day_text))
        except (TypeError, ValueError) as error:
            raise ValueError(f"invalid cutoff {text!r}; expected MM-28 from 04-28 to 09-28") from error
        if cutoff[0] not in range(4, 10) or cutoff[1] != 28:
            raise ValueError(f"invalid cutoff {text!r}; expected MM-28 from 04-28 to 09-28")
        cutoffs.append(cutoff)
    if not cutoffs:
        raise ValueError("at least one cutoff is required")
    return cutoffs


def predictions_at_cutoffs(
    pred_dist_all: torch.Tensor,
    monthly_month_ids: torch.Tensor,
    month_valid: torch.Tensor,
    cutoffs: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = []
    validity = []
    batch_index = torch.arange(pred_dist_all.shape[0], device=pred_dist_all.device)
    for month, _ in cutoffs:
        eligible = month_valid & (monthly_month_ids <= month)
        indices = eligible.long().cumsum(dim=1).argmax(dim=1)
        valid = eligible.any(dim=1)
        predictions.append(pred_dist_all[batch_index, indices])
        validity.append(valid)
    return torch.stack(predictions, dim=1), torch.stack(validity, dim=1)


def cutoff_regression_metrics(y_true, y_pred) -> dict:
    if len(y_true) < 2:
        return {
            "rmse": None,
            "mae": None,
            "r2": None,
            "corr": None,
            "corr2": None,
            "count": len(y_true),
        }
    return regression_metrics(y_true, y_pred)


@torch.no_grad()
def evaluate_final_checkpoint(
    checkpoint_path,
    dataset_jsonl,
    data_root,
    output_dir,
    *,
    device="cuda",
    batch_size=4,
    num_workers=0,
    cutoffs=DEFAULT_CUTOFFS,
) -> dict:
    output_dir = Path(output_dir).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    run_dir = checkpoint_path.parent
    if output_dir != run_dir or any(
            not (run_dir / name).is_file() for name in ("config.json", "run_metadata.json", ro.FINAL_CHECKPOINT)):
        raise ValueError("checkpoint must belong to an existing unified run directory")
    # 逐年留一：测试年份记录在训练时的 config.json；旧运行没有该字段，即原协议的2022年
    run_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    train_years, test_years = adapter.split_years(int(run_config.get("test_year", adapter.TEST_YEARS[0])))
    checkpoint, _ = ro.validate_unified_run(
        checkpoint_path,
        output_dir,
        model_name=MODEL_NAME,
        adaptation=ADAPTATION,
        dataset_jsonl=dataset_jsonl,
        train_years=train_years,
        test_years=test_years,
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
    device = torch.device(device)
    inference_config = dict(checkpoint["model_config"])
    inference_config["vit_pretrained"] = False
    model = build_model(inference_config).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    stats = adapter.stats_from_json(checkpoint["stats"])
    use_constructed = bool(checkpoint.get("use_constructed", True))
    rs_store = adapter.load_rs_store(checkpoint["model_config"])
    dataset = adapter.build_dataset(dataset_jsonl, data_root, test_years, load_images=rs_store is None)
    loader = adapter.build_loader(
        dataset, stats, batch_size, False,
        use_constructed=use_constructed,
        num_workers=num_workers,
        rs_store=rs_store,
    )
    cutoff_list = parse_cutoffs(cutoffs) if isinstance(cutoffs, str) else list(cutoffs)
    cutoff_values = {cutoff: ([], []) for cutoff in cutoff_list}
    rows = []

    for batch in loader:
        pred, _, aux = model(**adapter.model_inputs(batch, device))
        sample_valid = aux["sample_valid"]
        if not bool(sample_valid.all()):
            invalid = [sid for sid, valid in zip(batch["sample_id"], sample_valid.tolist()) if not valid]
            raise RuntimeError(f"samples have no valid remote-sensing month: {invalid}")
        offset = batch.get("county_offset")
        offset = torch.zeros(pred.shape[0]) if offset is None else offset.float()
        final_pred = torch.exp(pred[:, 0].cpu() + offset)
        labels = batch["labels"].float().cpu()
        for sid, fips, year, label, prediction in zip(
            batch["sample_id"], batch["fips"], batch["year"].tolist(), labels.tolist(), final_pred.tolist()
        ):
            rows.append({
                "sample_id": sid,
                "fips": fips,
                "year": int(year),
                "y_true": float(label),
                "y_pred": float(prediction),
            })

        cutoff_pred, cutoff_valid = predictions_at_cutoffs(
            aux["pred_dist_all"], aux["monthly_month_ids"], aux["month_valid"], cutoff_list
        )
        cutoff_pred = torch.exp(cutoff_pred[..., 0].cpu() + offset[:, None])
        cutoff_valid = cutoff_valid.cpu()
        for index, cutoff in enumerate(cutoff_list):
            valid = cutoff_valid[:, index]
            cutoff_values[cutoff][0].extend(labels[valid].tolist())
            cutoff_values[cutoff][1].extend(cutoff_pred[valid, index].tolist())

    metrics = ro.finish_test(
        output_dir, rows, dataset.sample_ids,
        checkpoint_path=checkpoint_path, years=test_years,
    )
    cutoff_metrics = {}
    for cutoff in cutoff_list:
        truth, prediction = cutoff_values[cutoff]
        key = f"{cutoff[0]:02d}-{cutoff[1]:02d}"
        cutoff_metrics[key] = cutoff_regression_metrics(truth, prediction)
    ro.write_json(Path(output_dir) / "cutoff_metrics.json", cutoff_metrics)
    return metrics


if __name__ == "__main__":
    args = build_parser().parse_args()
    evaluate_final_checkpoint(
        args.checkpoint,
        args.dataset_jsonl,
        args.data_root,
        Path(args.checkpoint).resolve().parent,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        cutoffs=args.cutoffs,
    )
