"""PhenoYieldNet-unified 最终测试入口：只评估 checkpoint-final.pth，只读 2022。

    python PhenoYieldNet/infer.py --checkpoint output_dir/phenoyieldnet/<run_id>/checkpoint-final.pth
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train_dataset import run_output as ro
from train_dataset.dataset import DATA_ROOT
from train_ft import evaluate_final_checkpoint


def build_parser():
    parser = argparse.ArgumentParser("PhenoYieldNet-unified final 2022 test")
    parser.add_argument("--checkpoint", required=True, help="output_dir/phenoyieldnet/<run_id>/checkpoint-final.pth")
    parser.add_argument("--dataset-jsonl", dest="dataset_jsonl",
                        default=str(ro.PROJECT_ROOT / "train_dataset" / "dataset.jsonl"))
    parser.add_argument("--data-root", dest="data_root", default=str(DATA_ROOT))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    evaluate_final_checkpoint(args.checkpoint, args.dataset_jsonl, args.data_root,
                              Path(args.checkpoint).resolve().parent, device=args.device,
                              num_workers=args.num_workers)
