"""GNN-RNN 最终测试入口：只评估 checkpoint-final.pth，只用 2018--2022 窗口预测 2022。

    python GNN-RNN/infer.py --checkpoint output_dir/gnn_rnn/<run_id>/checkpoint-final.pth
"""

import argparse
from pathlib import Path

from train import evaluate_final_checkpoint
from train_dataset import run_output as ro
from train_dataset.dataset import DATA_ROOT


def build_parser():
    parser = argparse.ArgumentParser("GNN-RNN final 2022 test")
    parser.add_argument("--checkpoint", required=True, help="output_dir/gnn_rnn/<run_id>/checkpoint-final.pth")
    parser.add_argument("--dataset-jsonl", dest="dataset_jsonl",
                        default=str(ro.PROJECT_ROOT / "train_dataset" / "dataset.jsonl"))
    parser.add_argument("--data-root", dest="data_root", default=str(DATA_ROOT))
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    evaluate_final_checkpoint(args.checkpoint, args.dataset_jsonl, args.data_root,
                              Path(args.checkpoint).resolve().parent, device=args.device)
