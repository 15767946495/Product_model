"""统一运行输出：output_root/<model>/<run_id>/ 下的 config、checkpoint、指标和元数据。"""

import datetime
import json
import os
from pathlib import Path

import torch

FINAL_CHECKPOINT = "checkpoint-final.pth"
LATEST_CHECKPOINT = "checkpoint-latest.pth"


def default_run_id():
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def make_run_dir(output_root, model_name, run_id):
    run_dir = Path(output_root) / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, set)):
        return list(value)
    return value


def write_json(path, payload):
    """原子写入，避免中断留下半个 JSON。"""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=_jsonable) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def update_json(path, **fields):
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    payload.update(fields)
    write_json(path, payload)


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=_jsonable) + "\n")


def args_config(args):
    return {k: _jsonable(v) for k, v in sorted(vars(args).items())}


def save_final_checkpoint(run_dir, model_without_ddp, *, epoch, model_config, train_years):
    """训练结束无条件写 final checkpoint；不包含任何验证指标或 best 选择状态。"""
    path = Path(run_dir) / FINAL_CHECKPOINT
    tmp = path.with_name(path.name + ".tmp")
    torch.save({
        "model": model_without_ddp.state_dict(),
        "epoch": epoch,
        "model_config": model_config,
        "train_years": list(train_years),
    }, tmp)
    os.replace(tmp, path)
    return path


def add_common_args(parser, *, model_name):
    """统一数据、输出、优化器和分布式参数；不再接受旧 root_dir / manifest 参数。"""
    from dataset_adapter import DATA_ROOT, DEFAULT_DATASET_JSONL

    parser.add_argument('--dataset-jsonl', dest='dataset_jsonl', default=str(DEFAULT_DATASET_JSONL),
                        help='unified CropNet JSONL (train_dataset/dataset.jsonl)')
    parser.add_argument('--data-root', dest='data_root', default=str(DATA_ROOT),
                        help='root that HDF5 paths in the JSONL are relative to')
    parser.add_argument('--output_root', default='./output_dir', help='runs go to output_root/<model>/<run_id>/')
    parser.add_argument('--run_id', default=None, help='defaults to a timestamp')
    parser.set_defaults(model_name=model_name)

    parser.add_argument('--embed_dim', default=512, type=int, help='embed dimensions')
    parser.add_argument('--model_pvt', default='pvt_tiny', type=str, help='PVT backbone name')
    parser.add_argument('--num_heads', default=8, type=int, help='heads of the PVT-SimCLR multi-modal transformer')
    parser.add_argument('--mm_depth', default=2, type=int, help='depth of the PVT-SimCLR multi-modal transformer')
    parser.add_argument('--no_pvt_pretrained', action='store_true', help='do not load ImageNet PVT weights')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR', help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint-latest.pth')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--save_freq', type=int, default=5, help='epochs between checkpoint-latest.pth writes')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


def prepare_run(args):
    """创建 run 目录并写 config.json；args.output_dir 指向 run 目录（misc.save_model 使用）。"""
    import util.misc as misc

    args.run_id = args.run_id or default_run_id()
    run_dir = Path(args.output_root) / args.model_name / args.run_id
    args.output_dir = str(run_dir)
    if misc.is_main_process():
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "config.json", args_config(args))
    return run_dir
