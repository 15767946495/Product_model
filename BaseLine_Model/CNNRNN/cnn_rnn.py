"""
CNN-RNN 基线(MMST-ViT 复现,去遥感):天气 1D CNN(局部时间模式)+ LSTM(长程依赖)+ 土壤。
输入: 逐日 275 步 × 11 特征(县均值,与 TFT 同粒度)+ 7 维土壤 → 单产 bu/ac。

数据/指标口径与 TFT 一致:DeepCropNet 9 玉米带州,训练 <2021,验证 2021,原始单产。

用法:
  python cnn_rnn.py
  python cnn_rnn.py --epochs 200 --lr 3e-3
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from common import data as D
from common.train import run_training


class CNNRNN(nn.Module):
    """天气 1D CNN(时间)+ LSTM + 土壤。输入 (B,275,11) + (B,7)。"""

    def __init__(self, hidden=64, soil_dim=D.SOIL_DIM):
        super().__init__()
        self.conv1 = nn.Conv1d(D.N_FEATS, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)          # 275 -> 137 -> 68
        self.lstm = nn.LSTM(64, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + soil_dim, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, weather, soil):
        x = weather.transpose(1, 2)          # (B,11,275)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)                     # (B,64,68)
        x = x.transpose(1, 2)                # (B,68,64)
        out, _ = self.lstm(x)
        h = out[:, -1]                       # (B,64)
        return self.head(torch.cat([h, soil], dim=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_year", type=int, default=2021)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--force_prep", action="store_true")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) if args.out_dir else D.OUT_DIR

    data = D.prepare(args.val_year, out_dir, args.force_prep)
    tr, va = data["train"], data["val"]
    Xtr = D.to_tensor(tr["weather"], device)
    Ytr = D.to_tensor(tr["y_std"], device).unsqueeze(1)
    Str = D.to_tensor(tr["soil"], device)
    Xva = D.to_tensor(va["weather"], device)
    Sva = D.to_tensor(va["soil"], device)
    N = len(Xtr)

    model = CNNRNN().to(device)
    print(f"[模型] CNN-RNN 参数: {sum(p.numel() for p in model.parameters()):,}")

    def make_batches():
        perm = torch.randperm(N).tolist()
        return [perm[i:i + args.batch_size] for i in range(0, N, args.batch_size)]

    def step_fn(model, opt, loss_fn, idx):
        idx_t = torch.tensor(idx, device=device)
        out = model(Xtr[idx_t], Str[idx_t])
        loss = loss_fn(out, Ytr[idx_t])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        return loss.item()

    def eval_pred_fn(model):
        return model(Xva, Sva).squeeze(1).detach().cpu().numpy()

    run_training(model, data, device, args, "cnn_rnn", out_dir,
                 make_batches, step_fn, eval_pred_fn)


if __name__ == "__main__":
    main()
