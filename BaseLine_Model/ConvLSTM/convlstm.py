"""
ConvLSTM 基线(MMST-ViT 复现,去遥感):县内网格(按经纬度排序)为 1D 空间场,
ConvLSTM 单元(卷积门控)沿时间推进,末态按有效网格掩码平均 + 土壤 -> 单产。

数据/指标口径与 TFT 一致:DeepCropNet 9 玉米带州,逐日 275 步,网格不截断,
训练 <2021,验证 2021,原始单产。

用法:
  python convlstm.py
  python convlstm.py --epochs 150
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from common import data as D
from common.train import run_training


class ConvLSTMCell(nn.Module):
    """1D 卷积 LSTM 单元:输入 (B,C_in,G) + 状态 (B,C,G) -> 新状态。G 为空间轴(网格)。"""

    def __init__(self, in_channels, hidden_channels, kernel=3):
        super().__init__()
        self.hidden = hidden_channels
        self.conv = nn.Conv1d(in_channels + hidden_channels, 4 * hidden_channels,
                              kernel_size=kernel, padding=kernel // 2)

    def forward(self, x, state):
        h, c = state
        z = self.conv(torch.cat([x, h], dim=1))   # (B, 4C, G)
        i, f, o, g = torch.split(z, self.hidden, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTM(nn.Module):
    """输入 (B,G,275,11) + 掩码 (B,G) + (B,7)。"""

    def __init__(self, hidden=32, soil_dim=D.SOIL_DIM, kernel=3):
        super().__init__()
        self.hidden = hidden
        self.cell = ConvLSTMCell(D.N_FEATS, hidden, kernel)
        self.head = nn.Sequential(nn.Linear(hidden + soil_dim, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, grid_weather, grid_mask, soil):
        B, G, T, F = grid_weather.shape
        h = torch.zeros(B, self.hidden, G, device=grid_weather.device)
        c = torch.zeros(B, self.hidden, G, device=grid_weather.device)
        x = grid_weather.permute(0, 2, 3, 1)       # (B,T,F,G) 特征为通道,网格为空间轴
        for t in range(T):
            h, c = self.cell(x[:, t], (h, c))
        mask = grid_mask.unsqueeze(1).to(h.dtype)  # (B,1,G)
        pooled = (h * mask).sum(dim=2) / mask.sum(dim=2).clamp(min=1)   # (B,hidden)
        return self.head(torch.cat([pooled, soil], dim=-1))


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
    Ytr = D.to_tensor(tr["y_std"], device).unsqueeze(1)
    Str = D.to_tensor(tr["soil"], device)
    Sva = D.to_tensor(va["soil"], device)
    N = len(Ytr)

    model = ConvLSTM().to(device)
    print(f"[模型] ConvLSTM 参数: {sum(p.numel() for p in model.parameters()):,}")

    # G 分桶:按网格数排序分桶,同桶 batch,避免大 G 县(≤134)拖慢整批;不截断,仅分组
    gs = np.array([gw.shape[0] for gw in tr["grid_weather"]])
    gorder = np.argsort(gs)
    buckets = [gorder[i:i + args.batch_size] for i in range(0, N, args.batch_size)]

    def make_batches():
        order = []
        for b in np.random.permutation(len(buckets)):
            bucket = buckets[b]
            order.extend(bucket[np.random.permutation(len(bucket))].tolist())
        return [order[i:i + args.batch_size] for i in range(0, N, args.batch_size)]

    def step_fn(model, opt, loss_fn, idx):
        idx_t = torch.tensor(idx, device=device)
        gwb, gmb = D.grid_batch(tr, idx, device)
        out = model(gwb, gmb, Str[idx_t])
        loss = loss_fn(out, Ytr[idx_t])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        return loss.item()

    def eval_pred_fn(model):
        return D.eval_convlstm(model, va, Sva, device)

    run_training(model, data, device, args, "convlstm", out_dir,
                 make_batches, step_fn, eval_pred_fn)


if __name__ == "__main__":
    main()
