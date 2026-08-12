"""
GNN-RNN 基线(MMST-ViT 复现,去遥感):县为图节点(kNN 邻接,按县质心距离),
GCN 一次性聚合邻县天气(einsum 向量化)+ LSTM 学时间依赖,末态 + 土壤 -> 单产。

数据/指标口径与 TFT 一致:DeepCropNet 9 玉米带州,逐日 275 步,
训练 <2021,验证 2021,原始单产。图要求全批训练(邻接关系需整图)。

注:训练图(训练县)与验证图(验证县)各自独立建图。

用法:
  python gnn_rnn.py
  python gnn_rnn.py --epochs 150
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


class GNNRNN(nn.Module):
    """GCN 聚合 + LSTM。输入 (N,275,11) + 归一化邻接 (N,N) + (N,7)。全批训练。"""

    def __init__(self, hidden=64, soil_dim=D.SOIL_DIM, gnn_hidden=32):
        super().__init__()
        self.gnn = nn.Linear(D.N_FEATS, gnn_hidden)
        self.lstm = nn.LSTM(gnn_hidden, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + soil_dim, 32), nn.ReLU(),
                                  nn.Linear(32, 1))

    def forward(self, weather, adj, soil):
        # GCN 是线性算子:一次性聚合所有时间步(einsum 等价逐步 matmul,避免巨量 autograd 中间节点)
        x = torch.einsum('ij,jtf->itf', adj, weather)      # (N,T,11)
        h_seq = torch.relu(self.gnn(x))                    # (N,T,32)
        out, _ = self.lstm(h_seq)
        h = out[:, -1]                                     # (N,64)
        return self.head(torch.cat([h, soil], dim=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val_year", type=int, default=2021)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--force_prep", action="store_true")
    ap.add_argument("--gnn_k", type=int, default=5)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir) if args.out_dir else D.OUT_DIR

    data = D.prepare(args.val_year, out_dir, args.force_prep, gnn_k=args.gnn_k)
    tr, va = data["train"], data["val"]
    Xtr = D.to_tensor(tr["weather"], device)
    Ytr = D.to_tensor(tr["y_std"], device).unsqueeze(1)
    Str = D.to_tensor(tr["soil"], device)
    Atr = D.to_tensor(tr["adjacency"], device)
    Xva = D.to_tensor(va["weather"], device)
    Sva = D.to_tensor(va["soil"], device)
    Ava = D.to_tensor(va["adjacency"], device)

    model = GNNRNN().to(device)
    print(f"[模型] GNN-RNN 参数: {sum(p.numel() for p in model.parameters()):,}")

    def make_batches():
        return [list(range(len(Ytr)))]          # 全批(图邻接要求整图)

    def step_fn(model, opt, loss_fn, idx):
        out = model(Xtr, Atr, Str)
        loss = loss_fn(out, Ytr)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        return loss.item()

    def eval_pred_fn(model):
        return model(Xva, Ava, Sva).squeeze(1).detach().cpu().numpy()

    run_training(model, data, device, args, "gnn_rnn", out_dir,
                 make_batches, step_fn, eval_pred_fn)


if __name__ == "__main__":
    main()
