# GNN-RNN 上游来源

- 结构依据：<https://github.com/gomes-lab/GNN-RNN>，commit `2daa02cde350d61f990d422a6401bc6fd51c5e29`（2026-10-14 克隆核对）。
- 许可证：上游仓库根目录没有 LICENSE 文件。本目录没有复制上游源码，只按 `gnn-rnn/model.py`、`gnn-rnn/train.py` 的结构重新实现。
- CropNet 参数参考：Out-of-Distribution Generalization in Climate-Aware Crop Yield Prediction with Earth Observation Data，<https://arxiv.org/abs/2510.07350>。论文只公开 GraphSAGE 4 层、dropout 0、aggregator mean/pool，没有公开实现。

## 沿用上游的部分

| 上游 | 本实现 |
| --- | --- |
| `SAGE_RNN`：逐年 encoder → 多层 SAGEConv（每层后 ReLU + dropout）→ LSTM → 两层回归头 | `model.GNNRNN`，结构一致 |
| `CNN` encoder：天气 1D CNN（64/128/256/512 通道）→ 80 维，土壤 → 40 维 | `model.WeatherSoilEncoder`，输出 120 维 |
| `loss_fn(mode="logcosh")`，标签按训练均值/方差标准化 | `train.logcosh`，标准化统计量只来自 2021 训练目标 |
| `c1=0, c2=1`：只监督序列最后一年 | 只监督 2021（训练）末年 |
| run_train.sh（corn）：`z_dim=64, bs=32, lr=1e-4, wd=1e-5, cosine T0=100, T_mult=2, eta_min=1e-6, 100 epochs` | `config.DEFAULTS` |

## 本地修改

- 不依赖 DGL：`model.SAGEConv` 用纯 PyTorch 实现 `dgl.nn.SAGEConv` 的 mean / pool 聚合（`fc_self(h) + fc_neigh(agg) + bias`）。
- 不做邻居采样：固定县集合只有 384 个节点，每步做全图前向，损失只取本步目标县。上游用的是 `MultiLayerNeighborSampler([10, 10])`。
- 天气输入：上游是 23 个 PRISM/NLDAS 变量、365 天或 52 周；这里是 CropNet 的 11 个 HRRR 变量、168 天，按县内有效网格取平均。卷积 stride 改为 1，最后加自适应池化，以适配 168 天。
- 土壤输入：上游是 20 个变量 × 6 层深度，用 1D 卷积；统一数据只有 7 个县级标量，改用线性层。
- 不使用管理/作物进度变量（等价于上游 `--no_management`），也不使用 `num_extra_vars`。
- GraphSAGE 4 层、dropout 0 取自 OOD 论文；上游默认是 2 层、dropout 0.5。
- 图：边只来自统一 JSONL 的 `neighbor_fips`（Census 县界接壤），限定在固定县集合内，并加自环。上游使用 `us_adj.pkl`。
- 上游 forward 中的 `y_pad` 实际没有进入网络；本实现完全不接收标签输入。
- 不设验证集、不早停、不按指标选 checkpoint；最终只评估 `checkpoint-final.pth`。
