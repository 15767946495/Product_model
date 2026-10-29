"""GNN-RNN 固定配置（规范 10.4）。

主配置预先固定 aggregator="mean"；"pool" 只作预声明消融，不得根据 2022 测试结果选择。
OOD 论文只公开 GraphSAGE 4 层、dropout 0、aggregator mean/pool；其余参数取自官方
gomes-lab/GNN-RNN 的 run_train.sh（corn，z_dim=64、bs=32、lr=1e-4、wd=1e-5、cosine T0=100、
eta_min=1e-6、100 epochs、logcosh 损失只作用于最后一年 c1=0/c2=1），属于本项目显式定义。
"""

MODEL_NAME = "gnn_rnn"
ADAPTATION = "GNN-RNN, CropNet adaptation (weather + soil + Census adjacency, no Sentinel-2)"

AGGREGATORS = ("mean", "pool")
MAIN_AGGREGATOR = "mean"
ABLATION_AGGREGATOR = "pool"

DEFAULTS = {
    "aggregator": MAIN_AGGREGATOR,
    "n_layers": 4,
    "dropout": 0.0,
    "z_dim": 64,
    "batch_size": 32,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "epochs": 100,
    "T0": 100,
    "T_mult": 2,
    "eta_min": 1e-6,
}

INPUTS = ("weather_daily_county_mean", "soil", "census_adjacency")
