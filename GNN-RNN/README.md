# GNN-RNN（CropNet 统一数据适配）

输入为县级天气、土壤和 Census 县界图，不使用 Sentinel-2。来源与本地修改见 [UPSTREAM.md](UPSTREAM.md)。

## 协议

- 训练：2017–2021 五年窗口，预测 2021 单产。
- 测试：2018–2022 五年窗口，预测 2022 单产，只评估 `checkpoint-final.pth`。
- 两个窗口的县集合、节点顺序和边完全一致，边只来自 `neighbor_fips` 加自环。
- 标准化统计量（天气、土壤、标签）只来自训练窗口。2022 单产只作为最终指标的 `y_true`，不进入节点特征或图消息。
- 主配置的 aggregator 固定为 `mean`；`pool` 是预先声明的消融，不能根据 2022 结果挑选。

## 运行

```bash
./GNN-RNN/run_train.sh
```

脚本默认 `EPOCHS=50`、`GPU 2`，读取 `train_dataset/dataset.jsonl`，输出根为 `GNN-RNN/output_dir`。可覆盖 `EPOCHS`、`RUN_ID`、`OUTPUT_ROOT`、`NUM_WORKERS`、`BATCH_SIZE`、`AGGREGATOR` 和 `DRY_RUN`；例如 `DRY_RUN=1 EPOCHS=1 RUN_ID=smoke ./GNN-RNN/run_train.sh`。脚本先训练再执行最终测试。

## 文件

- `dataset_adapter.py`：从 `CropNetDataset` 构造两个窗口，校验图结构，计算训练统计量。
- `graph.py`：Census 边集合校验。
- `model.py`：1D CNN 加土壤编码，接 GraphSAGE（纯 PyTorch 实现），再接 LSTM。
- `train.py` / `infer.py`：训练入口和最终测试入口。
- `config.py`：固定超参数及其来源。
