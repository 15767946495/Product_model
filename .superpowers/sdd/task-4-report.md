# Task 4 报告：最终 data-only 集成验证

## 状态

- 状态：通过。
- 本次仅更新 `tests/test_tft_ag_data.py`，并写入五州 AG runtime 的只读验证证据。
- 未实例化 TFT、未调用 TFT `forward`、未训练、未推理。
- 未修改 `TFT_model/models.py`、`TFT_model/train.py`、`TFT_model/infer.py`、Sentinel 原始文件或八州 `task8-active` runtime。

## 五州协议与审计统计

- 协议状态州：Illinois、Iowa、Louisiana、Mississippi、New York。
- 固定年份 split：train=2017--2020，val=2021，test=2022。
- 现有真实输入事实：`ag_integrity.json` 审计输入为 532 行，实际只有 Illinois；没有从八州输入中伪造或补齐其余四州。
- train：359 输入，355 有效，4 无效。
- val：79 输入，78 有效，1 无效。
- test：94 输入，93 有效，1 无效。
- 总计：526 有效，6 无效。
- 无效原因：4 个样本缺日期组，2 个样本缺 FIPS HDF5 group。
- 现有有效 manifest：train=355、val=78、test=93；每行均为两个 AG 季度路径。

## 真实 AG 样本验证

从 `cropnet-five-state/manifests/train.jsonl` 首条有效协议行读取真实样本：

- FIPS：`17001`
- Year：`2017`
- State：`illinois`
- County：`adams`
- seed：`0`
- transform：eval 模式 `CenterCrop(224)` 加 ImageNet 风格 AG Normalize，均值 `[0.466, 0.471, 0.380]`，标准差 `[0.195, 0.194, 0.192]`
- 12 个日期：`04-01, 04-15, 05-01, 05-15, 06-01, 06-15, 07-01, 07-15, 08-01, 08-15, 09-01, 09-15`
- tensor shape：`[12, 24, 3, 224, 224]`
- dtype：`torch.float32`
- `grid_count`：`24`
- 输出包含 `FIPS`、`Year`、`State`、`County`、`grid_count`、`ag_dates` 等 metadata。
- 输出不包含 `weather` 或 `label`，未产生对应加载副作用。

## 模型边界

- `TFTEncoderForYieldPrediction.forward` 签名保持为：`self, grid_feats, grid_coords, grid_mask, soil_feats, seq_lens`。
- 测试通过 monkeypatch 确认构造 TFT 不是 `AgricultureImageDataset` 的依赖。
- 测试仅检查模型类接口和数据集输出，没有构造模型或执行 forward。
- `TFT_model/models.py`、训练入口和推理入口在本次工作树中无 diff。

## 验证命令

- `/root/miniconda3/envs/hqx/bin/python -m pytest -q`
- `/root/miniconda3/envs/hqx/bin/python -m py_compile cropnet_protocol.py TFT_model/data.py tools/audit_tft_ag_data.py mmst_vit/config.py mmst_vit/manifest.py tests/test_tft_ag_data.py`

两项命令均通过；全量 pytest 统计为 `165 passed`，另有 1 条环境层面的 `pynvml` 弃用警告。

## 只读审计产物

已更新：`/data/raid0/hqx/Product_model_runtime/cropnet-five-state/audit/protocol.json`。

新增内容包括代表性样本的 12 日期、shape、dtype、metadata、transform、seed、TFT boundary、未训练/未推理标记及源文件 SHA256。源 AG HDF5 文件仅被读取，代表样本两个季度 SHA256 为：

- `Agriculture_17_IL_2017-04-01_2017-06-30.h5`: `3e3aacdb91d5a8ffd0292013d30873887634a3b2bf6193b8e74af870a8fd1393`
- `Agriculture_17_IL_2017-07-01_2017-09-30.h5`: `742bf0fc3f9e62047143c8ddd4991bf75d06e8847c3838d0103a31f127c5e`

## Concerns

- “五州”是协议范围，不是本次真实输入的覆盖证明：当前 `task8-active` 共享 JSONL 和生成的五州 manifest 实际只有 Illinois。Iowa、Louisiana、Mississippi、New York 没有有效行，因此本次不能声称完成五州真实样本覆盖。
- runtime `ag_integrity.json` 与声明的五州协议在州覆盖上存在事实差异；本次只在 `protocol.json` 和报告中留痕，没有修改八州 runtime 或伪造统计。
- 工作树开始时已有与本任务无关的删除和未跟踪文件：`_txt/*`、`BaseLine_Model/output`、`DataSrc`；未修改或清理这些文件。
