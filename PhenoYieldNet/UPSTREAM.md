# PhenoYieldNet 上游来源

- 官方仓库：<https://github.com/roroyo/PhenoYieldNet>，commit `5a14ed7f7882ef6e2d5d4680c373e12da0110147`（2026-10-14 克隆核对）。
- 许可证：CC BY-NC 4.0（见 [LICENSE](LICENSE)，原样复制）。其中 MAE / MAE-ST 派生文件同样是 CC BY-NC 4.0，所以本目录只能用于非商业用途。
- 结果标记：`PhenoYieldNet-unified`。没有作者的原始数据清单、checkpoint 或运行日志，结果不代表作者原始结果。

## 原样引入的文件

| 文件 | 说明 |
| --- | --- |
| `attention.py` | SpatialTransformer、MultiCropTrendAwareTemporalTransformer、MultiModalTransformer |
| `model/models_vit_tensor.py` | SpectralGPT 风格的 ViT 骨干 `vit_base_patch8_128_channel3_frame1_tpatch1` |
| `model/models_PhenoYieldNet.py` | `ViT_multimodal_multicrop`；唯一改动见下 |
| `util/video_vit.py`、`util/pos_embed.py`、`util/lr_sched.py` | 骨干依赖、pos_embed 插值、学习率调度 |
| `loss/contrastive_loss.py` | SimCLR `ContrastiveLoss` |

## 本地修改

- `model/models_PhenoYieldNet.py`：新增可选参数 `crop_growth_cycles`，标记为 `[unified CropNet]`。上游 corn 的时序窗口是 4–12 月，按 12 个月的轴索引；统一数据只有 4–9 月 6 个时相，所以窗口取交集 4–9 月，索引 0–5。
- 数据入口全部替换为 `dataset_adapter.py`，读取 `train_dataset/dataset.jsonl`。上游的 `dataset/*_loader.py`、`configs/build_config_multicrop.py` 和 `configs/multicrop/*.json` 都没有引入。
- 训练入口 `train_da.py`、`train_ft.py` 重写，但保留上游训练逻辑：
  - Stage 1：按 0.2 概率丢帧；每个 (网格, 月) 单图独立做两次 SimCLR 增强（RandomResizedCrop 128、翻转、旋转、高斯模糊 3）；配 9 维天气上下文；ContrastiveLoss；AdamW(0.9, 0.95)，`blr=1e-3`；200 epochs，warmup 40。
  - Stage 2：骨干 `num_classes=2` 后加载 Stage 1 权重；端到端训练，上游 `run_finetune.sh` 没有传 `--frozen_fm`，所以公开代码实际**不冻结**编码器，论文里写的是冻结，这里提供 `--frozen_fm` 开关、默认不冻结；MSE 作用于 log(yield)，即 `--yield_only_loss` 的语义；`blr=3e-4`；epochs 固定为 40，取 `run_finetune.sh` 的值，README 写的是 60；warmup 5。
  - 去掉验证集、`best_metrics` 和按验证指标保存 checkpoint 的逻辑。最终测试只评估 `checkpoint-final.pth`，只用 2022。
- 只实现 PhenoYieldNet-SC（corn）：始终使用 corn 的 crop query 和回归头（索引 1）。四作物参数形状保留，便于对照上游。
- 像素尺度：上游直接把 uint8 转成 float（0–255，不除以 255）。统一 Dataset 输出 [0,1]，adapter 乘回 255。
- `num_grid` 从 64 改为 80：统一数据单县最多 78 个网格，加 1 个 cls token。上游模型没有网格 mask，所以每步固定一个县年，与上游 `batch_size=1` 的县级 DataLoader 一致，不产生 padding。
- 不使用长期气象；`crop_balanced_loss`、`head_lr_balance` 只对多作物有意义，没有引入。

## SpectralGPT+ 权重

| 项 | 值 |
| --- | --- |
| URL | <https://zenodo.org/records/8412455/files/SpectralGPT+.pth?download=1> |
| 本地路径 | `/data/raid0/hqx/pretrained/SpectralGPT+.pth`（1,184,014,163 bytes） |
| MD5 | `aeaa08388f0937e3f07ecbd7757f20dc`（与 Zenodo API 记录一致） |
| SHA-256 | `981730de133f15090978d60a9eded6d04d970174834b4e7e16b93e1d39b8a437` |
| 许可证 | CC-BY-4.0（Zenodo 记录） |
| 骨干配置 | `vit_base_patch8_128_channel3_frame1_tpatch1`：img 128、patch 8、embed 768、depth 3、heads 12 |

按上游 `train_da.py` 的加载逻辑（删除形状不符的键，插值 pos_embed，`strict=False`）实测：

- 成功加载 33 个键：`blocks.{0,1,2}` 的 `norm1/norm2`、`attn.proj`、`mlp.fc1/fc2`，`norm`，`patch_embed.proj.bias`。
- 因形状不符被删除：`patch_embed.proj.weight`。SpectralGPT+ 是 `(768,1,3,8,8)` 的光谱 3D patch，本骨干是 `(768,3,1,8,8)`。
- 缺失键：`pos_embed`；`blocks.*.attn.to_qkv.*`（SpectralGPT+ 中是分离的 `attn.q/k/v`，名字对不上）；`proj_context`、`norm1`、`mlp`、`mm_transformer.*`、`head.*`。
- 意外键 237 个：decoder、`pos_embed_spatial/temporal`、`mask_token`、`blocks.3–11`、分离的 `attn.q/k/v`。

结论：按上游公开逻辑，SpectralGPT+ 只初始化了 3 层 block 的 LayerNorm、注意力输出投影和 MLP。注意力 QKV、patch 投影权重和位置编码都是随机初始化。这是上游代码的行为，本实现保持一致，没有修正。`backbone.load_pretrained` 会把完整报告写到 `pretrained_load_report.json`。
