# WeatherFormer 时空注意力实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将空间 attention 改为每个时间步具有独立 CLS query 的 WeatherFormer 四槽时空加性 cross-attention，并将消融实验收敛为 `mean` 与 `additive` 两组。

**Architecture:** 保留逐特征网格聚合和后续时序 VSN。`SpatialAttentionAggregator` 接收 `(B,G,T,H)` 内容 token、网格中心坐标和有效网格 mask，计算有效网格中心作为 CLS 坐标，使用统一的 `PE(t,lat,lon)` 构造 CLS query 与 grid key；value 只使用内容 token。删除空间 RoPE 和 `spatial_encoding` 参数，升级 checkpoint contract，当前 checkpoint 不兼容。

**Tech Stack:** Python 3.10, PyTorch, NumPy, unittest, existing `product` Conda environment.

## Global Constraints

- 时间使用相对日序 `t = 0, 1, ..., T - 1`。
- WeatherFormer 四槽公式固定为 `sin(t)`, `cos(t)`, `sin(lat)`, `cos(lon)`，频率为 `10000^(-4i/d)`。
- CLS 空间坐标是有效网格中心坐标的掩码均值。
- CLS 只作为 query，不进入 key/value；空间权重形状为 `(B,T,G)`。
- value 路径不加入位置编码。
- `mean` 不使用位置编码；`attention` 固定使用 WeatherFormer 时空加性编码。
- 删除 `_rotate_half_pairwise`、`_apply_rotary`、`_grid_axis_boundaries`、`_grid_rope_cos_sin`、RoPE CLI/配置和 `spatial_encoding` 参数。
- 模型 contract version 升级为 `4`；旧 checkpoint 和旧 `model_hparams.json` 必须拒绝加载。
- 网格缓存版本 3 和 `coord_type='grid_center'` 保持不变，不重建缓存。
- 15 个动态特征继续分别经过空间聚合，再进入时序 VSN。

---

### Task 1: 更新空间契约测试

**Files:**
- Modify: `tests/test_spatial_contracts.py`
- Modify: `tests/test_ablation_rope.py`

**Interfaces:**
- Tests define the required `SpatialAttentionAggregator._st_pe(coords, t_idx)` output and `forward_weights(tokens, coords, grid_mask)` behavior.
- Tests define `constructed_combo(0)` and `constructed_combo(1)` as the only two ablation variants.

- [ ] **Step 1: Replace obsolete RoPE tests with failing WeatherFormer tests.**

Add tests with these exact behaviors:

```python
def test_weatherformer_four_slot_encoding(self):
    aggregator = models.SpatialAttentionAggregator(hidden_size=8, dropout=0.0)
    coords = torch.tensor([[[10.0, 20.0]]])
    t = torch.tensor([0, 1], dtype=torch.long)
    pe = aggregator._st_pe(coords, t)
    freq = 10000.0 ** torch.tensor([0.0, -0.5])
    lat = torch.tensor(10.0 * torch.pi / 180.0)
    lon = torch.tensor(20.0 * torch.pi / 180.0)
    expected = torch.empty(1, 1, 2, 8)
    expected[..., 0::4] = torch.sin(t.float().view(1, 1, 2, 1) * freq)
    expected[..., 1::4] = torch.cos(t.float().view(1, 1, 2, 1) * freq)
    expected[..., 2::4] = torch.sin(lat * freq)
    expected[..., 3::4] = torch.cos(lon * freq)
    self.assertTrue(torch.allclose(pe, expected, atol=1e-6))
```

Use a direct vectorized expected tensor for all `d=8` slots so the test also verifies the frequency exponent and slot ordering. Add tests that:

- `hidden_size=6` raises a message containing `4`;
- `forward_weights` returns `(B,T,H)` and `(B,T,G)`, masks padding, and weights sum to one;
- changing `t` changes both CLS and grid positional encodings;
- the CLS coordinate equals the masked mean of valid grid coordinates, excluding padding;
- value 不含位置编码：将 `W_q`、`W_k` 的 weight/bias 全置零以产生均匀权重，将 `W_v` 设置为单位映射；对相同 token 和 mask、两组不同 coords 调用 `forward_weights`，断言两个输出完全相同。若位置误加到 value，该断言会失败；
- `MODEL_CONTRACT_VERSION == 4` and `CausalScaledDotProductAttention(..., use_rope=True)` raises `TypeError`.

Remove tests for `_spatial_pe`, 2D RoPE, and `spatial_encoding` constructor arguments.

- [ ] **Step 2: Update ablation tests to the two-combo contract.**

Replace the three-combo assertion with:

```python
self.assertEqual(
    [ABLATION_ROPE.constructed_combo(i) for i in range(2)],
    [
        {"use_constructed": True, "spatial_mode": "mean"},
        {"use_constructed": True, "spatial_mode": "attention"},
    ],
)
```

Change the GPU-count test from five GPUs to two GPUs and retain duplicate-combo and constructed-mode validation.

- [ ] **Step 3: Run the focused tests and verify they fail against the old implementation.**

Run:

```bash
conda run -n product python -m unittest tests/test_spatial_contracts.py tests/test_ablation_rope.py -v
```

Expected: failures for the new `_st_pe`/time-aware query, contract version, and two-combo assertions, with no test collection errors.

- [ ] **Step 4: Commit the red tests.**

```bash
git add tests/test_spatial_contracts.py tests/test_ablation_rope.py
git commit -m "Test WeatherFormer spatial attention contracts"
```

### Task 2: Implement WeatherFormer query/key encoding

**Files:**
- Modify: `TFT_model/models.py:14-664`
- Test: `tests/test_spatial_contracts.py`

**Interfaces:**
- `SpatialAttentionAggregator(hidden_size: int, dropout: float = 0.1)` has no encoding-mode argument.
- `_st_pe(coords: Tensor, t_idx: Tensor) -> Tensor` returns `(B,G,T,H)`.
- `forward_weights(tokens: Tensor, coords: Tensor, grid_mask: Tensor) -> Tuple[Tensor, Tensor]` returns `(B,T,H)` and `(B,T,G)`.
- `forward(tokens, coords, grid_mask) -> Tensor` returns only `(B,T,H)`.

- [ ] **Step 1: Remove obsolete rotation helpers and axis-boundary helper.**

Delete `_rotate_half_pairwise`, `_apply_rotary`, and `_grid_axis_boundaries`, along with all 2D RoPE code and mode validation. Keep `math`, `Tuple`, and other imports only where still used.

- [ ] **Step 2: Implement the shared four-slot positional encoder.**

Replace `_spatial_pe` with:

```python
def _st_pe(self, coords: torch.Tensor, t_idx: torch.Tensor) -> torch.Tensor:
    d = self.hidden_size
    if d % 4 != 0:
        raise ValueError("WeatherFormer position encoding requires hidden_size divisible by 4")
    nf = d // 4
    i = torch.arange(nf, device=coords.device, dtype=coords.dtype)
    freq = 10000.0 ** (-4.0 * i / d)
    t = t_idx.to(device=coords.device, dtype=coords.dtype).view(1, 1, -1, 1)
    lat = (coords[..., 0:1] * (math.pi / 180.0)).unsqueeze(2)
    lon = (coords[..., 1:2] * (math.pi / 180.0)).unsqueeze(2)
    pe = torch.zeros(*coords.shape[:-1], t_idx.numel(), d,
                     device=coords.device, dtype=coords.dtype)
    pe[..., 0::4] = torch.sin(t * freq)
    pe[..., 1::4] = torch.cos(t * freq)
    pe[..., 2::4] = torch.sin(lat * freq)
    pe[..., 3::4] = torch.cos(lon * freq)
    return pe
```

Use the exact vectorized implementation needed to satisfy the tests; do not introduce learnable positional parameters.

- [ ] **Step 3: Make CLS coordinates and query time-dependent.**

In `forward_weights`, compute:

```python
t_idx = torch.arange(T, device=tokens.device, dtype=torch.long)
valid_f = grid_mask.to(dtype=coords.dtype).unsqueeze(-1)
denom = valid_f.sum(dim=1, keepdim=True).clamp_min(1.0)
cls_coords = (coords * valid_f).sum(dim=1, keepdim=True) / denom
cls_coords = cls_coords.squeeze(1)
pe_grid = self._st_pe(coords, t_idx).transpose(1, 2)  # (B,T,G,H)
pe_cls = self._st_pe(cls_coords.unsqueeze(1), t_idx).squeeze(1)  # (B,T,H)
cls = self.cls_token.view(1, 1, H).expand(B, T, H)
q = self.W_q(cls + pe_cls).unsqueeze(2)
x = tokens.transpose(1, 2)
k = self.W_k(x + pe_grid)
v = self.W_v(x)
```

Mask only keys with `grid_mask.unsqueeze(1)`, compute `(B,T,G)` weights, and aggregate `v` with those weights. Preserve the existing output LayerNorm and dropout behavior.

- [ ] **Step 4: Run focused tests to verify the implementation.**

Run:

```bash
conda run -n product python -m unittest tests/test_spatial_contracts.py -v
```

Expected: all spatial contract tests pass.

- [ ] **Step 5: Commit the aggregator implementation.**

```bash
git add TFT_model/models.py tests/test_spatial_contracts.py
git commit -m "Add WeatherFormer spatiotemporal spatial attention"
```

### Task 3: Update model, training, and inference contracts

**Files:**
- Modify: `TFT_model/models.py`
- Modify: `TFT_model/train.py`
- Modify: `TFT_model/infer.py`
- Modify: `ablation/ablation.py`
- Test: `tests/test_spatial_contracts.py`

**Interfaces:**
- `TFTEncoderForYieldPrediction(..., spatial_mode="attention")` no longer accepts `spatial_encoding`.
- Training hparams contain `model_contract_version=4`, `spatial_mode`, `use_gdd`, and `use_constructed`, but no `spatial_encoding`.
- Inference requires hparams contract version 4 and constructs the model from `spatial_mode` only.

- [ ] **Step 1: Remove `spatial_encoding` from `TFTEncoderForYieldPrediction`.**

Change the constructor to accept `spatial_mode` only for spatial selection. For `attention`, instantiate `SpatialAttentionAggregator(hidden_size, dropout)`; for `mean`, keep the current masked mean path. Remove all `spatial_encoding` assignments and validation.

- [ ] **Step 2: Remove the training CLI/config field and upgrade the contract.**

Delete the `--spatial_encoding` parser block. Import `MODEL_CONTRACT_VERSION` as before, change the constant to 4, construct the model without `spatial_encoding`, and remove the hparams field. Keep `--spatial_mode` choices `attention` and `mean`.

- [ ] **Step 3: Make inference enforce the new contract.**

Require `model_contract_version == 4`; remove the check for `spatial_encoding`, remove its log output, and construct `TFTEncoderForYieldPrediction` using only `spatial_mode`. Missing hparams and old versions must raise `ValueError` before loading the checkpoint.

- [ ] **Step 4: Update the legacy ablation runner.**

Change its hparams version to 4 and remove `spatial_encoding` from the JSON and model constructor. Keep `spatial_mode="mean"` for mean and `spatial_mode="attention"` for attention.

- [ ] **Step 5: Run focused tests and syntax checks.**

Run:

```bash
conda run -n product python -m unittest tests/test_spatial_contracts.py -v
conda run -n product python -m py_compile TFT_model/models.py TFT_model/train.py TFT_model/infer.py ablation/ablation.py
```

Expected: focused tests pass and `py_compile` exits 0.

- [ ] **Step 6: Commit the contract changes.**

```bash
git add TFT_model/models.py TFT_model/train.py TFT_model/infer.py ablation/ablation.py tests/test_spatial_contracts.py
git commit -m "Update model contract for WeatherFormer attention"
```

### Task 4: Reduce the ablation runner to mean and additive

**Files:**
- Modify: `TFT_model/ablation_rope.py`
- Modify: `tests/test_ablation_rope.py`

**Interfaces:**
- `constructed_combo(0)` returns `{"use_constructed": True, "spatial_mode": "mean"}`.
- `constructed_combo(1)` returns `{"use_constructed": True, "spatial_mode": "attention"}`.
- `--parallel` requires `--constructed` and at least one visible GPU per pending combo.

- [ ] **Step 1: Replace the combo table and command construction.**

Use:

```python
def constructed_combo(i: int) -> dict:
    return [
        {"use_constructed": True, "spatial_mode": "mean"},
        {"use_constructed": True, "spatial_mode": "attention"},
    ][i]
```

Use names `mean` and `additive`, but pass only `--spatial_mode mean|attention` to `train.py`. Remove every `spatial_encoding` field and update the script description, range checks, summary columns, and output tag to a new `weatherformer2` tag.

- [ ] **Step 2: Make parallel assignment and validation use two combos.**

Set `max_combo=1`, preserve duplicate detection, and assign `zip(pending_combos, visible_devices)`. Keep the existing second validation against `len(pending_combos)` so `--combo 1` only requires one GPU.

- [ ] **Step 3: Run ablation CLI contract tests.**

Run:

```bash
conda run -n product python -m unittest tests/test_ablation_rope.py -v
```

Expected: all combo, parser, duplicate, and GPU-count tests pass.

- [ ] **Step 4: Commit the ablation runner changes.**

```bash
git add TFT_model/ablation_rope.py tests/test_ablation_rope.py
git commit -m "Reduce ablation runner to mean and additive"
```

### Task 5: Full verification and controlled new experiments

**Files:**
- No source changes expected.
- Runtime outputs: `TFT_model/train_output/weatherformer2_*` must remain untracked.

- [ ] **Step 1: Run the complete test and environment checks.**

```bash
conda run -n product python -m unittest discover -s tests -v
conda run -n product python -m py_compile TFT_model/models.py TFT_model/data.py TFT_model/train.py TFT_model/infer.py TFT_model/ablation_rope.py train_dataset/prepare_grid.py ablation/ablation.py
conda run -n product python -m pip check
git diff --check
```

Expected: all tests pass, syntax compilation exits 0, pip reports `No broken requirements found.`, and `git diff --check` is silent.

- [ ] **Step 2: Run a CPU shape smoke test.**

Run this from the repository root:

```bash
conda run -n product python - <<'PY'
import sys
import torch
sys.path.insert(0, "TFT_model")
from data import DEFAULT_DYNAMIC_FEATURE_NAMES, CONSTRUCTED_FEATURES
from models import TFTEncoderForYieldPrediction

names = list(DEFAULT_DYNAMIC_FEATURE_NAMES) + list(CONSTRUCTED_FEATURES)
grid_feats = torch.randn(2, 3, 5, 15)
grid_coords = torch.tensor([
    [[40.0, -90.0], [40.2, -89.8], [40.4, -89.6]],
    [[42.0, -92.0], [42.2, -91.8], [0.0, 0.0]],
])
grid_mask = torch.tensor([[True, True, True], [True, True, False]])
soil = torch.randn(2, 7)
seq_lens = torch.tensor([5, 4])
for mode in ("mean", "attention"):
    model = TFTEncoderForYieldPrediction(
        soil_dim=7,
        dynamic_feature_names=names,
        hidden_size=36,
        num_lstm_layers=1,
        dropout=0.0,
        output_size=1,
        num_heads=1,
        spatial_mode=mode,
    ).eval()
    with torch.no_grad():
        pred, _, _ = model(grid_feats, grid_coords, grid_mask, soil, seq_lens)
    assert pred.shape == (2, 5, 1), (mode, pred.shape)
    assert torch.isfinite(pred).all(), mode
print("CPU smoke passed")
PY
```

Expected: `CPU smoke passed`.

- [ ] **Step 3: Run the two real CUDA forwards before training.**

Create a temporary script under `/tmp/opencode/weatherformer_cuda_smoke.py` using the same model setup as the CPU smoke test, move tensors/models to CUDA, and call `model.spatial_agg.forward_weights` on one projected feature. Assert:

```python
assert torch.isfinite(pred).all()
assert weights.shape == (2, 5, 3)
assert torch.allclose(weights[1, :, 2], torch.zeros(5, device="cuda"))
assert torch.allclose(weights.sum(-1), torch.ones(2, 5, device="cuda"), atol=1e-6)
assert not torch.allclose(weights[:, 0], weights[:, 1])
```

Run with `conda run -n product python /tmp/opencode/weatherformer_cuda_smoke.py`. Expected: both modes print finite prediction shapes and attention prints `CUDA smoke passed`.

- [ ] **Step 4: Start the new two-group experiment on two GPUs.**

```bash
CUDA_VISIBLE_DEVICES=2,3 conda run -n product python -u /data/hqx/myself/Product_model/TFT_model/ablation_rope.py --constructed --parallel
```

Expected mapping: `mean->GPU 2`, `additive->GPU 3`. The new output directory must not reuse `spatial3_*` checkpoint files.

- [ ] **Step 5: Compare metrics only after both runs finish.**

Read the new `ablation_results.json`, both `infer_results.json` files, and training logs. Report final RMSE/R²/Corr, all cutoff nodes, best epoch, and whether attention weights vary with relative day.

- [ ] **Step 6: Review final worktree scope.**

Run:

```bash
git status --short --branch
git log --oneline -5
```

Confirm only intended source/test/docs commits are included and training outputs, data links, and unrelated existing deletions remain uncommitted.
