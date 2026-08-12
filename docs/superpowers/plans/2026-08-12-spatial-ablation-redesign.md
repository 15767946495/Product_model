# 空间消融重构实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild center-coordinate cache, remove temporal RoPE, use query-only CLS spatial pooling, and run three spatial ablations: mean, additive, and 2D RoPE.

**Architecture:** Keep per-feature spatial aggregation so the temporal VSN still selects all 15 features. Replace the current spatial aggregator with an explicit `spatial_encoding` mode and cross-attention where CLS is query-only. Rebuild `grid_cache.pt` from raw WRF-HRRR CSV using grid-center coordinates and version 3 metadata.

**Tech Stack:** Python 3.10, PyTorch, pandas, NumPy, unittest, existing `product` Conda environment.

## Global Constraints

- A `mean`: valid-grid arithmetic mean with no position encoding.
- B `additive`: query-only CLS cross-attention with spatial-only additive sinusoidal encoding.
- C `rope`: query-only CLS cross-attention with spatial-only 2D RoPE.
- Temporal RoPE is removed from model, train, infer, and ablation CLI.
- Grid coordinates are `(llcrnr + urcrnr) / 2`; cache version is `3`.
- The 15 dynamic features remain separate through spatial aggregation and are selected by temporal VSN.
- Old cache and checkpoints are intentionally incompatible with the new implementation.

---

### Task 1: Add failing tests for the new contracts

**Files:**
- Modify: `tests/test_ablation_rope.py`
- Create: `tests/test_spatial_contracts.py`

**Interfaces:**
- `prepare_grid.center_coords(row)` returns `(lat_center, lon_center)`.
- `SpatialAttentionAggregator(..., spatial_encoding="additive"|"rope")` returns output `(B,T,H)` and weights `(B,T,G)`.
- `TFTEncoderForYieldPrediction(..., spatial_encoding=...)` has no temporal RoPE option.

- [ ] **Step 1: Write the failing tests.**

  Test center coordinates from four boundary values, reject the old cache version, assert duplicate/three-combo validation, and assert the spatial aggregator weight last dimension is `G` rather than `G+1`.

- [ ] **Step 2: Run tests to verify they fail.**

  ```bash
  conda run -n product python -m unittest tests/test_ablation_rope.py tests/test_spatial_contracts.py -v
  ```

  Expected: failures for the not-yet-existing center helper, three-combo interface, and query-only weight shape.

### Task 2: Rebuild the grid cache with center coordinates

**Files:**
- Modify: `train_dataset/prepare_grid.py`
- Modify: `train_dataset/grid_cache_meta.json` only through the generator

- [ ] **Step 1: Add `center_coords(row)` and use upper/lower corner fields.**

  Replace `Lat (llcrnr)`/`Lon (llcrnr)` assignment with the arithmetic midpoint of `Lat (llcrnr)` and `Lat (urcrnr)`, and `Lon (llcrnr)` and `Lon (urcrnr)`.

- [ ] **Step 2: Change generated payload version to `3`.**

  Include the coordinate semantic in metadata, e.g. `"coord_type": "grid_center"`.

- [ ] **Step 3: Delete the old cache and regenerate it.**

  ```bash
  rm /data/hqx/myself/Product_model/train_dataset/grid_cache.pt
  conda run -n product python /data/hqx/myself/Product_model/train_dataset/prepare_grid.py
  ```

  Expected: the generator completes, writes version 3, and reports the existing sample/grid counts.

- [ ] **Step 4: Verify a cached coordinate against raw CSV boundaries.**

  Load one known grid row from the source CSV and assert cached coordinates equal both boundary midpoints within `1e-5` degrees.

### Task 3: Replace spatial aggregator and remove temporal RoPE

**Files:**
- Modify: `TFT_model/models.py`
- Modify: `TFT_model/train.py`
- Modify: `TFT_model/infer.py`

- [ ] **Step 1: Implement query-only spatial cross-attention.**

  In `SpatialAttentionAggregator`, construct `q` from CLS only and `k/v` from grid tokens only. Use valid-grid masking on the key dimension, return weights shaped `(B,T,G)`, and keep the query spatially neutral.

- [ ] **Step 2: Implement spatial-only additive encoding.**

  Replace `_st_pe(coords, t_idx)` with a coordinate-only encoder returning `(B,G,H)` and broadcast it over time. The additive path must not accept or use a time index.

- [ ] **Step 3: Keep spatial-only 2D RoPE.**

  Apply center-coordinate RoPE to grid keys only. Do not rotate or position-encode a CLS key/value because no CLS key/value exists.

- [ ] **Step 4: Remove temporal RoPE.**

  Delete temporal RoPE construction and application from `CausalScaledDotProductAttention`. Remove `use_time_rope` from model construction, saved hyperparameters, train CLI, infer hyperparameter loading, and all log output.

- [ ] **Step 5: Replace `grid_rope` with `spatial_encoding`.**

  Accept `none`, `additive`, and `rope`; use `none` for mean and the other two for attention. Reject invalid combinations.

- [ ] **Step 6: Run focused model tests.**

  ```bash
  conda run -n product python -m unittest tests/test_spatial_contracts.py -v
  conda run -n product python -m py_compile TFT_model/models.py TFT_model/train.py TFT_model/infer.py
  ```

### Task 4: Convert ablation runner to three experiments

**Files:**
- Modify: `TFT_model/ablation_rope.py`
- Modify: `tests/test_ablation_rope.py`

- [ ] **Step 1: Define three constructed combinations.**

  Map `0 -> mean/none`, `1 -> additive/additive`, and `2 -> rope/rope`; always append `--use_constructed` and pass `--spatial_encoding` to training.

- [ ] **Step 2: Remove time-RoPE fields and CLI paths.**

  Update output tags and summary columns to report spatial mode/encoding only. Preserve `--parallel`, assigning three combinations to three visible GPUs.

- [ ] **Step 3: Use a new output tag.**

  Use `spatial3` in the tag so old `ablation2d_*` directories and checkpoints cannot be reused.

- [ ] **Step 4: Run CLI and mapping tests.**

  ```bash
  conda run -n product python TFT_model/ablation_rope.py --help
  CUDA_VISIBLE_DEVICES=2,3,4 conda run -n product python TFT_model/ablation_rope.py --constructed --parallel --combo 0 1 2 --infer-only --val_year 2096
  ```

  Expected: mapping is `mean->2`, `additive->3`, `rope->4`; no training starts without checkpoints, and one summary JSON is written.

### Task 5: Final verification

**Files:**
- No additional source files

- [ ] **Step 1: Run all tests and static checks.**

  ```bash
  conda run -n product python -m unittest discover -s tests -v
  conda run -n product python -m py_compile TFT_model/models.py TFT_model/train.py TFT_model/infer.py TFT_model/ablation_rope.py train_dataset/prepare_grid.py
  conda run -n product python -m pip check
  ```

- [ ] **Step 2: Verify the cache contract.**

  Assert `torch.load("train_dataset/grid_cache.pt")["version"] == 3`, metadata says `grid_center`, and the cache file exists after the old file was removed.

- [ ] **Step 3: Verify a real forward pass.**

  Load one batch with the regenerated cache and run mean, additive, and rope models in `product`; assert finite predictions and expected output/attention shapes.
