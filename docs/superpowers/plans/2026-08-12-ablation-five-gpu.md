# 五卡消融并行 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a five-GPU parallel mode for the five constructed ablation experiments in `TFT_model/ablation_rope.py`.

**Architecture:** Keep the existing train and infer subprocess commands. Add a worker function that receives one combo and one physical GPU id, sets `CUDA_VISIBLE_DEVICES` for that worker, runs train then infer, and returns the same result record currently assembled by the serial loop. Use a bounded thread pool in the parent only for process orchestration; write `ablation_results.json` once after all workers finish.

**Tech Stack:** Python 3, `argparse`, `concurrent.futures.ThreadPoolExecutor`, existing `subprocess` and PyTorch scripts.

## Global Constraints

- Parallel mode is valid only with `--constructed`.
- GPU ids come from the caller's `CUDA_VISIBLE_DEVICES` list and map in order to pending combos.
- Do not modify the train/infer model implementation.
- Preserve serial behavior when `--parallel` is absent.
- Keep each combo's output and logs isolated.

### Task 1: Add parallel orchestration

**Files:**
- Modify: `TFT_model/ablation_rope.py`
- Test: manual CLI and syntax checks

**Interfaces:**
- Add CLI flag `--parallel`.
- Add helper `parse_visible_devices(value: str) -> list[str]`.
- Add worker helper taking `(combo_id, gpu_id, combo_fn, combo_name, args, base, feat_col)` and returning `(combo_name, result_dict)`.

- [ ] **Step 1: Add the flag and GPU parser.**

  Parse comma-separated visible devices, reject empty entries, and use `CUDA_VISIBLE_DEVICES` from the parent environment. In parallel constructed mode, reject fewer devices than pending valid combos before starting workers.

- [ ] **Step 2: Extract one-combo train/infer logic.**

  Move the current loop body into a worker-compatible function. Set the worker environment's `CUDA_VISIBLE_DEVICES` to the assigned physical id before calling existing `run`; preserve skip, `--force`, `--infer-only`, log paths, and result parsing.

- [ ] **Step 3: Dispatch pending combos.**

  Use `ThreadPoolExecutor(max_workers=len(pending_combos))`; submit one worker per pending combo. Collect completed futures and merge results by combo name. Keep invalid combos out of the worker list and keep skipped combos represented in the final results.

- [ ] **Step 4: Keep summary writing centralized.**

  Run the existing summary printer after either serial or parallel execution, and write `ablation_results.json` only in the parent process. Ensure a worker exception becomes a combo failure record rather than preventing the other futures from being collected.

- [ ] **Step 5: Run focused verification.**

  Run:

  ```bash
  conda run -n product python -m py_compile TFT_model/ablation_rope.py
  conda run -n product python TFT_model/ablation_rope.py --help
  CUDA_VISIBLE_DEVICES=2,3,4,5,6 conda run -n product python TFT_model/ablation_rope.py --constructed --parallel --combo 0 1 --infer-only
  ```

  Expected: syntax succeeds, help lists `--parallel`, and the two selected workers validate the GPU mapping/independent control flow without training when checkpoints are absent.

### Task 2: Verify CUDA environment

**Files:**
- No source changes

- [ ] **Step 1: Verify PyTorch and CUDA.**

  Run:

  ```bash
  conda run -n product python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
  ```

  Expected: `2.11.0+cu128`, CUDA `12.8`, availability `True`, and the expected visible-device count when `CUDA_VISIBLE_DEVICES` is set.

- [ ] **Step 2: Verify no dependency conflicts.**

  Run `conda run -n product python -m pip check` and expect `No broken requirements found.`
