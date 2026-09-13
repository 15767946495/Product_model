# Task 8 集成修复审计报告

## 状态

- 状态：**通过**
- 运行环境：`/root/miniconda3/envs/hqx/bin/python`
- 运行时根目录：`/data/raid0/hqx/Product_model_runtime`
- 生成方式：staging 生成、完整审计和只读 smoke 通过后，Linux `renameat2(RENAME_EXCHANGE)` 一次交换完整 runtime bundle；两个稳定兼容入口均指向该 bundle 的子目录，不逐文件更新 manifest
- 回滚：旧 `train_dataset` 和 MMST manifest 目录保留为 `.task8-legacy`，新 active bundle 失败时不会被半成品覆盖
- 训练/推理：未执行
- Sentinel、USDA、weather、soil：仅只读访问，未修改或删除
- 旧权重：保持已删除

## 运行时三件套

- `train_dataset/dataset.jsonl`：3247 行
- `train_dataset/grid_cache.pt`：3247 entries，全部非空，全部 168 步
- `train_dataset/grid_cache_meta.json`：version 4，`04-01--09-28`，`max_steps=168`
- `manifests/valid-no-ia.jsonl`：3247 行，与共享 JSONL `(FIPS,Year)` 集合严格相等
- `manifests/valid-no-ia.{train,val,test}.jsonl`：`2227/504/516` 行
- `manifests/{train,val,test}.official.no-ia.json`：`2227/504/516` 条
- `task8-active/train_dataset` 和 `task8-active/manifests` 属于同一外部 active bundle；默认入口解析后不会跨 bundle 混合

### SHA-256

```text
dataset.jsonl       749005984b6af4b673ff1bc04d60f57b4176df3094be3caea09fa6e91d8c44b8
grid_cache.pt       d899abefb92fbd0a306446edf27b7b129c402856f5509e8d59174135de373fce
grid_cache_meta.json f2cfee4e88e65236b94e92efa5e093fc85c8cc20f227d8c005536f131b75201c
valid-no-ia.jsonl    68d85a624f6b1c6432a80b1247e664866df33c90d29cb174aec7322a2932ccc3
valid-no-ia.train.jsonl 471ffdd956d852233cbb072114a510c4cb52cb30385c955dc99d4e7d271e9a71
valid-no-ia.val.jsonl 836aa3b62d4e289e09247574450362761f16556cc731a3e64ac4d1d279e63487
valid-no-ia.test.jsonl 5964a272279ac0d1ddece7bc265de6f9d7059c5c07ee0f6c34f3357454b24caf
train.official.no-ia.json 096dd7919cec195db00a17b877cba7ce5f6b66d7b538648d1086b61eb79d5f9a
val.official.no-ia.json 3cd09f77a3aed99b286302b5143e24ca999ec927cdb70439b8366bf448acb823
test.official.no-ia.json 6c2de48124d6d5ed0ab268f57760bdf1942042d16cf6c36d3edd23082d22e9c0
```

## 数据统计

### 州

| 州 | 行数 |
| --- | ---: |
| illinois | 532 |
| indiana | 391 |
| kentucky | 443 |
| michigan | 329 |
| minnesota | 410 |
| missouri | 328 |
| ohio | 463 |
| wisconsin | 351 |

### 年份

| 年份 | 行数 |
| --- | ---: |
| 2017 | 591 |
| 2018 | 514 |
| 2019 | 480 |
| 2020 | 642 |
| 2021 | 504 |
| 2022 | 516 |

每条 JSONL 和 grid entry 的日期序列均严格等于 `[(4,1)..(4,28), ..., (9,1)..(9,28)]`。

## Split 与跨管线集合

| split | shared JSONL / valid-no-ia | official JSON | `(FIPS,Year)` 集合 |
| --- | ---: | ---: | --- |
| train | 2227 / 2227 | 2227 | 相等 |
| val | 504 / 504 | 504 | 相等 |
| test | 516 / 516 | 516 | 相等 |

MMST manifest 现在从共享 JSONL 身份源生成；USDA/weather 不再独立决定 valid-no-ia 样本集合。

## 只读 Smoke Test

- `load_jsonl`：通过，3247 行
- `load_grid_cache`：通过，3247 entries
- `build_grid_samples`：通过，抽样构建 8 个样本
- baseline prepare：通过，train/val/test 为 `2207/499/510`；缺少 county soil 的样本按现有 grid 管线规则跳过
- DeepCropNet prepare：通过，train/val/test 为 `2227/504/516`
- DeepCropNet 默认 reload：通过，force=True 生成后 force=False 重新加载三份 cache，均为 `val_year=2021/test_year=2022`，行数 `2227/504/516`
- 未启动训练或推理

## 缓存协议

- TFT/grid cache、baseline cache、DeepCropNet cache 均检查 `version=4`、`time_window=04-01--09-28`、`days_per_month=28`、`max_steps=168`
- baseline 与 DeepCropNet 额外检查 `val_year/test_year`
- 缺失或不匹配的旧协议缓存会拒绝加载

## 测试

```text
91 passed, 1 warning
```

测试覆盖完整日期序列及长度/对齐门禁、缓存旧协议拒绝、共享 JSONL/root valid manifest 身份源、官方结构/状态/路径审计、运行时审计、meta `days_per_month`、val/test DeepCropNet split 年份拒绝、默认 reload smoke、audit-only CLI、整体原子目录交换和安装失败回滚。

机器可读明细：`.superpowers/sdd/task-8-report.json`。
