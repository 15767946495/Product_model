# Task 1 实现报告

## 状态

DONE_WITH_CONCERNS

## 提交哈希

- `273c979`：`refactor: define unified eight-state cropnet protocol`
- `49c655f`：`fix: preserve day fields in cropnet samples`

## 修改文件

- `tests/test_cropnet_protocol.py`
- `train_dataset/prepare_jsonl.py`
- `train_dataset/prepare_grid.py`
- `TFT_model/data.py`
- `BaseLine_Model/common/data.py`

实现内容：

- 建立八州允许列表：Minnesota、Wisconsin、Michigan、Illinois、Indiana、Ohio、Missouri、Kentucky。
- 统一 4--9 月、每月前 28 天和最大 168 步协议。
- USDA 样本按允许州列表过滤，天气行按月份和日期范围过滤。
- 县级 JSONL 和网格缓存均保留 `month`、`day` 时间字段。
- 网格缓存升级为版本 4，并写入 `time_window`、`days_per_month`、`max_steps` 元数据。
- TFT 网格缓存加载器拒绝旧版本、错误时间窗口或错误每月天数的缓存。
- 基线数据模块切换为八州和 `N_STEPS = 168`。
- 未删除运行产物，未训练模型，未修改后续 MMST-ViT manifest 或模型基线实现。

## 测试命令及实际结果

测试前发现指定环境未安装 `pytest`，使用同一指定 Python 安装后运行：

```text
/root/miniconda3/envs/hqx/bin/python -m pip install pytest
```

结果：成功安装 `pytest 9.1.1`。

协议测试：

```text
/root/miniconda3/envs/hqx/bin/python -m pytest tests/test_cropnet_protocol.py -q
```

结果：`4 passed, 1 warning in 1.28s`。

编译和差异检查：

```text
git diff --check && /root/miniconda3/envs/hqx/bin/python -m py_compile tests/test_cropnet_protocol.py train_dataset/prepare_jsonl.py train_dataset/prepare_grid.py TFT_model/data.py BaseLine_Model/common/data.py
```

结果：退出码 `0`，无输出。

## 未解决问题

- 测试运行时出现 PyTorch 的 `pynvml` 弃用警告；不影响本任务测试结果。
- 本任务未执行真实数据重建，因为真实数据生成和缓存重建属于后续任务，且用户明确要求本任务不删除运行产物、不训练模型。
- 当前工作区仍有任务前已存在的未跟踪运行产物 `DataSrc` 和 `BaseLine_Model/output`，未纳入提交。
