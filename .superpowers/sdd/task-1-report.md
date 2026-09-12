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

## 审查修复追加记录

### 修复提交

- `1d97eba`：`fix: enforce shared cropnet protocol contract`

### 修复内容

- 新增项目级轻量模块 `cropnet_protocol.py`，集中定义允许州、月份、每月天数、最大步数和时间窗口。
- `prepare_jsonl.py`、`prepare_grid.py`、`TFT_model/data.py`、`BaseLine_Model/common/data.py` 改为导入共享协议对象；脚本直接运行时会加入项目根目录到 import 路径。
- `load_grid_cache` 现在严格拒绝缺失或错误的 `version`、`time_window`、`days_per_month`、`max_steps`。
- `prepare_grid.py` 读取 JSONL 后主动校验每行州值属于八州允许列表。
- 新增天气行过滤、日历字段长度/范围/168 步上限、缓存元数据拒绝、州校验和共享对象一致性测试。
- 更新 Task 1 涉及文件中的旧九州、275 步、3--11 月协议注释；未修改构造特征 DOY 行为。

### 修复测试命令及实际结果

```text
/root/miniconda3/envs/hqx/bin/python -m pytest tests/test_cropnet_protocol.py -q
```

结果：`15 passed, 1 warning in 1.28s`。

```text
git diff --check && /root/miniconda3/envs/hqx/bin/python -m py_compile cropnet_protocol.py tests/test_cropnet_protocol.py train_dataset/prepare_jsonl.py train_dataset/prepare_grid.py TFT_model/data.py BaseLine_Model/common/data.py
```

结果：退出码 `0`，无输出。

### 修复后 concerns

- 测试仍有 PyTorch 的 `pynvml` 弃用警告。
- 按任务范围未执行真实数据重建、未训练模型，未修改后续 MMST-ViT manifest 或其他基线模型文件。
- 工作区仍保留既有未跟踪运行产物 `DataSrc` 和 `BaseLine_Model/output`，未修改、未提交。
