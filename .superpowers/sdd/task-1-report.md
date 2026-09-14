# Task 1 报告：五州协议和年份划分接口

## 状态

已实现并通过验证。

## 实现内容

- 在 `cropnet_protocol.py` 增加 `CROPNET_FIVE_STATES`，值为规范化的小写完整州名：Illinois、Iowa、Louisiana、Mississippi、New York。
- 在 `TFT_model/data.py` 导出 `validate_five_state_sample(sample) -> None`。
  - 州名执行字符串化、去首尾空白和小写归一化。
  - 仅接受五州集合及 2017--2022 协议年份。
  - 协议外州或年份抛出 `ValueError`。
- 在 `TFT_model/data.py` 导出 `split_samples_by_year(...)`。
  - 默认训练年份为 2017--2020，验证年份为 2021，测试年份为 2022。
  - 自定义训练、验证、测试年份组先执行固定 2017--2022 协议校验；协议外分组年份直接抛出 `ValueError`。
  - 每个样本统一调用 `validate_five_state_sample` 校验五州和固定协议年份，再按指定年份集合划分。
  - 返回 `train`、`val`、`test` 三个列表，并复制样本字典。
  - 协议外州或年份抛出带样本索引的 `ValueError`。
- `split_samples_by_year` 返回类型标注为 `dict[str, list[dict]]`。
- `tests/test_tft_ag_data.py` 覆盖四个训练年份、州名归一化、样本复制、协议外州、协议外年份、自定义协议外年份和常量内容。

## TDD 验证

先运行缺少实现时的测试，收集阶段按预期失败：

```text
ImportError: cannot import name 'CROPNET_FIVE_STATES' from 'cropnet_protocol'
```

补充实现后，目标测试结果：

```text
9 passed, 1 warning in 1.21s
```

## 完整验证

执行：

```text
/root/miniconda3/envs/hqx/bin/python -m pytest -q
/root/miniconda3/envs/hqx/bin/python -m py_compile cropnet_protocol.py TFT_model/data.py tests/test_tft_ag_data.py
```

结果：

```text
112 passed, 1 warning in 29.28s
py_compile exit code 0，无输出
```

唯一警告来自环境中的 `pynvml` 弃用提示，不是本任务代码产生的失败。

## 范围确认

- 未修改 TFT 模型结构、`TFT_model/models.py` 或训练/推理逻辑。
- 未实现 AG 加载；该内容属于后续任务。
- 未读取、修改、删除或训练/推理运行数据及原始 Sentinel 数据。
- 工作区中已有的无关变更（`_txt/` 删除、`BaseLine_Model/output` 和 `DataSrc` 未跟踪内容）未触碰。
