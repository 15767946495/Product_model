"""Canonical metadata for the P/N/S/A/U experiment matrix."""

from dataclasses import dataclass
from enum import Enum

from .capabilities import Capability, missing_capabilities
from .protocol import DEFAULT_MODEL_IDENTITY, DEFAULT_PROTOCOL


class ExperimentStatus(str, Enum):
    READY = "ready"
    BLOCKED = "blocked"
    EXTERNAL = "external"
    ANALYSIS_ONLY = "analysis-only"


EXECUTABLE_RUNNERS = {
    "official_mmst": ["python", "MMST-ViT/pretrain_current.py"],
}
KNOWN_RUNNERS = frozenset(EXECUTABLE_RUNNERS)

@dataclass(frozen=True)
class ExperimentDefinition:
    experiment_id: str
    group: str
    title: str
    research_question: str
    baseline_id: str | None
    changes: tuple[tuple[str, str], ...]
    fixed_factors: tuple[tuple[str, str], ...]
    required_capabilities: frozenset[Capability]
    metrics: tuple[str, ...]
    output_name: str
    runner: str | None
    declared_status: ExperimentStatus | None = None

    @property
    def missing_capabilities(self) -> tuple[Capability, ...]:
        return missing_capabilities(self.required_capabilities)

    @property
    def effective_status(self) -> ExperimentStatus:
        if self.missing_capabilities:
            return ExperimentStatus.BLOCKED
        if self.declared_status == ExperimentStatus.ANALYSIS_ONLY:
            return ExperimentStatus.ANALYSIS_ONLY
        if self.runner not in KNOWN_RUNNERS:
            return ExperimentStatus.BLOCKED
        if self.declared_status == ExperimentStatus.EXTERNAL:
            return ExperimentStatus.EXTERNAL
        return ExperimentStatus.READY


PRETRAIN_METRICS = tuple(sorted({
    "nt_xent_train", "positive_cosine", "negative_cosine",
    "similarity_margin", "embedding_std", "covariance_spectrum", "daily_metrics",
    "patch_attention", "negative_source_ratio", "linear_probe_metric",
    "downstream_test_rmse",
}))
SUPERVISED_METRICS = tuple(sorted({
    "rmse", "mae", "r2", "corr", "parameter_count", "training_time",
}))
PROBABILISTIC_METRICS = tuple(sorted({
    "rmse", "mae", "r2", "corr", "parameter_count", "training_time",
    "nll", "interval_coverage", "interval_width",
}))

S0_CAPABILITIES = frozenset({Capability.LINEAR_GRID_WEATHER_ENCODER})
S1_CAPABILITIES = S0_CAPABILITIES | frozenset({Capability.GRID_VSN})
S2_CAPABILITIES = S1_CAPABILITIES | frozenset({Capability.SOIL_CONDITIONING})
S3_CAPABILITIES = S2_CAPABILITIES | frozenset({Capability.TFT_TEMPORAL_BACKBONE})
S4_CAPABILITIES = S3_CAPABILITIES | frozenset({Capability.SAME_GRID_SUPERVISED_REMOTE})
S5_CAPABILITIES = S4_CAPABILITIES | frozenset({Capability.P5_WEIGHT_TRANSFER})

_P_FACTORS = (
    ("sequence_position", "none"),
    ("temporal_pooling", "mean"),
    ("spatial_alignment", "sample_grid"),
    ("contrast_unit", "sample"),
    ("negative_rule", "all_batch"),
)


def _definition(
    experiment_id: str,
    group: str,
    title: str,
    question: str,
    baseline_id: str | None,
    changes: tuple[tuple[str, str], ...],
    fixed_factors: tuple[tuple[str, str], ...],
    required_capabilities: frozenset[Capability],
    metrics: tuple[str, ...],
    runner: str | None = "internal",
    declared_status: ExperimentStatus | None = None,
) -> ExperimentDefinition:
    return ExperimentDefinition(
        experiment_id, group, title, question, baseline_id, changes, fixed_factors,
        required_capabilities, metrics, f"{experiment_id.lower()}_results", runner,
        declared_status,
    )


def _with_factors(**updates: str) -> tuple[tuple[str, str], ...]:
    return tuple((key, updates.get(key, value)) for key, value in _P_FACTORS)


def _build_registry() -> dict[str, ExperimentDefinition]:
    experiments: dict[str, ExperimentDefinition] = {
        "P0": _definition(
            "P0", "P", "MMST-ViT 官方预训练", "官方 MMST-ViT 预训练基线在当前机器上是否保持可复现实验行为？",
            None, (("model", "official_mmst"),), _P_FACTORS,
            frozenset({Capability.OFFICIAL_MMST_RUNNER}), PRETRAIN_METRICS, "official_mmst",
            ExperimentStatus.EXTERNAL,
        ),
        "P1": _definition(
            "P1", "P", "当前 TFT 预训练", "当前 TFT 预训练的天气 Query、patch 注意力、时间池化和样本级对比目标表现如何？",
            "P0", (("model", "tft"),), _P_FACTORS,
            frozenset({Capability.LEGACY_TFT_PRETRAIN}), PRETRAIN_METRICS, None,
        ),
        "P2": _definition(
            "P2", "P", "纯跨模态注意力", "不额外加入天气位置编码时，纯跨模态注意力的基线表现如何？",
            "P1", (("sequence_position", "none"),), _with_factors(sequence_position="none"),
            frozenset({Capability.LEGACY_TFT_PRETRAIN}), PRETRAIN_METRICS,
        ),
        "P3": _definition(
            "P3", "P", "去除时间池化", "保留逐日融合 token 是否优于对 28 天输出进行时间平均？",
            "P2", (("temporal_pooling", "none"),), _with_factors(sequence_position="none", temporal_pooling="none"),
            frozenset({Capability.LEGACY_TFT_PRETRAIN, Capability.UNPOOLED_DAILY_PRETRAIN_TOKENS}), PRETRAIN_METRICS,
        ),
        "P4": _definition(
            "P4", "P", "对应网格遥感", "严格同网格遥感 patch 对齐是否改善逐日跨模态预训练表示？",
            "P3", (("spatial_alignment", "strict_same_grid"),), _with_factors(sequence_position="none", temporal_pooling="none", spatial_alignment="strict_same_grid"),
            frozenset({Capability.LEGACY_TFT_PRETRAIN, Capability.UNPOOLED_DAILY_PRETRAIN_TOKENS, Capability.SAME_GRID_PATCH_ATTENTION}), PRETRAIN_METRICS,
        ),
        "P5": _definition(
            "P5", "P", "逐日网格对比与空间元数据负样本", "(t,g)级对比和空间元数据负样本是否构成完整的改进预训练方案？",
            "P4", (("contrast_unit", "day_grid"), ("negative_rule", "same_county_space_time")), _with_factors(sequence_position="none", temporal_pooling="none", spatial_alignment="strict_same_grid", contrast_unit="day_grid", negative_rule="same_county_space_time"),
            frozenset({Capability.LEGACY_TFT_PRETRAIN, Capability.UNPOOLED_DAILY_PRETRAIN_TOKENS, Capability.SAME_GRID_PATCH_ATTENTION, Capability.PRETRAIN_METADATA_BATCH, Capability.MASKED_TG_NT_XENT}), PRETRAIN_METRICS,
        ),
    }
    n_rules = (
        "all_batch", "same_county_space_time", "cross_county_only",
        "hierarchical_same_county_then_cross_county",
    )
    p5 = dict(experiments["P5"].fixed_factors)
    for index, rule in enumerate(n_rules, 1):
        factors = tuple((key, rule if key == "negative_rule" else value) for key, value in p5.items())
        experiments[f"N{index}"] = _definition(
            f"N{index}", "N", f"负样本规则 {rule}", f"负样本规则“{rule}”是否能降低假负样本并改善预训练表示？",
            "P5", (("negative_rule", rule),), factors,
            frozenset(experiments["P5"].required_capabilities), PRETRAIN_METRICS,
        )
    s_values = (
        ("S0", None, {"weather_encoder": "linear_grid", "soil_conditioning": "disabled", "temporal_backbone": "shared_simple", "remote_fusion": "none", "initialization": "random"}, "网格线性气象编码作为监督基准的表现如何？", S0_CAPABILITIES),
        ("S1", "S0", {"weather_encoder": "grid_vsn", "soil_conditioning": "disabled", "temporal_backbone": "shared_simple", "remote_fusion": "none", "initialization": "random"}, "网格 VSN 相较于线性气象编码是否改善变量选择？", S1_CAPABILITIES),
        ("S2", "S1", {"weather_encoder": "grid_vsn", "soil_conditioning": "enabled", "temporal_backbone": "shared_simple", "remote_fusion": "none", "initialization": "random"}, "土壤静态上下文是否改善网格变量选择和时序建模？", S2_CAPABILITIES),
        ("S3", "S2", {"weather_encoder": "grid_vsn", "soil_conditioning": "enabled", "temporal_backbone": "tft", "remote_fusion": "none", "initialization": "random"}, "TFT 时序主干相较于共享简单时序模块是否带来监督增益？", S3_CAPABILITIES),
        ("S4", "S3", {"weather_encoder": "grid_vsn", "soil_conditioning": "enabled", "temporal_backbone": "tft", "remote_fusion": "same_grid", "initialization": "random"}, "同网格遥感融合相较于无遥感 S3 是否改善监督预测？", S4_CAPABILITIES),
        ("S5", "S4", {"weather_encoder": "grid_vsn", "soil_conditioning": "enabled", "temporal_backbone": "tft", "remote_fusion": "same_grid", "initialization": "p5"}, "改进 P5 预训练初始化相较于随机初始化是否改善同网格监督迁移？", S5_CAPABILITIES),
    )
    for experiment_id, baseline, factors, question, capabilities in s_values:
        fixed = tuple(factors.items()) + (("loss", "mse"), ("output", "point"))
        change = (("weather_encoder", factors["weather_encoder"]),) if baseline is None else tuple((key, value) for key, value in factors.items() if dict(experiments[baseline].fixed_factors).get(key) != value)
        experiments[experiment_id] = _definition(experiment_id, "S", experiment_id, question, baseline, change, fixed, capabilities | frozenset({Capability.SUPERVISED_EXPERIMENT_RUNNER}), SUPERVISED_METRICS)
    a_values = (
        ("A0", "none", frozenset({Capability.SUPERVISED_NO_REMOTE}), "无遥感 S3 监督基准是否足以支持县级预测？"),
        ("A1", "same_grid_current_month", frozenset({Capability.SAME_GRID_SUPERVISED_REMOTE}), "同一网格当前月份遥感是否优于无遥感监督基准？"),
        ("A2", "county_retrieval_distance", frozenset({Capability.COUNTY_REMOTE_RETRIEVAL_DISTANCE}), "县内遥感检索加距离偏置是否优于固定同网格连接？"),
        ("A3", "county_retrieval_no_distance", frozenset({Capability.COUNTY_REMOTE_RETRIEVAL_NO_DISTANCE}), "移除距离偏置的县内遥感检索是否改变监督结果？"),
        ("A4", "historical_observed_no_future", frozenset({Capability.HISTORICAL_REMOTE_ONLY}), "仅使用历史已观测遥感且禁止未来信息是否影响监督结果？"),
    )
    for experiment_id, connection, capabilities, question in a_values:
        experiments[experiment_id] = _definition(experiment_id, "A", experiment_id, question, "S3", (("remote_connection", connection),), (("backbone", "S3"), ("remote_connection", connection)), S3_CAPABILITIES | capabilities | frozenset({Capability.SUPERVISED_EXPERIMENT_RUNNER}), SUPERVISED_METRICS)
    u_values = (
        ("U0", "S4", (("prediction_head", "mean"), ("loss", "mse")), (("prediction_head", "mean"), ("loss", "mse"), ("calibration", "none")), frozenset({Capability.MEAN_MSE_HEAD}), "本注册版本固定 S4；若验证选择 S5，则应生成另一固定协议而非动态混用。单均值头与 MSE 是否提供可靠的概率预测基准？"),
        ("U1", "U0", (("prediction_head", "mean_variance"), ("loss", "beta_nll")), (("prediction_head", "mean_variance"), ("loss", "beta_nll"), ("calibration", "none")), frozenset({Capability.HETEROSCEDASTIC_BETA_NLL}), "均值方差头与 Beta-NLL 是否改善异方差不确定性建模？"),
        ("U2", "U1", (("loss", "beta_nll_temporal_kl"),), (("prediction_head", "mean_variance"), ("loss", "beta_nll_temporal_kl"), ("calibration", "none")), frozenset({Capability.HETEROSCEDASTIC_BETA_NLL, Capability.TEMPORAL_KL}), "时间 KL 约束是否改善概率预测的时间一致性？"),
        ("U3", "U2", (("calibration", "variance"),), (("prediction_head", "mean_variance"), ("loss", "beta_nll_temporal_kl"), ("calibration", "variance")), frozenset({Capability.HETEROSCEDASTIC_BETA_NLL, Capability.TEMPORAL_KL, Capability.VARIANCE_CALIBRATION}), "方差校准是否改善预测区间覆盖率和宽度？"),
    )
    for experiment_id, baseline, changes, factors, capabilities, question in u_values:
        fixed = (("supervised_structure", "S4"),) + factors
        experiments[experiment_id] = _definition(experiment_id, "U", experiment_id, question, baseline, changes, fixed, S4_CAPABILITIES | capabilities | frozenset({Capability.SUPERVISED_EXPERIMENT_RUNNER}), PROBABILISTIC_METRICS)
    return experiments


EXPERIMENTS = _build_registry()


def get_experiment(experiment_id: str) -> ExperimentDefinition:
    return EXPERIMENTS[experiment_id]


def _factor_delta(current: ExperimentDefinition, baseline: ExperimentDefinition) -> set[str]:
    current_factors = dict(current.fixed_factors)
    baseline_factors = dict(baseline.fixed_factors)
    return {key for key in current_factors if current_factors[key] != baseline_factors.get(key)}


def validate_registry() -> tuple[str, ...]:
    errors: list[str] = []
    expected_ids = {"P0", "P1", "P2", "P3", "P4", "P5", "N1", "N2", "N3", "N4", "S0", "S1", "S2", "S3", "S4", "S5", "A0", "A1", "A2", "A3", "A4", "U0", "U1", "U2", "U3"}
    if set(EXPERIMENTS) != expected_ids:
        errors.append("registry: exact experiment ID set mismatch")
    if len({item.output_name for item in EXPERIMENTS.values()}) != len(EXPERIMENTS):
        errors.append("registry: output names must be unique")
    if (
        2022 in DEFAULT_PROTOCOL.pretrain_years
        or DEFAULT_PROTOCOL.finetune_year == DEFAULT_PROTOCOL.evaluation_year
        or DEFAULT_PROTOCOL.evaluation_year != 2022
        or DEFAULT_PROTOCOL.checkpoint_rule != "final_epoch"
    ):
        errors.append("protocol: 2022 must be reserved for final evaluation")
    if DEFAULT_PROTOCOL.finetune_epochs != 100 or DEFAULT_PROTOCOL.warmup_epochs != 5:
        errors.append("protocol: TFT fine-tuning epoch and warmup budget mismatch")
    budget = dict(DEFAULT_PROTOCOL.shared_budget)
    if budget.get("finetune_epochs") != str(DEFAULT_PROTOCOL.finetune_epochs):
        errors.append("protocol: shared fine-tuning budget disagrees with canonical field")
    if budget.get("warmup_epochs") != str(DEFAULT_PROTOCOL.warmup_epochs):
        errors.append("protocol: shared warmup budget disagrees with canonical field")
    for experiment_id, experiment in EXPERIMENTS.items():
        if not experiment_id.startswith(experiment.group):
            errors.append(f"{experiment_id}: group prefix mismatch")
        if experiment.baseline_id is not None and experiment.baseline_id not in EXPERIMENTS:
            errors.append(f"{experiment_id}: invalid baseline {experiment.baseline_id}")
        if not experiment.research_question or not experiment.metrics:
            errors.append(f"{experiment_id}: missing research question or metrics")
    expected_changes = {
        "P0": (("model", "official_mmst"),),
        "P1": (("model", "tft"),),
        "P2": (("sequence_position", "none"),),
        "P3": (("temporal_pooling", "none"),),
        "P4": (("spatial_alignment", "strict_same_grid"),),
        "P5": (("contrast_unit", "day_grid"), ("negative_rule", "same_county_space_time")),
        "N1": (("negative_rule", "all_batch"),),
        "N2": (("negative_rule", "same_county_space_time"),),
        "N3": (("negative_rule", "cross_county_only"),),
        "N4": (("negative_rule", "hierarchical_same_county_then_cross_county"),),
        "S0": (("weather_encoder", "linear_grid"),),
        "S1": (("weather_encoder", "grid_vsn"),),
        "S2": (("soil_conditioning", "enabled"),),
        "S3": (("temporal_backbone", "tft"),),
        "S4": (("remote_fusion", "same_grid"),),
        "S5": (("initialization", "p5"),),
        "U0": (("prediction_head", "mean"), ("loss", "mse")),
        "U1": (("prediction_head", "mean_variance"), ("loss", "beta_nll")),
        "U2": (("loss", "beta_nll_temporal_kl"),),
        "U3": (("calibration", "variance"),),
    }
    for experiment_id, changes in expected_changes.items():
        if EXPERIMENTS[experiment_id].changes != changes:
            errors.append(f"{experiment_id}: exact changes mismatch")
    if (
        EXPERIMENTS["P0"].baseline_id is not None
        or EXPERIMENTS["P0"].runner != "official_mmst"
        or EXPERIMENTS["P0"].declared_status != ExperimentStatus.EXTERNAL
    ):
        errors.append("P0: must be the external official MMST root experiment")
    if EXPERIMENTS["P1"].baseline_id != "P0" or dict(EXPERIMENTS["P1"].fixed_factors) != dict(EXPERIMENTS["P0"].fixed_factors):
        errors.append("P1: must be the TFT root with the P0 factor snapshot")
    for experiment_id, expected_delta in {"P2": set(), "P3": {"temporal_pooling"}, "P4": {"spatial_alignment"}, "P5": {"contrast_unit", "negative_rule"}}.items():
        baseline_id = EXPERIMENTS[experiment_id].baseline_id
        if baseline_id is None or _factor_delta(EXPERIMENTS[experiment_id], EXPERIMENTS[baseline_id]) != expected_delta:
            errors.append(f"{experiment_id}: adjacent factor delta mismatch")
    for experiment_id in ("N1", "N2", "N3", "N4"):
        experiment = EXPERIMENTS[experiment_id]
        if experiment.required_capabilities != EXPERIMENTS["P5"].required_capabilities:
            errors.append(f"{experiment_id}: capability closure must equal P5")
        expected_n_delta = set() if experiment_id == "N2" else {"negative_rule"}
        if _factor_delta(experiment, EXPERIMENTS["P5"]) != expected_n_delta:
            errors.append(f"{experiment_id}: must vary only negative_rule from P5")
        if set(dict(experiment.fixed_factors)) != set(dict(EXPERIMENTS["P5"].fixed_factors)):
            errors.append(f"{experiment_id}: must retain complete P5 factor snapshot")
        if experiment.metrics != PRETRAIN_METRICS:
            errors.append(f"{experiment_id}: pretraining metrics mismatch")
    for experiment_id in ("P0", "P1", "P2", "P3", "P4", "P5"):
        if EXPERIMENTS[experiment_id].metrics != PRETRAIN_METRICS:
            errors.append(f"{experiment_id}: pretraining metrics mismatch")
    for experiment_id in ("S1", "S2", "S3", "S4", "S5"):
        expected = {"weather_encoder"} if experiment_id == "S1" else {"soil_conditioning"} if experiment_id == "S2" else {"temporal_backbone"} if experiment_id == "S3" else {"remote_fusion"} if experiment_id == "S4" else {"initialization"}
        baseline_id = EXPERIMENTS[experiment_id].baseline_id
        if baseline_id is None or _factor_delta(EXPERIMENTS[experiment_id], EXPERIMENTS[baseline_id]) != expected:
            errors.append(f"{experiment_id}: adjacent factor delta mismatch")
    expected_s_factors = {
        "S0": {"weather_encoder": "linear_grid", "soil_conditioning": "disabled", "temporal_backbone": "shared_simple", "remote_fusion": "none", "initialization": "random"},
        "S1": {"weather_encoder": "grid_vsn", "soil_conditioning": "disabled", "temporal_backbone": "shared_simple", "remote_fusion": "none", "initialization": "random"},
        "S2": {"weather_encoder": "grid_vsn", "soil_conditioning": "enabled", "temporal_backbone": "shared_simple", "remote_fusion": "none", "initialization": "random"},
        "S3": {"weather_encoder": "grid_vsn", "soil_conditioning": "enabled", "temporal_backbone": "tft", "remote_fusion": "none", "initialization": "random"},
        "S4": {"weather_encoder": "grid_vsn", "soil_conditioning": "enabled", "temporal_backbone": "tft", "remote_fusion": "same_grid", "initialization": "random"},
        "S5": {"weather_encoder": "grid_vsn", "soil_conditioning": "enabled", "temporal_backbone": "tft", "remote_fusion": "same_grid", "initialization": "p5"},
    }
    expected_s_capabilities = {
        "S0": S0_CAPABILITIES | {Capability.SUPERVISED_EXPERIMENT_RUNNER},
        "S1": S1_CAPABILITIES | {Capability.SUPERVISED_EXPERIMENT_RUNNER},
        "S2": S2_CAPABILITIES | {Capability.SUPERVISED_EXPERIMENT_RUNNER},
        "S3": S3_CAPABILITIES | {Capability.SUPERVISED_EXPERIMENT_RUNNER},
        "S4": S4_CAPABILITIES | {Capability.SUPERVISED_EXPERIMENT_RUNNER},
        "S5": S5_CAPABILITIES | {Capability.SUPERVISED_EXPERIMENT_RUNNER},
    }
    for experiment_id, expected in expected_s_factors.items():
        factors = dict(EXPERIMENTS[experiment_id].fixed_factors)
        if set(factors) != {"weather_encoder", "soil_conditioning", "temporal_backbone", "remote_fusion", "initialization", "loss", "output"}:
            errors.append(f"{experiment_id}: S factor key schema mismatch")
        if any(factors.get(key) != value for key, value in expected.items()) or factors.get("loss") != "mse" or factors.get("output") != "point":
            errors.append(f"{experiment_id}: exact supervised factor snapshot mismatch")
        if EXPERIMENTS[experiment_id].metrics != SUPERVISED_METRICS:
            errors.append(f"{experiment_id}: supervised metrics mismatch")
        if EXPERIMENTS[experiment_id].required_capabilities != expected_s_capabilities[experiment_id]:
            errors.append(f"{experiment_id}: capability closure mismatch")
    for experiment_id in ("A0", "A1", "A2", "A3", "A4"):
        factors = dict(EXPERIMENTS[experiment_id].fixed_factors)
        if factors.get("backbone") != "S3" or set(factors) != {"backbone", "remote_connection"}:
            errors.append(f"{experiment_id}: A group must fix S3 and vary remote_connection")
        if EXPERIMENTS[experiment_id].metrics != SUPERVISED_METRICS:
            errors.append(f"{experiment_id}: alignment metrics mismatch")
    expected_a_connections = {
        "A0": "none", "A1": "same_grid_current_month", "A2": "county_retrieval_distance",
        "A3": "county_retrieval_no_distance", "A4": "historical_observed_no_future",
    }
    for experiment_id, connection in expected_a_connections.items():
        if dict(EXPERIMENTS[experiment_id].fixed_factors).get("remote_connection") != connection:
            errors.append(f"{experiment_id}: exact remote connection mismatch")
        if EXPERIMENTS[experiment_id].changes != (("remote_connection", connection),):
            errors.append(f"{experiment_id}: exact changes mismatch")
    expected_a_capabilities = {
        "A0": {Capability.SUPERVISED_NO_REMOTE},
        "A1": {Capability.SAME_GRID_SUPERVISED_REMOTE},
        "A2": {Capability.COUNTY_REMOTE_RETRIEVAL_DISTANCE},
        "A3": {Capability.COUNTY_REMOTE_RETRIEVAL_NO_DISTANCE},
        "A4": {Capability.HISTORICAL_REMOTE_ONLY},
    }
    for experiment_id, expected in expected_a_capabilities.items():
        if EXPERIMENTS[experiment_id].required_capabilities != S3_CAPABILITIES | expected | {Capability.SUPERVISED_EXPERIMENT_RUNNER}:
            errors.append(f"{experiment_id}: capability mapping mismatch")
    for experiment_id in ("U0", "U1", "U2", "U3"):
        experiment = EXPERIMENTS[experiment_id]
        if dict(experiment.fixed_factors).get("supervised_structure") != "S4":
            errors.append(f"{experiment_id}: U group must fix supervised_structure=S4")
        if experiment.metrics != PROBABILISTIC_METRICS:
            errors.append(f"{experiment_id}: probabilistic metrics mismatch")
    if [EXPERIMENTS[key].baseline_id for key in ("U0", "U1", "U2", "U3")] != ["S4", "U0", "U1", "U2"]:
        errors.append("U group: recursive baseline chain mismatch")
    expected_u_capabilities = {
        "U0": {Capability.MEAN_MSE_HEAD},
        "U1": {Capability.HETEROSCEDASTIC_BETA_NLL},
        "U2": {Capability.HETEROSCEDASTIC_BETA_NLL, Capability.TEMPORAL_KL},
        "U3": {Capability.HETEROSCEDASTIC_BETA_NLL, Capability.TEMPORAL_KL, Capability.VARIANCE_CALIBRATION},
    }
    for experiment_id, expected in expected_u_capabilities.items():
        if EXPERIMENTS[experiment_id].required_capabilities != S4_CAPABILITIES | expected | {Capability.SUPERVISED_EXPERIMENT_RUNNER}:
            errors.append(f"{experiment_id}: capability mapping mismatch")
    if any(capability in EXPERIMENTS[experiment_id].required_capabilities for experiment_id in ("P1", "P2", "P3", "P4", "P5") for capability in (Capability.TFT_TEMPORAL_BACKBONE, Capability.MEAN_MSE_HEAD)):
        errors.append("P group: unrelated supervised capabilities must not be required")
    return tuple(errors)


def validate_experiment(experiment_id: str) -> tuple[ExperimentDefinition, tuple[str, ...]]:
    experiment = get_experiment(experiment_id)
    errors = list(validate_registry())
    errors.extend(f"{experiment_id}: missing capability {capability.value}" for capability in experiment.missing_capabilities)
    if experiment.runner not in KNOWN_RUNNERS:
        errors.append(f"{experiment_id}: unknown runner; no executable command is registered")
    return experiment, tuple(errors)
