from enum import Enum
from typing import Iterable


class Capability(str, Enum):
    OFFICIAL_MMST_RUNNER = "official_mmst_runner"
    LEGACY_TFT_PRETRAIN = "legacy_tft_pretrain"
    FIXED_SEASON_TIME_ENCODING = "fixed_season_time_encoding"
    UNPOOLED_DAILY_PRETRAIN_TOKENS = "unpooled_daily_pretrain_tokens"
    SAME_GRID_PATCH_ATTENTION = "same_grid_patch_attention"
    PRETRAIN_METADATA_BATCH = "pretrain_metadata_batch"
    MASKED_TG_NT_XENT = "masked_tg_nt_xent"
    LINEAR_GRID_WEATHER_ENCODER = "linear_grid_weather_encoder"
    GRID_VSN = "grid_vsn"
    SOIL_CONDITIONING = "soil_conditioning"
    TFT_TEMPORAL_BACKBONE = "tft_temporal_backbone"
    SUPERVISED_NO_REMOTE = "supervised_no_remote"
    COUNTY_REMOTE_RETRIEVAL_DISTANCE = "county_remote_retrieval_distance"
    COUNTY_REMOTE_RETRIEVAL_NO_DISTANCE = "county_remote_retrieval_no_distance"
    SAME_GRID_SUPERVISED_REMOTE = "same_grid_supervised_remote"
    HISTORICAL_REMOTE_ONLY = "historical_remote_only"
    MEAN_MSE_HEAD = "mean_mse_head"
    HETEROSCEDASTIC_BETA_NLL = "heteroscedastic_beta_nll"
    TEMPORAL_KL = "temporal_kl"
    VARIANCE_CALIBRATION = "variance_calibration"
    P5_WEIGHT_TRANSFER = "p5_weight_transfer"
    SUPERVISED_EXPERIMENT_RUNNER = "supervised_experiment_runner"


AVAILABLE_CAPABILITIES = frozenset(
    {
        Capability.OFFICIAL_MMST_RUNNER,
        Capability.FIXED_SEASON_TIME_ENCODING,
        Capability.UNPOOLED_DAILY_PRETRAIN_TOKENS,
        Capability.SAME_GRID_PATCH_ATTENTION,
        Capability.LINEAR_GRID_WEATHER_ENCODER,
        Capability.GRID_VSN,
        Capability.SOIL_CONDITIONING,
        Capability.TFT_TEMPORAL_BACKBONE,
        Capability.COUNTY_REMOTE_RETRIEVAL_DISTANCE,
        Capability.SUPERVISED_NO_REMOTE,
        Capability.MEAN_MSE_HEAD,
        Capability.HETEROSCEDASTIC_BETA_NLL,
        Capability.TEMPORAL_KL,
    }
)


def missing_capabilities(required: Iterable[Capability]) -> tuple[Capability, ...]:
    """Return unavailable requirements in Capability declaration order."""
    missing = set(required) - AVAILABLE_CAPABILITIES
    return tuple(capability for capability in Capability if capability in missing)
