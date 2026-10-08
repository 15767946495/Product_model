from dataclasses import dataclass


DEFAULT_MODEL_IDENTITY = "default_two_stage_pure_cross_attention_beta_nll_temporal_kl"


@dataclass(frozen=True)
class ExperimentProtocol:
    pretrain_years: tuple[int, ...] = (2017, 2018, 2019, 2020)
    finetune_year: int = 2021
    evaluation_year: int = 2022
    states: tuple[str, ...] = (
        "illinois",
        "iowa",
        "louisiana",
        "mississippi",
        "new york",
    )
    seeds: tuple[int, ...] = (17, 42, 73)
    pretrain_manifest: str = "MMST-ViT/data/pretrain_2017_2020.json"
    pretrain_epochs: int = 100
    finetune_epochs: int = 100
    warmup_epochs: int = 5
    checkpoint_rule: str = "final_epoch"
    shared_budget: tuple[tuple[str, str], ...] = (
        ("optimizer", "AdamW"),
        ("pretrain_epochs", "100"),
        ("finetune_epochs", "100"),
        ("warmup_epochs", "5"),
        ("scheduler", "cosine"),
        ("batch_budget", "fixed_within_group"),
        ("augmentation", "fixed_unless_ablation_factor"),
    )
    required_result_metadata: tuple[str, ...] = (
        "experiment_id",
        "seed",
        "git_commit",
        "manifest_hash",
        "config",
        "total_parameters",
        "trainable_parameters",
        "checkpoint",
    )


DEFAULT_PROTOCOL = ExperimentProtocol()
