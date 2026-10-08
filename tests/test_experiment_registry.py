import unittest
from dataclasses import FrozenInstanceError
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from unittest.mock import patch

import experiments
import experiments.cli as cli
from experiments.cli import main
from experiments.capabilities import (
    AVAILABLE_CAPABILITIES,
    Capability,
    missing_capabilities,
)
from experiments.protocol import DEFAULT_PROTOCOL, ExperimentProtocol
from experiments.registry import (
    EXECUTABLE_RUNNERS,
    EXPERIMENTS,
    ExperimentDefinition,
    ExperimentStatus,
    get_experiment,
    validate_experiment,
    validate_registry,
)


PRETRAIN_METRICS = {
    "nt_xent_train", "positive_cosine", "negative_cosine",
    "similarity_margin", "embedding_std", "covariance_spectrum", "daily_metrics",
    "patch_attention", "negative_source_ratio", "linear_probe_metric",
    "downstream_test_rmse",
}
SUPERVISED_METRICS = {
    "rmse", "mae", "r2", "corr", "parameter_count", "training_time",
}
PROBABILISTIC_METRICS = {
    "rmse", "mae", "r2", "corr", "parameter_count", "training_time",
    "nll", "interval_coverage", "interval_width",
}


EXPECTED_IDS = {
    "P0", "P1", "P2", "P3", "P4", "P5",
    "N1", "N2", "N3", "N4",
    "S0", "S1", "S2", "S3", "S4", "S5",
    "A0", "A1", "A2", "A3", "A4",
    "U0", "U1", "U2", "U3",
}


class ProtocolTest(unittest.TestCase):
    def test_package_root_exports_public_interface(self):
        expected = (
            "ExperimentProtocol",
            "DEFAULT_PROTOCOL",
            "Capability",
            "AVAILABLE_CAPABILITIES",
            "missing_capabilities",
            "ExperimentStatus",
            "ExperimentDefinition",
            "EXPERIMENTS",
            "get_experiment",
            "validate_registry",
            "validate_experiment",
        )
        self.assertEqual(getattr(experiments, "__all__"), expected)
        for name in expected:
            self.assertIs(getattr(experiments, name), globals()[name])

    def test_default_protocol_reserves_2022_for_independent_test(self):
        self.assertEqual(DEFAULT_PROTOCOL.finetune_year, 2021)
        self.assertEqual(DEFAULT_PROTOCOL.evaluation_year, 2022)
        self.assertNotIn(2022, DEFAULT_PROTOCOL.pretrain_years)

    def test_default_protocol_uses_five_supervised_states(self):
        self.assertEqual(
            DEFAULT_PROTOCOL.states,
            ("illinois", "iowa", "louisiana", "mississippi", "new york"),
        )

    def test_default_protocol_uses_exact_training_inputs_and_seeds(self):
        self.assertEqual(DEFAULT_PROTOCOL.pretrain_years, (2017, 2018, 2019, 2020))
        self.assertEqual(DEFAULT_PROTOCOL.finetune_year, 2021)
        self.assertEqual(DEFAULT_PROTOCOL.evaluation_year, 2022)
        self.assertEqual(DEFAULT_PROTOCOL.seeds, (17, 42, 73))
        self.assertEqual(
            DEFAULT_PROTOCOL.pretrain_manifest,
            "MMST-ViT/data/pretrain_2017_2020.json",
        )

    def test_default_protocol_uses_exact_checkpoint_and_budget_rules(self):
        self.assertEqual(DEFAULT_PROTOCOL.pretrain_epochs, 100)
        self.assertEqual(DEFAULT_PROTOCOL.finetune_epochs, 100)
        self.assertEqual(DEFAULT_PROTOCOL.warmup_epochs, 5)
        self.assertEqual(DEFAULT_PROTOCOL.checkpoint_rule, "final_epoch")
        self.assertEqual(
            DEFAULT_PROTOCOL.shared_budget,
            (
                ("optimizer", "AdamW"),
                ("pretrain_epochs", "100"),
                ("finetune_epochs", "100"),
                ("warmup_epochs", "5"),
                ("scheduler", "cosine"),
                ("batch_budget", "fixed_within_group"),
                ("augmentation", "fixed_unless_ablation_factor"),
            ),
        )

    def test_protocol_does_not_expose_validation_selection_semantics(self):
        self.assertNotIn("supervised_train_years", ExperimentProtocol.__dataclass_fields__)
        self.assertNotIn("validation_year", ExperimentProtocol.__dataclass_fields__)
        self.assertNotIn("test_year", ExperimentProtocol.__dataclass_fields__)
        self.assertEqual(DEFAULT_PROTOCOL.checkpoint_rule, "final_epoch")

    def test_default_protocol_requires_exact_result_metadata(self):
        self.assertEqual(
            DEFAULT_PROTOCOL.required_result_metadata,
            (
                "experiment_id",
                "seed",
                "git_commit",
                "manifest_hash",
                "config",
                "total_parameters",
                "trainable_parameters",
                "checkpoint",
            ),
        )

    def test_experiment_protocol_is_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            setattr(DEFAULT_PROTOCOL, "evaluation_year", 2023)

    def test_capability_enum_is_complete(self):
        expected = (
            ("OFFICIAL_MMST_RUNNER", "official_mmst_runner"),
            ("LEGACY_TFT_PRETRAIN", "legacy_tft_pretrain"),
            ("FIXED_SEASON_TIME_ENCODING", "fixed_season_time_encoding"),
            ("UNPOOLED_DAILY_PRETRAIN_TOKENS", "unpooled_daily_pretrain_tokens"),
            ("SAME_GRID_PATCH_ATTENTION", "same_grid_patch_attention"),
            ("PRETRAIN_METADATA_BATCH", "pretrain_metadata_batch"),
            ("MASKED_TG_NT_XENT", "masked_tg_nt_xent"),
            ("LINEAR_GRID_WEATHER_ENCODER", "linear_grid_weather_encoder"),
            ("GRID_VSN", "grid_vsn"),
            ("SOIL_CONDITIONING", "soil_conditioning"),
            ("TFT_TEMPORAL_BACKBONE", "tft_temporal_backbone"),
            ("SUPERVISED_NO_REMOTE", "supervised_no_remote"),
            ("COUNTY_REMOTE_RETRIEVAL_DISTANCE", "county_remote_retrieval_distance"),
            ("COUNTY_REMOTE_RETRIEVAL_NO_DISTANCE", "county_remote_retrieval_no_distance"),
            ("SAME_GRID_SUPERVISED_REMOTE", "same_grid_supervised_remote"),
            ("HISTORICAL_REMOTE_ONLY", "historical_remote_only"),
            ("MEAN_MSE_HEAD", "mean_mse_head"),
            ("HETEROSCEDASTIC_BETA_NLL", "heteroscedastic_beta_nll"),
            ("TEMPORAL_KL", "temporal_kl"),
            ("VARIANCE_CALIBRATION", "variance_calibration"),
            ("P5_WEIGHT_TRANSFER", "p5_weight_transfer"),
            ("SUPERVISED_EXPERIMENT_RUNNER", "supervised_experiment_runner"),
        )
        self.assertEqual(
            tuple((capability.name, capability.value) for capability in Capability),
            expected,
        )

    def test_unimplemented_capabilities_are_not_available(self):
        unavailable = {
            Capability.PRETRAIN_METADATA_BATCH,
            Capability.MASKED_TG_NT_XENT,
            Capability.SAME_GRID_SUPERVISED_REMOTE,
            Capability.VARIANCE_CALIBRATION,
            Capability.P5_WEIGHT_TRANSFER,
            Capability.COUNTY_REMOTE_RETRIEVAL_NO_DISTANCE,
            Capability.SUPERVISED_EXPERIMENT_RUNNER,
        }
        self.assertTrue(unavailable.isdisjoint(AVAILABLE_CAPABILITIES))

    def test_missing_capabilities_reports_unimplemented_pretraining_features(self):
        missing = missing_capabilities(
            {
                Capability.FIXED_SEASON_TIME_ENCODING,
                Capability.PRETRAIN_METADATA_BATCH,
                Capability.MASKED_TG_NT_XENT,
            }
        )
        self.assertEqual(
            missing,
            (
                Capability.PRETRAIN_METADATA_BATCH,
                Capability.MASKED_TG_NT_XENT,
            ),
        )

    def test_missing_capabilities_handles_empty_input(self):
        self.assertEqual(missing_capabilities(()), ())

    def test_missing_capabilities_deduplicates_and_uses_declaration_order(self):
        first = missing_capabilities(
            [
                Capability.P5_WEIGHT_TRANSFER,
                Capability.MASKED_TG_NT_XENT,
                Capability.P5_WEIGHT_TRANSFER,
                Capability.PRETRAIN_METADATA_BATCH,
            ]
        )
        second = missing_capabilities(
            [
                Capability.PRETRAIN_METADATA_BATCH,
                Capability.P5_WEIGHT_TRANSFER,
                Capability.MASKED_TG_NT_XENT,
            ]
        )
        expected = (
            Capability.PRETRAIN_METADATA_BATCH,
            Capability.MASKED_TG_NT_XENT,
            Capability.P5_WEIGHT_TRANSFER,
        )
        self.assertEqual(first, expected)
        self.assertEqual(second, expected)


class RegistryTest(unittest.TestCase):
    def test_pretraining_metrics_do_not_claim_validation_set(self):
        self.assertNotIn("nt_xent_val", PRETRAIN_METRICS)
    def test_registry_contains_complete_matrix(self):
        self.assertEqual(set(EXPERIMENTS), EXPECTED_IDS)
        self.assertEqual(len({item.output_name for item in EXPERIMENTS.values()}), 25)

    def test_registry_definitions_are_complete_and_baselines_exist(self):
        for experiment_id, experiment in EXPERIMENTS.items():
            self.assertEqual(experiment.experiment_id, experiment_id)
            self.assertTrue(experiment.research_question)
            self.assertTrue(experiment.metrics)
            self.assertTrue(experiment.fixed_factors)
            self.assertTrue(experiment.changes)
            self.assertTrue(experiment.output_name)
            if experiment.baseline_id is not None:
                self.assertIn(experiment.baseline_id, EXPERIMENTS)

    def test_p_chain_uses_cumulative_factor_maps(self):
        self.assertEqual(dict(EXPERIMENTS["P1"].fixed_factors)["sequence_position"], "none")
        self.assertEqual(dict(EXPERIMENTS["P2"].fixed_factors)["sequence_position"], "none")
        self.assertEqual(dict(EXPERIMENTS["P3"].fixed_factors)["temporal_pooling"], "none")
        self.assertEqual(dict(EXPERIMENTS["P4"].fixed_factors)["spatial_alignment"], "strict_same_grid")
        self.assertEqual(dict(EXPERIMENTS["P5"].fixed_factors)["contrast_unit"], "day_grid")
        self.assertEqual(dict(EXPERIMENTS["P5"].fixed_factors)["negative_rule"], "same_county_space_time")
        self.assertEqual(EXPERIMENTS["P5"].baseline_id, "P4")

    def test_p_roots_have_exact_semantics(self):
        self.assertIsNone(EXPERIMENTS["P0"].baseline_id)
        self.assertEqual(EXPERIMENTS["P0"].changes, (("model", "official_mmst"),))
        self.assertEqual(EXPERIMENTS["P1"].baseline_id, "P0")
        self.assertEqual(EXPERIMENTS["P1"].changes, (("model", "tft"),))
        self.assertEqual(dict(EXPERIMENTS["P0"].fixed_factors), dict(EXPERIMENTS["P1"].fixed_factors))

    def test_pretraining_groups_use_required_metrics_and_explicit_questions(self):
        for experiment_id in ("P0", "P1", "P2", "P3", "P4", "P5", "N1", "N2", "N3", "N4"):
            experiment = EXPERIMENTS[experiment_id]
            self.assertTrue(PRETRAIN_METRICS.issubset(set(experiment.metrics)))
            self.assertTrue(any(char > "\x7f" for char in experiment.research_question))
            self.assertNotIn("What is the effect", experiment.research_question)

    def test_n_group_maps_complete_p5_snapshot_and_exact_rules(self):
        expected_rules = {
            "N1": "all_batch",
            "N2": "same_county_space_time",
            "N3": "cross_county_only",
            "N4": "hierarchical_same_county_then_cross_county",
        }
        p5_factors = dict(EXPERIMENTS["P5"].fixed_factors)
        for experiment_id, rule in expected_rules.items():
            experiment = EXPERIMENTS[experiment_id]
            factors = dict(experiment.fixed_factors)
            self.assertEqual(experiment.baseline_id, "P5")
            self.assertEqual(set(factors), set(p5_factors))
            self.assertEqual({key: value for key, value in factors.items() if key != "negative_rule"},
                             {key: value for key, value in p5_factors.items() if key != "negative_rule"})
            self.assertEqual(factors["negative_rule"], rule)
            self.assertEqual(experiment.changes, (("negative_rule", rule),))
        self.assertEqual(EXPERIMENTS["N2"].fixed_factors, EXPERIMENTS["P5"].fixed_factors)

    def test_s_group_is_exact_cumulative_chain(self):
        expected = {
            "S0": {"weather_encoder": "linear_grid", "soil_conditioning": "disabled", "temporal_backbone": "shared_simple", "remote_fusion": "none", "initialization": "random"},
            "S1": {"weather_encoder": "grid_vsn", "soil_conditioning": "disabled", "temporal_backbone": "shared_simple", "remote_fusion": "none", "initialization": "random"},
            "S2": {"weather_encoder": "grid_vsn", "soil_conditioning": "enabled", "temporal_backbone": "shared_simple", "remote_fusion": "none", "initialization": "random"},
            "S3": {"weather_encoder": "grid_vsn", "soil_conditioning": "enabled", "temporal_backbone": "tft", "remote_fusion": "none", "initialization": "random"},
            "S4": {"weather_encoder": "grid_vsn", "soil_conditioning": "enabled", "temporal_backbone": "tft", "remote_fusion": "same_grid", "initialization": "random"},
            "S5": {"weather_encoder": "grid_vsn", "soil_conditioning": "enabled", "temporal_backbone": "tft", "remote_fusion": "same_grid", "initialization": "p5"},
        }
        for experiment_id, factors in expected.items():
            actual = dict(EXPERIMENTS[experiment_id].fixed_factors)
            self.assertEqual(set(actual), set(factors) | {"loss", "output"})
            self.assertEqual({key: actual[key] for key in factors}, factors)
            self.assertEqual({key: actual[key] for key in ("loss", "output")}, {"loss": "mse", "output": "point"})
        for experiment_id in ("S1", "S2", "S3", "S4", "S5"):
            current = dict(EXPERIMENTS[experiment_id].fixed_factors)
            previous = dict(EXPERIMENTS[EXPERIMENTS[experiment_id].baseline_id].fixed_factors)
            delta = {key for key in current if current[key] != previous[key]}
            expected_delta = {"weather_encoder"} if experiment_id == "S1" else {"soil_conditioning"} if experiment_id == "S2" else {"temporal_backbone"} if experiment_id == "S3" else {"remote_fusion"} if experiment_id == "S4" else {"initialization"}
            self.assertEqual(delta, expected_delta)
        self.assertEqual(EXPERIMENTS["S4"].effective_status, ExperimentStatus.BLOCKED)
        self.assertIn(Capability.SAME_GRID_SUPERVISED_REMOTE, EXPERIMENTS["S4"].missing_capabilities)
        self.assertIn(Capability.P5_WEIGHT_TRANSFER, EXPERIMENTS["S5"].missing_capabilities)

    def test_s_a_u_n_capabilities_are_cumulative_closures(self):
        expected_s = {
            "S0": {Capability.LINEAR_GRID_WEATHER_ENCODER, Capability.SUPERVISED_EXPERIMENT_RUNNER},
            "S1": {Capability.LINEAR_GRID_WEATHER_ENCODER, Capability.GRID_VSN, Capability.SUPERVISED_EXPERIMENT_RUNNER},
            "S2": {Capability.LINEAR_GRID_WEATHER_ENCODER, Capability.GRID_VSN, Capability.SOIL_CONDITIONING, Capability.SUPERVISED_EXPERIMENT_RUNNER},
            "S3": {Capability.LINEAR_GRID_WEATHER_ENCODER, Capability.GRID_VSN, Capability.SOIL_CONDITIONING, Capability.TFT_TEMPORAL_BACKBONE, Capability.SUPERVISED_EXPERIMENT_RUNNER},
            "S4": {Capability.LINEAR_GRID_WEATHER_ENCODER, Capability.GRID_VSN, Capability.SOIL_CONDITIONING, Capability.TFT_TEMPORAL_BACKBONE, Capability.SAME_GRID_SUPERVISED_REMOTE, Capability.SUPERVISED_EXPERIMENT_RUNNER},
            "S5": {Capability.LINEAR_GRID_WEATHER_ENCODER, Capability.GRID_VSN, Capability.SOIL_CONDITIONING, Capability.TFT_TEMPORAL_BACKBONE, Capability.SAME_GRID_SUPERVISED_REMOTE, Capability.P5_WEIGHT_TRANSFER, Capability.SUPERVISED_EXPERIMENT_RUNNER},
        }
        for experiment_id, capabilities in expected_s.items():
            self.assertEqual(EXPERIMENTS[experiment_id].required_capabilities, frozenset(capabilities))
        s3 = expected_s["S3"]
        for experiment_id in ("A0", "A1", "A2", "A3", "A4"):
            connection = EXPERIMENTS[experiment_id].required_capabilities - s3
            self.assertEqual(EXPERIMENTS[experiment_id].required_capabilities, frozenset(s3) | connection)
        s4 = expected_s["S4"]
        for experiment_id in ("U0", "U1", "U2", "U3"):
            self.assertTrue(frozenset(s4).issubset(EXPERIMENTS[experiment_id].required_capabilities))
        p5 = EXPERIMENTS["P5"].required_capabilities
        for experiment_id in ("N1", "N2", "N3", "N4"):
            self.assertEqual(EXPERIMENTS[experiment_id].required_capabilities, p5)

    def test_s_factor_schema_has_only_brief_keys_plus_fixed_outputs(self):
        expected_keys = {
            "weather_encoder", "soil_conditioning", "temporal_backbone",
            "remote_fusion", "initialization", "loss", "output",
        }
        for experiment_id in ("S0", "S1", "S2", "S3", "S4", "S5"):
            self.assertEqual(set(dict(EXPERIMENTS[experiment_id].fixed_factors)), expected_keys)

    def test_alignment_and_uncertainty_groups_are_exact(self):
        expected_alignment = {
            "A0": "none", "A1": "same_grid_current_month",
            "A2": "county_retrieval_distance", "A3": "county_retrieval_no_distance",
            "A4": "historical_observed_no_future",
        }
        for experiment_id, connection in expected_alignment.items():
            experiment = EXPERIMENTS[experiment_id]
            self.assertEqual(dict(experiment.fixed_factors), {"backbone": "S3", "remote_connection": connection})
            self.assertEqual(experiment.changes, (("remote_connection", connection),))
            self.assertEqual(experiment.metrics, tuple(sorted(SUPERVISED_METRICS)))
        self.assertEqual(EXPERIMENTS["A2"].effective_status, ExperimentStatus.BLOCKED)
        self.assertIn(Capability.SUPERVISED_EXPERIMENT_RUNNER, EXPERIMENTS["A2"].missing_capabilities)
        self.assertEqual(EXPERIMENTS["A3"].effective_status, ExperimentStatus.BLOCKED)
        self.assertIn(Capability.COUNTY_REMOTE_RETRIEVAL_NO_DISTANCE, EXPERIMENTS["A3"].missing_capabilities)
        for experiment_id in ("U0", "U1", "U2", "U3"):
            self.assertEqual(dict(EXPERIMENTS[experiment_id].fixed_factors)["supervised_structure"], "S4")
            self.assertEqual(EXPERIMENTS[experiment_id].metrics, tuple(sorted(PROBABILISTIC_METRICS)))
        self.assertEqual([EXPERIMENTS[key].baseline_id for key in ("U0", "U1", "U2", "U3")], ["S4", "U0", "U1", "U2"])
        self.assertEqual(EXPERIMENTS["U0"].changes, (("prediction_head", "mean"), ("loss", "mse")))
        self.assertEqual(EXPERIMENTS["U1"].changes, (("prediction_head", "mean_variance"), ("loss", "beta_nll")))
        self.assertEqual(EXPERIMENTS["U2"].changes, (("loss", "beta_nll_temporal_kl"),))
        self.assertEqual(EXPERIMENTS["U3"].changes, (("calibration", "variance"),))

    def test_s_chain_and_group_fixed_factors_are_explicit(self):
        self.assertEqual(EXPERIMENTS["S2"].baseline_id, "S1")
        self.assertEqual(EXPERIMENTS["S5"].baseline_id, "S4")
        for experiment_id in ("S0", "S1", "S2", "S3", "S4", "S5"):
            factors = dict(EXPERIMENTS[experiment_id].fixed_factors)
            self.assertEqual(factors["loss"], "mse")
            self.assertEqual(factors["output"], "point")
        for experiment_id in ("A0", "A1", "A2", "A3", "A4"):
            self.assertEqual(dict(EXPERIMENTS[experiment_id].fixed_factors)["backbone"], "S3")
        for experiment_id in ("U0", "U1", "U2", "U3"):
            self.assertEqual(
                dict(EXPERIMENTS[experiment_id].fixed_factors)["supervised_structure"],
                "S4",
            )

    def test_u_experiments_record_head_loss_and_calibration(self):
        for experiment_id in ("U0", "U1", "U2", "U3"):
            factors = dict(EXPERIMENTS[experiment_id].fixed_factors)
            self.assertIn("prediction_head", factors)
            self.assertIn("loss", factors)
            self.assertIn("calibration", factors)

    def test_remote_capabilities_match_connection_type(self):
        self.assertEqual(EXPERIMENTS["A1"].effective_status, ExperimentStatus.BLOCKED)
        self.assertEqual(EXPERIMENTS["A2"].effective_status, ExperimentStatus.BLOCKED)
        self.assertEqual(EXPERIMENTS["A3"].effective_status, ExperimentStatus.BLOCKED)
        self.assertIn(Capability.SAME_GRID_SUPERVISED_REMOTE, EXPERIMENTS["A1"].missing_capabilities)
        self.assertIn(Capability.COUNTY_REMOTE_RETRIEVAL_NO_DISTANCE, EXPERIMENTS["A3"].missing_capabilities)

    def test_supervised_experiments_are_blocked_without_a_specific_runner(self):
        for experiment_id in ("S0", "S1", "S2", "S3", "S4", "S5", "A0", "A1", "A2", "A3", "A4", "U0", "U1", "U2", "U3"):
            experiment = EXPERIMENTS[experiment_id]
            self.assertEqual(experiment.effective_status, ExperimentStatus.BLOCKED)
            self.assertIn(Capability.SUPERVISED_EXPERIMENT_RUNNER, experiment.missing_capabilities)

    def test_status_is_blocked_when_capability_is_missing(self):
        self.assertEqual(EXPERIMENTS["P5"].effective_status, ExperimentStatus.BLOCKED)
        self.assertIn(Capability.PRETRAIN_METADATA_BATCH, EXPERIMENTS["P5"].missing_capabilities)
        self.assertIn(Capability.MASKED_TG_NT_XENT, EXPERIMENTS["P5"].missing_capabilities)

    def test_external_and_analysis_only_do_not_bypass_missing_capabilities(self):
        external = ExperimentDefinition(
            experiment_id="X1", group="X", title="external", research_question="q",
            baseline_id=None, changes=(("x", "y"),), fixed_factors=(("x", "y"),),
            required_capabilities=frozenset({Capability.P5_WEIGHT_TRANSFER}),
            metrics=("rmse",), output_name="x1", runner="official_mmst",
            declared_status=ExperimentStatus.EXTERNAL,
        )
        analysis = ExperimentDefinition(
            experiment_id="X2", group="X", title="analysis", research_question="q",
            baseline_id=None, changes=(("x", "y"),), fixed_factors=(("x", "y"),),
            required_capabilities=frozenset({Capability.VARIANCE_CALIBRATION}),
            metrics=("rmse",), output_name="x2", runner=None,
            declared_status=ExperimentStatus.ANALYSIS_ONLY,
        )
        self.assertEqual(external.effective_status, ExperimentStatus.BLOCKED)
        self.assertEqual(analysis.effective_status, ExperimentStatus.BLOCKED)
        complete_analysis = ExperimentDefinition(
            experiment_id="X6", group="X", title="analysis", research_question="q",
            baseline_id=None, changes=(("x", "y"),), fixed_factors=(("x", "y"),),
            required_capabilities=frozenset({Capability.LINEAR_GRID_WEATHER_ENCODER}),
            metrics=("rmse",), output_name="x6", runner=None,
            declared_status=ExperimentStatus.ANALYSIS_ONLY,
        )
        self.assertEqual(complete_analysis.effective_status, ExperimentStatus.ANALYSIS_ONLY)

    def test_unknown_runner_is_blocked_even_when_capabilities_are_available(self):
        experiment = ExperimentDefinition(
            experiment_id="X4", group="X", title="unknown runner", research_question="q",
            baseline_id=None, changes=(("x", "y"),), fixed_factors=(("x", "y"),),
            required_capabilities=frozenset({Capability.LINEAR_GRID_WEATHER_ENCODER}),
            metrics=("rmse",), output_name="x4", runner="unknown_runner",
        )
        self.assertEqual(experiment.effective_status, ExperimentStatus.BLOCKED)
        self.assertEqual(experiment.missing_capabilities, ())

    def test_registry_and_known_experiment_validate(self):
        self.assertEqual(validate_registry(), ())
        self.assertEqual(EXPERIMENTS["P0"].runner, "official_mmst")
        self.assertEqual(EXPERIMENTS["P0"].declared_status, ExperimentStatus.EXTERNAL)

        experiment, errors = validate_experiment("P5")
        self.assertIs(experiment, EXPERIMENTS["P5"])
        self.assertTrue(any("pretrain_metadata_batch" in error for error in errors))
        with self.assertRaises(KeyError):
            get_experiment("missing")

    def test_legacy_tft_pretraining_is_not_an_executable_registry_runner(self):
        self.assertNotIn("legacy_tft_pretrain", EXECUTABLE_RUNNERS)
        self.assertIsNone(EXPERIMENTS["P1"].runner)
        self.assertEqual(EXPERIMENTS["P1"].effective_status, ExperimentStatus.BLOCKED)

    def test_validate_registry_checks_exact_schema_contract(self):
        self.assertEqual(validate_registry(), ())
        for experiment_id in EXPECTED_IDS:
            self.assertEqual(EXPERIMENTS[experiment_id].group, experiment_id[0])
            self.assertTrue(EXPERIMENTS[experiment_id].changes)
        self.assertEqual(set(dict(EXPERIMENTS["S0"].fixed_factors)), {
            "weather_encoder", "soil_conditioning", "temporal_backbone", "remote_fusion",
            "initialization", "loss", "output",
        })
        self.assertIn("本注册版本固定 S4", EXPERIMENTS["U0"].research_question)
        self.assertIn("另一固定协议", EXPERIMENTS["U0"].research_question)


class CliTest(unittest.TestCase):
    def run_cli(self, *arguments):
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            status = main(arguments)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_list_prints_all_experiment_ids(self):
        status, stdout, _ = self.run_cli("list")

        self.assertEqual(status, 0)
        rows = stdout.splitlines()
        self.assertEqual(len(rows), 25)
        self.assertEqual({row.split("\t", 1)[0] for row in rows}, EXPECTED_IDS)
        for row in rows:
            experiment_id, group, title, status_text = row.split("\t", 3)
            self.assertIn(experiment_id, EXPECTED_IDS)
            self.assertEqual(group, EXPERIMENTS[experiment_id].group)
            self.assertEqual(title, EXPERIMENTS[experiment_id].title)
            self.assertEqual(status_text, EXPERIMENTS[experiment_id].effective_status.value)

    def test_show_prints_factor_and_status_json(self):
        status, stdout, _ = self.run_cli("show", "P5")

        self.assertEqual(status, 0)
        payload = json.loads(stdout)
        self.assertEqual(payload["experiment_id"], "P5")
        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["fixed_factors"]["negative_rule"], "same_county_space_time")

    def test_show_json_exposes_fixed_protocol_fields(self):
        status, stdout, _ = self.run_cli("show", "P1")

        self.assertEqual(status, 0)
        protocol = json.loads(stdout)["protocol"]
        self.assertEqual(protocol["pretrain_years"], [2017, 2018, 2019, 2020])
        self.assertEqual(protocol["finetune_year"], 2021)
        self.assertEqual(protocol["evaluation_year"], 2022)
        self.assertEqual(protocol["pretrain_epochs"], 100)
        self.assertEqual(protocol["finetune_epochs"], 100)
        self.assertEqual(protocol["warmup_epochs"], 5)
        self.assertEqual(protocol["checkpoint_rule"], "final_epoch")

    def test_unknown_experiment_returns_nonzero(self):
        status, _, stderr = self.run_cli("show", "missing")

        self.assertNotEqual(status, 0)
        payload = json.loads(stderr)
        self.assertEqual(payload, {"error": "unknown experiment: missing"})

    def test_p0_dry_run_returns_official_command_without_execution(self):
        with patch("subprocess.run") as run, patch("builtins.open") as open_file:
            status, stdout, _ = self.run_cli("dry-run", "P0")

        payload = json.loads(stdout)
        self.assertEqual(status, 0)
        self.assertEqual(set(payload), {"experiment_id", "status", "output_dir", "protocol", "command"})
        self.assertEqual(payload["experiment_id"], "P0")
        self.assertEqual(payload["status"], "external")
        self.assertEqual(payload["command"], ["python", "MMST-ViT/pretrain_current.py"])
        run.assert_not_called()
        open_file.assert_not_called()
        experiment = EXPERIMENTS["P0"]
        with patch.object(cli, "validate_experiment", return_value=(experiment, ("registry: invalid",))):
            status, stdout, _ = self.run_cli("dry-run", "P0")
        payload = json.loads(stdout)
        self.assertNotEqual(status, 0)
        self.assertEqual(payload["status"], "external")
        self.assertIn("registry: invalid", payload["errors"])
        self.assertNotIn("command", payload)

    def test_p1_legacy_pretraining_has_no_executable_command(self):
        status, stdout, _ = self.run_cli("dry-run", "P1")

        payload = json.loads(stdout)
        self.assertNotEqual(status, 0)
        self.assertEqual(payload["status"], "blocked")
        self.assertNotIn("command", payload)

    def test_dry_run_unknown_runner_is_safe_and_has_no_command(self):
        experiment = ExperimentDefinition(
            experiment_id="X3", group="X", title="unknown runner", research_question="q",
            baseline_id=None, changes=(("x", "y"),), fixed_factors=(("x", "y"),),
            required_capabilities=frozenset(), metrics=("rmse",), output_name="x3",
            runner="unknown_runner",
        )
        with patch.object(
            cli,
            "validate_experiment",
            return_value=(experiment, ("X3: unknown runner; no executable command is registered",)),
        ):
            status, stdout, _ = self.run_cli("dry-run", "X3")

        payload = json.loads(stdout)
        self.assertNotEqual(status, 0)
        self.assertEqual(set(payload), {"experiment_id", "status", "missing_capabilities", "errors"})
        self.assertEqual(payload["status"], "blocked")
        self.assertIn("unknown runner", payload["errors"][0])
        self.assertNotIn("command", payload)

    def test_validate_unknown_runner_reports_runner_error(self):
        experiment = ExperimentDefinition(
            experiment_id="X5", group="X", title="unknown runner", research_question="q",
            baseline_id=None, changes=(("x", "y"),), fixed_factors=(("x", "y"),),
            required_capabilities=frozenset(), metrics=("rmse",), output_name="x5",
            runner="unknown_runner",
        )
        with patch.object(cli, "validate_experiment", return_value=(experiment, ("X5: unknown runner; no executable command is registered",))):
            status, stdout, _ = self.run_cli("validate", "X5")
        payload = json.loads(stdout)
        self.assertNotEqual(status, 0)
        self.assertEqual(payload["status"], "blocked")
        self.assertIn("unknown runner", " ".join(payload["errors"]))
    def test_validate_blocked_experiment_lists_missing_capabilities(self):
        status, stdout, _ = self.run_cli("validate", "P5")

        self.assertNotEqual(status, 0)
        payload = json.loads(stdout)
        self.assertEqual(payload["status"], "blocked")
        self.assertIn("pretrain_metadata_batch", payload["missing_capabilities"])
        self.assertIn("masked_tg_nt_xent", payload["missing_capabilities"])

    def test_validate_ready_external_has_exact_success_fields(self):
        status, stdout, _ = self.run_cli("validate", "P0")

        payload = json.loads(stdout)
        self.assertEqual(status, 0)
        self.assertEqual(set(payload), {"experiment_id", "status", "missing_capabilities", "errors"})
        self.assertEqual(payload["status"], "external")
        self.assertEqual(payload["missing_capabilities"], [])
        self.assertEqual(payload["errors"], [])

    def test_dry_run_blocked_experiment_does_not_emit_command(self):
        status, stdout, _ = self.run_cli("dry-run", "P5")

        self.assertNotEqual(status, 0)
        payload = json.loads(stdout)
        self.assertEqual(set(payload), {"experiment_id", "status", "missing_capabilities", "errors"})
        self.assertEqual(payload["status"], "blocked")
        self.assertIn("missing_capabilities", payload)
        self.assertNotIn("command", payload)

    def test_dry_run_internal_experiments_never_emits_a_command(self):
        for experiment_id in EXPECTED_IDS - {"P0"}:
            status, stdout, _ = self.run_cli("dry-run", experiment_id)
            payload = json.loads(stdout)
            self.assertNotEqual(status, 0)
            self.assertEqual(payload["status"], "blocked")
            self.assertIn("missing_capabilities", payload)
            self.assertNotIn("command", payload)


class PaperConsistencyTest(unittest.TestCase):
    PAPER = Path(__file__).resolve().parents[1] / "paper.md"

    def test_conclusion_is_evidence_dependent_and_covers_registered_questions(self):
        paper = self.PAPER.read_text(encoding="utf-8")
        conclusion = paper[paper.index("## 6 结论与展望"):paper.index("## 参考文献")]

        required_phrases = (
            "逐日网格对比", "纯跨模态注意力", "P5", "网格 VSN", "土壤条件化",
            "TFT", "同网格遥感融合", "概率头", "时间 KL", "NLL", "覆盖率",
            "纯跨模态注意力不等于跨日时序建模", "中间时间步没有独立真实产量标签",
            "共享目标分布假设", "负样本独立性只是近似",
            "县级土壤仍是粗粒度",
        )
        for phrase in required_phrases:
            self.assertIn(phrase, conclusion)

        self.assertIn("只有在", conclusion)
        self.assertIn("待填写", conclusion)
        self.assertNotIn("实验结果表明", conclusion)
        self.assertNotIn("显著改善", conclusion)

    def test_paper_distinguishes_pretraining_patch_and_supervised_remote_token(self):
        paper = self.PAPER.read_text(encoding="utf-8")
        self.assertIn("同网格遥感 patch", paper)
        self.assertIn("同网格遥感 token", paper)

    def test_paper_has_formal_experiment_heading_and_year_protocol(self):
        paper = self.PAPER.read_text(encoding="utf-8")
        self.assertIn("\n## 5 实验设计\n", paper)
        self.assertNotRegex(paper, r"监督微调限定在五州 2021|监督数据集为五州 2021|五州 2021 年有标签数据上进行监督微调")
        self.assertIn("预训练集为五州 2017—2020 年", paper)
        self.assertIn("微调集为五州 2021 年有标签数据", paper)
        self.assertIn("2022 年只用于最终独立评估", paper)
        self.assertIn("不设置验证集、早停或最佳 checkpoint 选择", paper)
        self.assertNotIn("所有完整覆盖州", paper)
        self.assertNotIn("全州无标签预训练", paper)
        for phrase in ("0.9", "0.95", "0.05", "blr=1e-3", "batch_size / 256", "10 epochs", "cosine", "lr=1e-3", "5 epochs"):
            self.assertIn(phrase, paper)

    def test_paper_uses_strict_same_grid_as_supervised_main_path(self):
        paper = self.PAPER.read_text(encoding="utf-8")
        method = paper[paper.index("### 4.1 问题定义与总体框架"):paper.index("### 4.3 多模态自监督预训练")]
        self.assertIn("严格同网格", method)
        self.assertIn("A2/A3", method)
        self.assertNotIn("当前月份全部有效遥感网格作为 Key 和 Value", method)
        self.assertNotIn("G_i\\times G_i", method)
        self.assertNotIn("网格距离作为注意力偏置", method)

    def test_paper_states_single_key_same_grid_degeneracy_and_multi_candidate_variants(self):
        paper = self.PAPER.read_text(encoding="utf-8")
        method = paper[paper.index("### 4.3 多模态自监督预训练"):paper.index("### 4.7 时序编码与因果注意力")]
        self.assertIn("28个天气 Key/Value", method)
        self.assertIn("遥感 Query", method)
        self.assertIn("月度融合", method)
        self.assertIn("A2/A3", method)
        self.assertIn("多网格候选", method)
        self.assertIn("选择", method)

    def test_paper_supervised_path_has_no_obsolete_daily_query_topology(self):
        paper = self.PAPER.read_text(encoding="utf-8")
        method = paper[paper.index("### 4.1 问题定义与总体框架"):paper.index("### 4.3 多模态自监督预训练")]
        self.assertNotIn("逐日气象 Query—同网格当月遥感 Key/Value", method)
        self.assertNotIn("逐日气象网格 token 作为 Query", method)
        self.assertNotIn("weather_position", paper)
        self.assertNotIn("GRN_fuse", paper)

    def test_paper_has_a_group_evidence_rules_and_no_obsolete_joint_ablation(self):
        paper = self.PAPER.read_text(encoding="utf-8")
        conclusion = paper[paper.index("## 6 结论与展望"):paper.index("## 参考文献")]
        for phrase in ("A0", "A1", "A2", "A3", "A4", "固定 S3", "同网格", "距离检索", "不带距离", "历史无未来", "多 seed"):
            self.assertIn(phrase, conclusion)
        self.assertIn("只有当", conclusion)
        self.assertNotIn("变量选择位置与空间聚合方式联合", paper)
        self.assertNotIn("均值模型与注意力模型作为主消融", paper)
        self.assertNotRegex(paper, r"Delta_\{\\mathrm\{(?:VSN|Attn|Int)\}\}")
        self.assertNotIn("5.5 节的季中预报评估", paper)
        self.assertNotIn("自然递减", paper)
        self.assertNotIn("天然趋向末步收敛", paper)
        self.assertNotIn("已通过概率约束习得", paper)
        self.assertIn("P5 主规则", conclusion)
        self.assertIn("N2", conclusion)
        self.assertIn("N4", conclusion)


if __name__ == "__main__":
    unittest.main()
