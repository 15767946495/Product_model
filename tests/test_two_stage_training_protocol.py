import importlib
import os
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]


class TwoStageTrainingProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = importlib.import_module("TFT_model.train")
        cls.infer = importlib.import_module("TFT_model.infer")

    @staticmethod
    def _complete_mmst_pvt_source(model):
        source = {}
        for target_key, value in model.state_dict().items():
            prefix = "vit_encoder.backbone."
            if not target_key.startswith(prefix):
                continue
            suffix = target_key[len(prefix):]
            if suffix.startswith("stages."):
                stage, remainder = suffix[len("stages."):].split(".", 1)
                source[f"backbone.patch_embed{int(stage) + 1}.{remainder}"] = value.clone() + 1
            elif suffix.startswith("blocks."):
                stage, block, remainder = suffix[len("blocks."):].split(".", 2)
                source[f"backbone.block{int(stage) + 1}.{block}.{remainder}"] = value.clone() + 1
            elif suffix.startswith("pos_embeds."):
                stage = suffix[len("pos_embeds."):]
                source[f"backbone.pos_embed{int(stage) + 1}"] = value.clone() + 1
        return source

    @staticmethod
    def _mmst_source_key_for_target(target_key):
        prefix = "vit_encoder.backbone."
        suffix = target_key[len(prefix):]
        if suffix.startswith("stages."):
            stage, remainder = suffix[len("stages."):].split(".", 1)
            return f"backbone.patch_embed{int(stage) + 1}.{remainder}"
        if suffix.startswith("blocks."):
            stage, block, remainder = suffix[len("blocks."):].split(".", 2)
            return f"backbone.block{int(stage) + 1}.{block}.{remainder}"
        if suffix.startswith("pos_embeds."):
            stage = suffix[len("pos_embeds."):]
            return f"backbone.pos_embed{int(stage) + 1}"
        raise AssertionError(f"not an early PVT target: {target_key}")

    def _materialize_mmst_checkpoint(self, directory):
        real_path = ROOT / "MMST-ViT/output_dir/pvt_simclr/checkpoint-final.pth"
        if real_path.is_file():
            return real_path

        sys.path.insert(0, str(ROOT / "MMST-ViT"))
        try:
            simclr = importlib.import_module("models_pvt_simclr")
            source_model = simclr.PVTSimCLR("pvt_tiny", pretrained=False)
            source = source_model.state_dict()
        finally:
            sys.path.remove(str(ROOT / "MMST-ViT"))
        path = Path(directory) / "checkpoint-final.pth"
        torch.save(
            {"model": source, "protocol": {"pretrain_years": [2017, 2020]}},
            path,
        )
        return path

    def test_training_parser_defaults_to_fixed_2021_protocol(self):
        args = self.train.build_parser().parse_args([])

        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.train_year, 2021)
        self.assertFalse(args.early_stop)
        self.assertEqual(args.checkpoint_name, "final_model.pth")
        self.assertTrue(args.pretrained.endswith("MMST-ViT/output_dir/pvt_simclr/checkpoint-final.pth"))
        self.assertNotIn("no_pretrained", vars(args))
        self.assertTrue(args.use_constructed)
        self.assertFalse(self.train.build_parser().parse_args(["--no-use-constructed"]).use_constructed)
        with self.assertRaises(SystemExit):
            self.train.build_parser().parse_args(["--no-pretrained"])

    def test_finetune_parser_defaults_match_mmst_schedule(self):
        args = self.train.build_parser().parse_args([])

        self.assertEqual(args.epochs, 100)
        self.assertAlmostEqual(args.lr, 1e-3)
        self.assertAlmostEqual(args.weight_decay, 0.05)
        self.assertEqual(args.warmup_epochs, 5)

    def test_finetune_launcher_uses_mmst_checkpoint_and_paper_schedule(self):
        content = (ROOT / "TFT_model/run_finetune.sh").read_text()

        self.assertIn("../MMST-ViT/output_dir/pvt_simclr/checkpoint-final.pth", content)
        self.assertIn("--pretrained", content)
        self.assertIn("--epochs 100", content)
        self.assertIn("--lr 1e-3", content)
        self.assertIn("--weight_decay 0.05", content)
        self.assertIn("--warmup_epochs 5", content)
        for argument in (
            "--hidden_size 512", "--num_lstm_layers 1", "--num_heads 4", "--dropout 0.1",
        ):
            self.assertIn(argument, content)
        self.assertNotIn("tft_pretrain.pth", content)

    def test_finetune_launcher_appends_fixed_protocol_after_user_arguments(self):
        result = subprocess.run(
            [
                str(ROOT / "TFT_model/run_finetune.sh"),
                "--epochs", "1", "--lr", "9", "--weight_decay", "9",
                "--warmup_epochs", "99",
            ],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHON": "echo"},
        )
        arguments = shlex.split(result.stdout)

        for option, fixed_value in (
            ("--epochs", "100"),
            ("--lr", "1e-3"),
            ("--weight_decay", "0.05"),
            ("--warmup_epochs", "5"),
        ):
            positions = [index for index, value in enumerate(arguments) if value == option]
            self.assertEqual(len(positions), 2)
            self.assertEqual(arguments[positions[-1] + 1], fixed_value)

    def test_training_launchers_are_executable(self):
        for relative_path in ("TFT_model/run_finetune.sh", "TFT_model/run_pretrain_mmst.sh"):
            with self.subTest(path=relative_path):
                self.assertTrue(os.access(ROOT / relative_path, os.X_OK))

    def test_legacy_tft_pretraining_launcher_cannot_start_training(self):
        content = (ROOT / "TFT_model/run_pretrain_mmst.sh").read_text()

        self.assertIn("legacy", content.lower())
        self.assertIn('MMST_SCRIPT="${SCRIPT_DIR}/../MMST-ViT/run_pretrain.sh"', content)
        self.assertIn("${MMST_SCRIPT}", content)
        self.assertNotIn("pretrain_mmst.py", content)
        self.assertNotIn("tft_pretrain.pth", content)

    def test_training_launchers_have_valid_bash_syntax(self):
        for relative_path in ("TFT_model/run_finetune.sh", "TFT_model/run_pretrain_mmst.sh"):
            with self.subTest(path=relative_path):
                subprocess.run(
                    ["bash", "-n", str(ROOT / relative_path)],
                    check=True,
                    capture_output=True,
                    text=True,
                )

    def test_protocol_defaults_pass_strict_argument_validation(self):
        args = self.train.build_parser().parse_args([])

        self.train.validate_protocol_args(
            args.epochs, args.train_year, args.early_stop, args.checkpoint_name,
            args.lr, args.weight_decay, args.warmup_epochs, args.pretrained,
        )

    def test_protocol_rejects_each_non_protocol_training_argument(self):
        valid = (
            100, 2021, False, "final_model.pth", 1e-3, 0.05, 5,
            str(ROOT / "MMST-ViT/output_dir/pvt_simclr/checkpoint-final.pth"),
        )
        invalid_values = [
            ("epochs", (99, *valid[1:])),
            ("train_year", (valid[0], 2022, *valid[2:])),
            ("early_stop", (valid[0], valid[1], True, *valid[3:])),
            ("checkpoint_name", (*valid[:3], "best_model.pth", *valid[4:])),
            ("lr", (*valid[:4], 1e-4, *valid[5:])),
            ("weight_decay", (*valid[:5], 5e-4, valid[6], valid[7])),
            ("warmup_epochs", (*valid[:6], 10, valid[7])),
            ("pretrained", (*valid[:7], str(ROOT / "other-checkpoint.pth"))),
        ]
        for name, args in invalid_values:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.train.validate_protocol_args(*args)

    def test_formal_main_rejects_missing_default_checkpoint_with_guidance(self):
        with patch.object(self.train, "load_jsonl") as load_jsonl:
            with patch("sys.argv", ["train.py", "--device", "cpu"]), self.assertRaisesRegex(
                FileNotFoundError, "MMST-ViT/run_pretrain.sh"
            ):
                self.train.main()
        load_jsonl.assert_not_called()

    def test_formal_main_rejects_replaced_checkpoint_before_data_loading(self):
        replacement = str(ROOT / "temporary-checkpoint.pth")
        with patch.object(self.train, "load_jsonl") as load_jsonl:
            with patch(
                "sys.argv", ["train.py", "--pretrained", replacement]
            ), self.assertRaisesRegex(ValueError, "canonical MMST PVT checkpoint"):
                self.train.main()
        load_jsonl.assert_not_called()

    def test_nonfinite_loss_does_not_step_optimizer(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        step = patch.object(optimizer, "step", wraps=optimizer.step)
        with step as step_mock:
            self.assertFalse(self.train.finetune_optimizer_step(torch.tensor(float("inf")), model, optimizer))
        self.assertEqual(step_mock.call_count, 0)
        self.assertIsNone(model.weight.grad)

    def test_nonfinite_gradient_does_not_step_optimizer(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        model.weight.register_hook(lambda gradient: torch.full_like(gradient, float("inf")))
        loss = model.weight.sum()

        with patch.object(optimizer, "step", wraps=optimizer.step) as step_mock:
            self.assertFalse(self.train.finetune_optimizer_step(loss, model, optimizer))
        self.assertEqual(step_mock.call_count, 0)
        self.assertIsNone(model.weight.grad)
        before = model.weight.detach().clone()
        self.assertTrue(torch.equal(model.weight, before))

    def test_train_loss_accumulation_counts_only_valid_samples(self):
        loss_sum, sample_count = self.train.accumulate_train_loss(
            torch.tensor(2.0), torch.tensor([True, False, True, False])
        )

        self.assertEqual(loss_sum, 4.0)
        self.assertEqual(sample_count, 2)
        self.assertEqual(loss_sum / sample_count, 2.0)

    def test_finetune_learning_rate_has_warmup_and_cosine_endpoints(self):
        model = torch.nn.Linear(2, 1)
        optimizer = self.train.create_finetune_optimizer(model, 1e-3, 0.05)

        self.assertEqual(
            self.train.adjust_finetune_learning_rate(
                optimizer, 0.0, base_lr=1e-3, warmup_epochs=5, total_epochs=100,
            ),
            0.0,
        )
        self.assertAlmostEqual(
            self.train.adjust_finetune_learning_rate(
                optimizer, 5.0, base_lr=1e-3, warmup_epochs=5, total_epochs=100,
            ),
            1e-3,
        )
        self.assertAlmostEqual(
            self.train.adjust_finetune_learning_rate(
                optimizer, 100.0, base_lr=1e-3, warmup_epochs=5, total_epochs=100,
            ),
            0.0,
        )
        cosine_values = [
            self.train.adjust_finetune_learning_rate(
                optimizer, progress, base_lr=1e-3, warmup_epochs=5, total_epochs=100,
            )
            for progress in (5.0, 20.0, 50.0, 80.0, 100.0)
        ]
        self.assertTrue(all(left >= right for left, right in zip(cosine_values, cosine_values[1:])))

    def test_finetune_optimizer_uses_mmst_adamw_groups_and_betas(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.LayerNorm(3))
        optimizer = self.train.create_finetune_optimizer(model, 1e-3, 0.05)

        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.999))
        self.assertEqual(len(optimizer.param_groups), 2)
        decay_by_parameter = {
            parameter: group["weight_decay"]
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(decay_by_parameter[model[0].weight], 0.05)
        self.assertEqual(decay_by_parameter[model[0].bias], 0.0)
        self.assertEqual(decay_by_parameter[model[1].weight], 0.0)
        self.assertEqual(decay_by_parameter[model[1].bias], 0.0)

    def test_training_updates_finetune_learning_rate_once_per_iteration(self):
        class TinyTrainingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameter = torch.nn.Parameter(torch.ones(()))
                self.vit_encoder = torch.nn.Module()
                self.vit_encoder.backbone = torch.nn.Linear(1, 1)
                for name in (
                    "weather_context_grn", "remote_context_grn", "weather_remote_attn",
                    "grid_vsn", "spatial_agg", "lstm_encoder", "attention", "mean_head",
                    "variance_head",
                ):
                    setattr(self, name, torch.nn.Linear(1, 1))

            def forward(self, **kwargs):
                prediction = self.parameter.expand(1, 2).clone()
                return prediction, None, {
                    "sample_valid": torch.ones(1, dtype=torch.bool),
                    "grad_tensors": {"pred_dist_all": prediction.view(1, 1, 2)},
                    "month_valid": torch.ones(1, 1, dtype=torch.bool),
                }

        model = TinyTrainingModel()
        batches = [
            (
                torch.zeros(1, 1, 1), torch.zeros(1, 1, 2), torch.ones(1, 1, dtype=torch.bool),
                torch.ones(1, 1, dtype=torch.long), torch.ones(1, 1, dtype=torch.long),
                torch.zeros(1, 1), torch.ones(1, 1), torch.ones(1, dtype=torch.long),
            )
        ]
        with patch.object(self.train, "adjust_finetune_learning_rate", wraps=self.train.adjust_finetune_learning_rate) as adjust:
            self.train.train_model(
                model, batches, val_loader=None, epochs=1, lr=1e-3,
                warmup_epochs=0, device="cpu", ckpt_path=tempfile.mktemp(),
            )
        self.assertEqual(adjust.call_count, len(batches))

    def test_training_loader_selects_only_five_state_2021_labeled_pairs(self):
        pairs = [
            ({"Year": 2021, "State": "illinois", "yield_per_acre": 1}, "2021-il"),
            ({"Year": 2021, "State": "iowa", "yield_per_acre": 2}, "2021-ia"),
            ({"Year": 2021, "State": "louisiana", "yield_per_acre": 3}, "2021-la"),
            ({"Year": 2021, "State": "mississippi", "yield_per_acre": 4}, "2021-ms"),
            ({"Year": 2021, "State": "new_york", "yield_per_acre": 5}, "2021-ny"),
            ({"Year": 2022, "State": "illinois", "yield_per_acre": 6}, "2022-il"),
            ({"Year": 2021, "State": "texas", "yield_per_acre": 7}, "2021-tx"),
        ]

        selected = self.train.select_2021_training_pairs(pairs)

        self.assertEqual(len(selected), 5)
        self.assertTrue(all(int(meta["Year"]) == 2021 for meta, _ in selected))

    def test_fixed_protocol_never_uses_validation_or_best_checkpoint(self):
        self.assertFalse(self.train.FIXED_PROTOCOL_USES_VALIDATION)
        self.assertEqual(self.train.final_checkpoint_name(), "final_model.pth")

    def test_inference_parser_defaults_to_2022_and_final_checkpoint(self):
        args = self.infer.build_parser().parse_args([])

        self.assertEqual(args.val_year, "2022")
        self.assertEqual(args.ckpt, "final_model.pth")

    def test_train_and_infer_share_fixed_output_directory(self):
        self.assertEqual(self.train.DEFAULT_OUTPUT_DIR, self.infer.DEFAULT_OUTPUT_DIR)
        self.assertEqual(self.train.DEFAULT_OUTPUT_DIR, "/data/raid0/hqx/TFT_train/fixed_2021")
        self.assertEqual(self.train.build_parser().parse_args([]).output_dir, self.train.DEFAULT_OUTPUT_DIR)
        self.assertEqual(self.infer.build_parser().parse_args([]).output_dir, self.infer.DEFAULT_OUTPUT_DIR)

    def test_protocol_rejects_custom_checkpoint_and_partial_states(self):
        with self.assertRaises(ValueError):
            self.train.validate_protocol_args(
                100, 2021, False, "other.pth", 1e-3, 0.05, 5,
                self.train.DEFAULT_PRETRAINED_CHECKPOINT,
            )
        with self.assertRaises(ValueError):
            self.train.validate_protocol_states("illinois,iowa")
        self.assertEqual(
            self.train.validate_protocol_states(",".join(sorted(self.train.CROPNET_FIVE_STATES))),
            self.train.CROPNET_FIVE_STATES,
        )

    def test_inference_protocol_rejects_non_2022_and_partial_states(self):
        with self.assertRaises(ValueError):
            self.infer.validate_infer_args("2021", "illinois,iowa")
        with self.assertRaises(ValueError):
            self.infer.validate_infer_args("2022,2021", ",".join(sorted(self.infer.CROPNET_FIVE_STATES)))

    def test_inference_pair_selection_accepts_lowercase_fields(self):
        pairs = [
            ({"year": 2022, "state": state, "yield_per_acre": 1}, state)
            for state in self.infer.CROPNET_FIVE_STATES
        ]
        pairs.append(({"Year": 2021, "State": "iowa", "yield_per_acre": 2}, "old"))
        selected = self.infer.select_2022_evaluation_pairs(pairs)
        self.assertEqual(len(selected), 5)
        self.assertEqual({meta["state"] for meta, _ in selected}, self.infer.CROPNET_FIVE_STATES)

    def test_inference_dataset_binds_agriculture_images(self):
        pairs = [({"Year": 2022, "State": "new york", "FIPS": "36001"}, {})]
        samples = [{"sample": 1, "FIPS": "36001", "year": 2022}]
        with patch.object(self.infer, "build_grid_samples", return_value=samples), \
             patch.object(self.infer, "AgricultureImageDataset") as ag_cls, \
             patch.object(self.infer, "GridTimeSeriesDataset") as grid_cls:
            ag = ag_cls.return_value
            self.infer.build_inference_dataset(pairs, {}, ["x"])
        expected_meta = dict(pairs[0][0], State="new_york")
        ag_cls.assert_called_once_with([expected_meta], train=False, seed=0)
        grid_cls.assert_called_once_with(samples, ag_dataset=ag)

    def test_grid_collate_returns_per_grid_remote_mask(self):
        stats = {
            "dynamic": (torch.zeros(1), torch.ones(1)),
            "soil": (torch.zeros(7), torch.ones(7)),
        }
        collate = self.train.make_grid_collate_fn(stats, ["x"])
        ag_images = torch.ones(6, 1, 3, 224, 224)
        sample = (
            torch.zeros(2, 1, 1), torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
            torch.tensor([4]), torch.tensor([1]), torch.zeros(7), torch.ones(1),
            1, "iowa", 2022, "19001", "x", ag_images,
            torch.ones(6, dtype=torch.bool), torch.tensor([[0.0, 0.0]]),
        )
        batch = collate([sample])
        ag_grid_mask = batch[-1]
        self.assertEqual(ag_grid_mask.shape, (1, 6, 2))
        self.assertTrue(ag_grid_mask[0, :, 0].all())
        self.assertFalse(ag_grid_mask[0, :, 1].any())

    def test_training_dataset_aligns_ag_after_sample_filtering(self):
        pairs = [
            ({"Year": 2021, "State": "new york", "FIPS": "36001"}, {}),
            ({"Year": 2021, "State": "iowa", "FIPS": "19001"}, {}),
        ]
        samples = [{"FIPS": "19001", "year": 2021}]
        with patch.object(self.train, "build_grid_samples", return_value=samples), \
             patch.object(self.train, "AgricultureImageDataset") as ag_cls, \
             patch.object(self.train, "GridTimeSeriesDataset") as grid_cls:
            ag = ag_cls.return_value
            self.train.build_supervised_dataset(pairs, {}, ["x"], train=True, seed=42)
        ag_cls.assert_called_once_with(
            [{"Year": 2021, "State": "iowa", "FIPS": "19001"}], train=True, seed=42
        )
        grid_cls.assert_called_once_with(samples, ag_dataset=ag)

    def test_training_pair_selection_accepts_lowercase_fields_and_requires_coverage(self):
        pairs = [
            ({"year": 2021, "state": state, "yield_per_acre": 1}, state)
            for state in self.train.CROPNET_FIVE_STATES
        ]
        selected = self.train.select_2021_training_pairs(pairs)
        self.assertEqual({meta["state"] for meta, _ in selected}, self.train.CROPNET_FIVE_STATES)
        with self.assertRaises(ValueError):
            self.train.select_2021_training_pairs(pairs[:-1])
        with self.assertRaises(ValueError):
            self.train.select_2021_training_pairs([{"Year": 2021}, "missing"])

    def test_checkpoint_helpers_normalize_dataparallel_and_wrapped_state(self):
        state = {"layer.weight": torch.ones(1)}
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/final_model.pth"
            parallel_model = torch.nn.DataParallel(torch.nn.Linear(1, 1))
            self.train.save_final_checkpoint(parallel_model, path)
            saved = self.train.load_checkpoint_state(path)
            self.assertTrue(saved)
            self.assertTrue(all(not key.startswith("module.") for key in saved))
            torch.save({"model": {"module.layer.weight": state["layer.weight"]}}, path)
            normalized = self.train.load_checkpoint_state(path)
        self.assertEqual(list(normalized), ["layer.weight"])

    def test_final_checkpoint_has_protocol_metadata(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/final_model.pth"
            self.train.save_final_checkpoint(
                model, path, epoch=100,
                protocol={"finetune_year": 2021},
                config={"weight_decay": 5e-4}, seed=42,
                feature_norm_path="feature_norm.json",
                hparams_path="model_hparams.json",
                optimizer=optimizer,
            )
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload["epoch"], 100)
            self.assertEqual(payload["protocol"]["finetune_year"], 2021)
            self.assertEqual(payload["seed"], 42)
            self.assertEqual(payload["feature_norm_path"], "feature_norm.json")
            self.assertEqual(payload["experiment_id"], self.train.DEFAULT_MODEL_IDENTITY)
            self.assertEqual(payload["config"]["experiment_id"], self.train.DEFAULT_MODEL_IDENTITY)

    def test_beta_nll_matches_fixed_numeric_formula(self):
        prediction = torch.tensor([[1.0, 0.4], [2.0, -0.2]])
        target = torch.tensor([[1.5], [1.0]])
        actual = self.train.BetaNLLLoss(beta=0.5)(prediction, target)
        mean, alpha = prediction.chunk(2, dim=-1)
        expected = (
            0.5 * torch.exp(0.5 * alpha.detach())
            * ((target - mean) ** 2 / torch.exp(alpha) + alpha)
        ).mean()
        self.assertTrue(torch.allclose(actual, expected))

    def test_pretrained_components_map_aliases_without_strict_loading(self):
        feature_name = self.train.PRETRAIN_WEATHER_FEATURE_NAMES[0]
        model = self.train.TFTEncoderForYieldPrediction(
            soil_dim=7, dynamic_feature_names=[feature_name], hidden_size=8,
            num_lstm_layers=1, dropout=0.0, num_heads=2,
        )
        target = model.state_dict()
        source = {
            "pvt.stages.0.proj.weight": target["vit_encoder.backbone.stages.0.proj.weight"].clone(),
            "feature_projection.0.weight": target[f"per_feature_linear.{feature_name}.weight"].clone(),
            "vsn.flattened_grn.fc1.weight": target["grid_vsn.flattened_grn.fc1.weight"].clone(),
            "vsn.single_var_grns.0.fc1.weight": target[f"grid_vsn.single_var_grns.{feature_name}.fc1.weight"].clone(),
            "cross_modal.q.weight": torch.ones(3),
            "unrelated.weight": torch.ones(3),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/pretrain.pth"
            torch.save({"model": source}, path)
            report = self.train.load_pretrained_components(model, path)
        self.assertIn("vit_encoder.backbone.stages.0.proj.weight", report["loaded"])
        self.assertIn(f"per_feature_linear.{feature_name}.weight", report["loaded"])
        self.assertIn(f"grid_vsn.single_var_grns.{feature_name}.fc1.weight", report["loaded"])
        self.assertIn("cross_modal.q.weight", report["skipped"])
        self.assertIn("unrelated.weight", report["skipped"])

    def test_mmst_pvt_loader_maps_early_backbone_and_preserves_non_pvt_modules(self):
        feature_name = self.train.PRETRAIN_WEATHER_FEATURE_NAMES[0]
        model = self.train.TFTEncoderForYieldPrediction(
            soil_dim=7, dynamic_feature_names=[feature_name], hidden_size=8,
            num_lstm_layers=1, dropout=0.0, num_heads=2,
        )
        before = {
            key: value.clone()
            for key, value in model.state_dict().items()
            if not key.startswith("vit_encoder.backbone.")
        }
        target = model.state_dict()
        source = self._complete_mmst_pvt_source(model)
        source["backbone.pos_embed4"] = torch.randn(1, 2, 512)
        source["backbone.norm.weight"] = torch.randn(512)
        source["backbone.head.weight"] = torch.randn(1000, 512)
        source["mm_transformer.blocks.0.norm.weight"] = torch.randn(4)
        source["head.weight"] = torch.randn(2, 4)

        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/checkpoint-final.pth"
            torch.save({"model": source, "protocol": {"pretrain_years": [2017, 2020]}}, path)
            report = self.train.load_mmst_pvt_backbone(model, path)

        self.assertGreater(len(report["loaded"]), 0)
        self.assertIn("vit_encoder.backbone.pos_embeds.0", report["loaded"])
        self.assertIn("backbone.pos_embed4", report["mismatched"])
        self.assertIn("mm_transformer.blocks.0.norm.weight", report["skipped"])
        self.assertEqual(report["source_protocol"], {"pretrain_years": [2017, 2020]})
        self.assertEqual(report["source_checkpoint"], path)
        for key, value in before.items():
            self.assertTrue(torch.equal(model.state_dict()[key], value), key)

    def test_real_or_synthetic_mmst_checkpoint_maps_all_first_three_stages_exactly(self):
        model = self.train.TFTEncoderForYieldPrediction(
            soil_dim=7, dynamic_feature_names=["x"], hidden_size=8,
            num_lstm_layers=1, dropout=0.0, num_heads=2,
        )
        before_non_pvt = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
            if not key.startswith("vit_encoder.backbone.")
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = self._materialize_mmst_checkpoint(directory)
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            source = payload["model"]
            report = self.train.load_mmst_pvt_backbone(model, str(checkpoint_path))

        expected_targets = {
            key
            for key in model.state_dict()
            if any(
                key.startswith(f"vit_encoder.backbone.{prefix}{stage}.")
                for prefix in ("stages.", "blocks.")
                for stage in range(3)
            )
            or key in {
                "vit_encoder.backbone.pos_embeds.0",
                "vit_encoder.backbone.pos_embeds.1",
                "vit_encoder.backbone.pos_embeds.2",
            }
        }
        self.assertEqual(set(report["loaded"]), expected_targets)
        self.assertGreater(len(report["loaded"]), 0)
        self.assertEqual(report["source_protocol"], payload.get("protocol"))
        for target_key in expected_targets:
            source_key = self._mmst_source_key_for_target(target_key)
            self.assertIn(source_key, source)
            self.assertTrue(torch.equal(model.state_dict()[target_key], source[source_key]), target_key)
        for key, value in before_non_pvt.items():
            self.assertTrue(torch.equal(model.state_dict()[key], value), key)

    def test_tft_supervised_step_and_final_checkpoint_embed_transfer_metadata(self):
        feature_name = "x"
        model = self.train.TFTEncoderForYieldPrediction(
            soil_dim=7, dynamic_feature_names=[feature_name], hidden_size=8,
            num_lstm_layers=1, dropout=0.0, num_heads=2,
        )
        model._forward_remote_sensing = lambda *args: torch.zeros(1, 1, 6, 8)
        batch_size, grids, timesteps = 1, 1, 168
        month_ids = torch.tensor([[month for month in range(4, 10) for _ in range(28)]])
        day_ids = torch.tensor([[day for _ in range(6) for day in range(1, 29)]])
        with torch.no_grad():
            initial = {
                key: value.detach().clone()
                for key, value in model.named_parameters()
                if value.requires_grad
            }
        prediction, _, aux = model(
            grid_feats=torch.zeros(batch_size, grids, timesteps, 1),
            grid_coords=torch.zeros(batch_size, grids, 2),
            grid_mask=torch.ones(batch_size, grids, dtype=torch.bool),
            soil_feats=torch.zeros(batch_size, 7),
            seq_lens=torch.tensor([timesteps]),
            month_ids=month_ids,
            day_ids=day_ids,
            ag_images=torch.zeros(batch_size, 6, grids, 3, 224, 224),
            ag_mask=torch.ones(batch_size, 6, dtype=torch.bool),
            ag_grid_mask=torch.ones(batch_size, 6, grids, dtype=torch.bool),
        )
        self.assertEqual(prediction.shape, (batch_size, 2))
        self.assertEqual(aux["pred_dist_all"].shape, (batch_size, 6, 2))
        loss = self.train.BetaNLLLoss()(prediction, torch.zeros(batch_size, 1))
        optimizer = self.train.create_finetune_optimizer(model, 1e-3, 0.05)
        learning_rate = self.train.adjust_finetune_learning_rate(
            optimizer, 0.5, base_lr=1e-3, warmup_epochs=5, total_epochs=100,
        )
        self.assertAlmostEqual(learning_rate, 1e-4)
        self.assertTrue(self.train.finetune_optimizer_step(loss, model, optimizer))
        self.assertTrue(any(not torch.equal(parameter, initial[name]) for name, parameter in model.named_parameters()))

        transfer_report = {
            "loaded": ["vit_encoder.backbone.stages.0.proj.weight"],
            "source_checkpoint": "synthetic/checkpoint-final.pth",
        }
        metadata = self.train.build_checkpoint_metadata(
            protocol={"finetune_year": 2021, "checkpoint_rule": "final_epoch"},
            config={"epochs": 100}, seed=42, feature_norm_path=None,
            hparams_path=None, model_hparams={"transfer_report": transfer_report},
            feature_norm={}, manifest_hash=None,
            schedule={"name": "warmup_cosine", "total_epochs": 100},
            source_checkpoint=transfer_report["source_checkpoint"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "final_model.pth"
            self.train.save_final_checkpoint(model, str(path), epoch=100, **metadata)
            saved = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(saved["source_checkpoint"], transfer_report["source_checkpoint"])
        self.assertEqual(saved["model_hparams"]["transfer_report"], transfer_report)
        self.assertEqual(saved["schedule"]["name"], "warmup_cosine")
        self.assertEqual(saved["experiment_id"], self.train.DEFAULT_MODEL_IDENTITY)
        self.assertEqual(saved["config"]["experiment_id"], self.train.DEFAULT_MODEL_IDENTITY)

    def test_mmst_pvt_loader_requires_complete_early_stage_parameter_sets(self):
        model = self.train.TFTEncoderForYieldPrediction(
            soil_dim=7, dynamic_feature_names=["x"], hidden_size=8,
            num_lstm_layers=1, dropout=0.0, num_heads=2,
        )
        target = model.state_dict()
        with tempfile.TemporaryDirectory() as directory:
            for stage in range(3):
                source = {
                    f"backbone.patch_embed{stage + 1}.proj.weight": target[
                        f"vit_encoder.backbone.stages.{stage}.proj.weight"
                    ].clone()
                }
                path = f"{directory}/stage-{stage}.pth"
                torch.save({"model": source}, path)
                with self.assertRaisesRegex(ValueError, "missing keys"):
                    self.train.load_mmst_pvt_backbone(model, path)

    def test_mmst_pvt_loader_reports_missing_early_block_bias(self):
        model = self.train.TFTEncoderForYieldPrediction(
            soil_dim=7, dynamic_feature_names=["x"], hidden_size=8,
            num_lstm_layers=1, dropout=0.0, num_heads=2,
        )
        source = self._complete_mmst_pvt_source(model)
        missing_source_key = "backbone.block2.0.attn.q.bias"
        del source[missing_source_key]
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/missing-block-bias.pth"
            torch.save({"model": source}, path)
            with self.assertRaisesRegex(ValueError, "blocks.1.0.attn.q.bias"):
                self.train.load_mmst_pvt_backbone(model, path)

    def test_mmst_pvt_loader_rejects_zero_loaded_or_missing_early_stage(self):
        model = self.train.TFTEncoderForYieldPrediction(
            soil_dim=7, dynamic_feature_names=["x"], hidden_size=8,
            num_lstm_layers=1, dropout=0.0, num_heads=2,
        )
        with tempfile.TemporaryDirectory() as directory:
            empty_path = f"{directory}/empty.pth"
            torch.save({"model": {"mm_transformer.weight": torch.ones(1)}}, empty_path)
            with self.assertRaises(ValueError):
                self.train.load_mmst_pvt_backbone(model, empty_path)

            partial_path = f"{directory}/partial.pth"
            target = model.state_dict()["vit_encoder.backbone.stages.0.proj.weight"]
            torch.save({"model": {"backbone.patch_embed1.proj.weight": target}}, partial_path)
            with self.assertRaises(ValueError):
                self.train.load_mmst_pvt_backbone(model, partial_path)

    def test_pretrained_weather_mapping_uses_feature_names_not_indices(self):
        names = list(self.train.DEFAULT_DYNAMIC_FEATURE_NAMES)
        model = self.train.TFTEncoderForYieldPrediction(
            soil_dim=7, dynamic_feature_names=names, hidden_size=8,
            num_lstm_layers=1, dropout=0.0, num_heads=2,
        )
        target = model.state_dict()
        source = {
            "feature_projection.7.weight": target[
                "per_feature_linear.Downward Shortwave Radiation Flux (W m**-2).weight"
            ].clone(),
            "feature_projection.8.weight": target[
                "per_feature_linear.Vapor Pressure Deficit (kPa).weight"
            ].clone(),
            "vsn.flattened_grn.fc1.weight": torch.zeros_like(target["grid_vsn.flattened_grn.fc1.weight"]),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/pretrain.pth"
            torch.save({"model": source}, path)
            report = self.train.load_pretrained_components(model, path)
        self.assertIn("per_feature_linear.Downward Shortwave Radiation Flux (W m**-2).weight", report["loaded"])
        self.assertIn("per_feature_linear.Vapor Pressure Deficit (kPa).weight", report["loaded"])
        self.assertIn("vsn.flattened_grn.fc1.weight", report["skipped"])

    def test_cutoff_predictions_use_requested_timesteps(self):
        values = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
        monthly_months = torch.tensor([[4, 5, 6, 7, 8, 9]])
        month_valid = torch.tensor([[True, True, True, False, False, False]])
        result, valid = self.infer.predictions_at_cutoffs(
            values, monthly_months, month_valid, [(4, 28), (5, 28), (6, 28)]
        )
        self.assertEqual(result.tolist(), [[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
        self.assertEqual(valid.tolist(), [[True, True, True]])

    def test_cutoff_before_first_observation_is_invalid(self):
        values = torch.tensor([[[1.0, 0.0], [2.0, 0.0]]])
        _, valid = self.infer.predictions_at_cutoffs(
            values, torch.tensor([[4, 5, 6, 7, 8, 9]]),
            torch.tensor([[False, True, False, False, False, False]]),
            [(4, 28), (5, 28)],
        )
        self.assertEqual(valid.tolist(), [[False, True]])

    def test_non_contiguous_month_slots_are_preserved_for_cutoffs(self):
        values = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0]]])
        result, valid = self.infer.predictions_at_cutoffs(
            values,
            torch.tensor([[4, 5, 6, 7, 8, 9]]),
            torch.tensor([[True, False, True, True, True, True]]),
            [(6, 28)],
        )
        self.assertEqual(result.tolist(), [[[3.0, 0.0]]])
        self.assertEqual(valid.tolist(), [[True]])

    def test_cutoff_parser_rejects_non_month_end(self):
        with self.assertRaisesRegex(ValueError, "month-end"):
            self.infer.parse_cutoffs("04-28,05-01")

    def test_non_contiguous_missing_months_are_not_packed_as_a_prefix(self):
        model = self.train.TFTEncoderForYieldPrediction(
            soil_dim=7, dynamic_feature_names=["a"], hidden_size=8,
            num_lstm_layers=1, dropout=0.0, num_heads=2,
        ).eval()
        model._forward_remote_sensing = lambda *args: torch.zeros(1, 1, 6, 8)
        months = torch.tensor([[4] * 28 + [6] * 28 + [7] * 28 + [8] * 28 + [9] * 28 + [0] * 28])
        days = torch.tensor([[d for m in (4, 6, 7, 8, 9, 0) for d in range(1, 29)]])
        with torch.no_grad():
            _, attention, aux = model(
                grid_feats=torch.zeros(1, 1, 168, 1),
                grid_coords=torch.zeros(1, 1, 2),
                grid_mask=torch.ones(1, 1, dtype=torch.bool),
                soil_feats=torch.zeros(1, 7),
                seq_lens=torch.tensor([140]), month_ids=months, day_ids=days,
                ag_images=torch.zeros(1, 6, 1, 3, 1, 1),
                ag_mask=torch.tensor([[True, False, True, True, True, True]]),
            )
        self.assertEqual(aux["month_valid"].tolist(), [[True, False, True, True, True, True]])
        self.assertEqual(aux["pred_dist_all"].shape, (1, 6, 2))
        self.assertTrue(torch.equal(aux["temporal_feat"][:, 1], torch.zeros(1, 8)))
        self.assertTrue(torch.equal(attention[:, 1], torch.zeros(1, 6)))

    def test_all_remote_grid_mask_excludes_months_from_spatial_attention(self):
        model = self.train.TFTEncoderForYieldPrediction(
            soil_dim=7, dynamic_feature_names=["a"], hidden_size=8,
            num_lstm_layers=1, dropout=0.0, num_heads=2,
        ).eval()
        model._forward_remote_sensing = lambda *args: torch.ones(1, 2, 6, 8)
        with torch.no_grad():
            _, _, aux = model(
                grid_feats=torch.zeros(1, 2, 168, 1),
                grid_coords=torch.zeros(1, 2, 2),
                grid_mask=torch.ones(1, 2, dtype=torch.bool),
                soil_feats=torch.zeros(1, 7), seq_lens=torch.tensor([168]),
                month_ids=torch.tensor([[m for m in range(4, 10) for _ in range(28)]]),
                day_ids=torch.tensor([[d for _ in range(6) for d in range(1, 29)]]),
                ag_images=torch.zeros(1, 6, 2, 3, 1, 1),
                ag_mask=torch.ones(1, 6, dtype=torch.bool),
                ag_grid_mask=torch.zeros(1, 6, 2, dtype=torch.bool),
            )
        self.assertTrue(torch.equal(aux["weather_fused"], torch.zeros_like(aux["weather_fused"])))
        self.assertTrue(torch.equal(aux["spatial_weights"], torch.zeros_like(aux["spatial_weights"])))
        self.assertEqual(aux["month_valid"].tolist(), [[False, False, False, False, False, False]])
        self.assertTrue(torch.equal(aux["temporal_feat"], torch.zeros_like(aux["temporal_feat"])))
        self.assertTrue(torch.equal(aux["pred_dist_all"], torch.zeros_like(aux["pred_dist_all"])))
        self.assertTrue(torch.equal(aux["pred_all"], torch.zeros_like(aux["pred_all"])))

    def test_model_exposes_sample_valid_and_last_kl_mask_for_all_invalid_sample(self):
        model = self.train.TFTEncoderForYieldPrediction(
            soil_dim=7, dynamic_feature_names=["a"], hidden_size=8,
            num_lstm_layers=1, dropout=0.0, num_heads=2,
        ).eval()
        model._forward_remote_sensing = lambda *args: torch.ones(1, 1, 6, 8)
        with torch.no_grad():
            final, attention, aux = model(
                grid_feats=torch.zeros(1, 1, 168, 1),
                grid_coords=torch.zeros(1, 1, 2),
                grid_mask=torch.ones(1, 1, dtype=torch.bool),
                soil_feats=torch.zeros(1, 7), seq_lens=torch.tensor([168]),
                month_ids=torch.tensor([[m for m in range(4, 10) for _ in range(28)]]),
                day_ids=torch.tensor([[d for _ in range(6) for d in range(1, 29)]]),
                ag_images=torch.zeros(1, 6, 1, 3, 1, 1),
                ag_mask=torch.ones(1, 6, dtype=torch.bool),
                ag_grid_mask=torch.zeros(1, 6, 1, dtype=torch.bool),
            )
        self.assertEqual(aux["sample_valid"].tolist(), [False])
        self.assertTrue(torch.equal(final, torch.zeros_like(final)))
        self.assertTrue(torch.equal(aux["pred_dist_all"], torch.zeros_like(aux["pred_dist_all"])))
        self.assertTrue(torch.equal(attention, torch.zeros_like(attention)))

    def test_loss_functions_can_exclude_all_invalid_samples(self):
        prediction = torch.tensor([[1.0, 0.0], [100.0, 0.0]])
        target = torch.tensor([[1.0], [0.0]])
        sample_valid = torch.tensor([True, False])
        beta = self.train.BetaNLLLoss()
        expected = beta(prediction[:1], target[:1])
        self.assertTrue(torch.allclose(beta(prediction, target, sample_valid), expected))
        crucial = self.train.CRUCIALLoss()
        self.assertTrue(torch.isfinite(crucial(prediction, target, sample_valid)))
        self.assertEqual(crucial._crucial_from_l_raw(torch.tensor([1.0]), torch.tensor([False])).item(), 0.0)

    def test_inference_sample_filter_excludes_all_invalid_rows(self):
        sample_valid = torch.tensor([True, False, True])
        self.assertEqual(
            self.infer.sample_valid_indices(sample_valid).tolist(), [0, 2]
        )
        self.assertEqual(
            int((~sample_valid).sum().item()), 1
        )

    def test_monthly_cutoffs_select_six_month_outputs_from_daily_protocol_dates(self):
        values = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0], [6.0, 0.0]]])
        result, valid = self.infer.predictions_at_cutoffs(
            values, torch.tensor([[4, 5, 6, 7, 8, 9]]),
            torch.ones(1, 6, dtype=torch.bool),
            [(4, 28), (5, 28), (6, 28)],
        )
        self.assertEqual(result.tolist(), [[[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]]])
        self.assertEqual(valid.tolist(), [[True, True, True]])

    def test_supervised_model_exposes_all_timestep_distributions(self):
        model = self.train.TFTEncoderForYieldPrediction(
            soil_dim=7, dynamic_feature_names=["a"], hidden_size=8,
            num_lstm_layers=1, dropout=0.0, num_heads=2,
        ).eval()
        model._forward_remote_sensing = lambda *args: torch.zeros(1, 1, 6, 8)
        with torch.no_grad():
            final, _, aux = model(
                 grid_feats=torch.zeros(1, 1, 168, 1),
                grid_coords=torch.zeros(1, 1, 2),
                grid_mask=torch.ones(1, 1, dtype=torch.bool),
                soil_feats=torch.zeros(1, 7),
                 seq_lens=torch.tensor([168]),
                 month_ids=torch.tensor([[4] * 28 + [5] * 28 + [6] * 28 + [7] * 28 + [8] * 28 + [9] * 28]),
                 day_ids=torch.tensor([[d for _ in range(6) for d in range(1, 29)]]),
                ag_images=torch.zeros(1, 6, 1, 3, 1, 1),
                ag_mask=torch.ones(1, 6, dtype=torch.bool),
            )
        self.assertEqual(aux["pred_dist_all"].shape, (1, 6, 2))
        self.assertTrue(torch.allclose(final, aux["pred_dist_all"][:, -1]))

    def test_supervised_mmst_path_runs_all_temporal_blocks_for_six_months(self):
        model = self.train.TFTEncoderForYieldPrediction(
            soil_dim=7, dynamic_feature_names=["a"], hidden_size=8,
            num_lstm_layers=1, dropout=0.0, num_heads=2,
        ).eval()
        model._forward_remote_sensing = lambda *args: torch.zeros(1, 1, 6, 8)
        with torch.no_grad():
            _, attn_weights, aux = model(
                grid_feats=torch.zeros(1, 1, 168, 1),
                grid_coords=torch.zeros(1, 1, 2),
                grid_mask=torch.ones(1, 1, dtype=torch.bool),
                soil_feats=torch.zeros(1, 7),
                seq_lens=torch.tensor([168]),
                month_ids=torch.tensor([[m for m in range(4, 10) for _ in range(28)]]),
                day_ids=torch.tensor([[d for _ in range(6) for d in range(1, 29)]]),
                ag_images=torch.zeros(1, 6, 1, 3, 1, 1),
                ag_mask=torch.ones(1, 6, dtype=torch.bool),
            )
        self.assertEqual(aux["weather_fused"].shape, (1, 1, 6, 8))
        self.assertEqual(aux["temporal_feat"].shape, (1, 6, 8))
        self.assertEqual(attn_weights.shape, (1, 6, 6))
        self.assertEqual(aux["pred_dist_all"].shape, (1, 6, 2))

    def test_checkpoint_embeds_hparams_norm_and_audit_fields(self):
        model = torch.nn.Linear(1, 1)
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/final_model.pth"
            self.train.save_final_checkpoint(
                model, path, epoch=100,
                manifest_hash="hash", model_hparams={
                    "hidden_size": 4, "use_constructed": True,
                    "dynamic_feature_dim": 15,
                },
                feature_norm={"dynamic": {"mean": [0], "std": [1]}},
            )
            payload = torch.load(path, weights_only=False)
        for key in ("experiment_id", "git_commit", "manifest_hash", "total_parameters", "trainable_parameters", "model_hparams", "feature_norm"):
            self.assertIn(key, payload)
        self.assertEqual(payload["experiment_id"], self.train.DEFAULT_MODEL_IDENTITY)
        self.assertEqual(payload["config"]["experiment_id"], self.train.DEFAULT_MODEL_IDENTITY)

    def test_fixed_training_checkpoint_metadata_matches_save_signature(self):
        model = torch.nn.Linear(1, 1)
        metadata = self.train.build_checkpoint_metadata(
            protocol={"finetune_year": 2021},
            config={"epochs": 1},
            seed=42,
            feature_norm_path="feature_norm.json",
            hparams_path="model_hparams.json",
            model_hparams={"hidden_size": 4},
            feature_norm={"dynamic": {"mean": [0], "std": [1]}},
            manifest_hash="hash",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/final_model.pth"
            self.train.save_final_checkpoint(model, path, epoch=50, **metadata)
            payload = torch.load(path, weights_only=False)
        self.assertEqual(payload["experiment_id"], self.train.DEFAULT_MODEL_IDENTITY)
        self.assertEqual(payload["config"]["experiment_id"], self.train.DEFAULT_MODEL_IDENTITY)

    def test_final_checkpoint_records_finetune_schedule_and_source_checkpoint(self):
        model = torch.nn.Linear(1, 1)
        metadata = self.train.build_checkpoint_metadata(
            protocol={"finetune_year": 2021},
            config={},
            seed=42,
            feature_norm_path=None,
            hparams_path=None,
            model_hparams={},
            feature_norm={},
            manifest_hash=None,
            schedule={"name": "warmup_cosine", "total_epochs": 100},
            source_checkpoint="checkpoint-final.pth",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/final_model.pth"
            self.train.save_final_checkpoint(model, path, **metadata)
            payload = torch.load(path, weights_only=False)
        self.assertEqual(payload["schedule"]["name"], "warmup_cosine")
        self.assertEqual(payload["source_checkpoint"], "checkpoint-final.pth")

    def test_inference_checkpoint_prefers_embedded_hparams_and_norm(self):
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/final_model.pth"
            torch.save({
                "model": {"module.layer.weight": torch.ones(1)},
                "model_hparams": {
                    "hidden_size": 4, "use_constructed": True,
                    "dynamic_feature_dim": 15,
                },
                "feature_norm": {"dynamic": {"mean": [0], "std": [1]}},
            }, path)
            state, hparams, norm = self.infer.load_inference_checkpoint(path)
        self.assertEqual(list(state), ["layer.weight"])
        self.assertEqual(hparams["hidden_size"], 4)
        self.assertTrue(hparams["use_constructed"])
        self.assertEqual(hparams["dynamic_feature_dim"], 15)
        self.assertIn("dynamic", norm)


if __name__ == "__main__":
    unittest.main()
