import unittest
import json
import sys
import tempfile
from unittest.mock import patch
from pathlib import Path

import torch

from TFT_model.models import WeatherRemotePatchPretrain


ROOT = Path(__file__).resolve().parents[1]
TFT_ROOT = ROOT / "TFT_model"
if str(TFT_ROOT) not in sys.path:
    sys.path.insert(0, str(TFT_ROOT))
import pretrain_mmst


class WeatherRemotePatchPretrainTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = WeatherRemotePatchPretrain(
            hidden_size=8, num_heads=2, dropout=0.0
        ).eval()
        self.weather = torch.randn(2, 28, 8)
        self.patches = torch.randn(2, 50, 8)

    def test_returns_unpooled_daily_tokens_and_patch_weights(self):
        tokens, weights = self.model(
            self.weather, self.patches
        )

        self.assertEqual(tokens.shape, (2, 28, 8))
        self.assertEqual(weights.shape, (2, 28, 50))
        self.assertTrue(torch.isfinite(tokens).all())
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(
            torch.allclose(weights.sum(dim=-1), torch.ones(2, 28), atol=1e-6)
        )

    def test_weather_cannot_bypass_identical_remote_values(self):
        patches = torch.zeros(2, 50, 8)
        first_weather = torch.randn(2, 28, 8)
        second_weather = torch.randn(2, 28, 8)

        first, _ = self.model(first_weather, patches)
        second, _ = self.model(second_weather, patches)

        self.assertTrue(torch.allclose(first, second, atol=1e-6))

    def test_forward_loss_backward_supports_unpooled_tokens(self):
        model = pretrain_mmst.MMSTWeatherPretrain(hidden_size=8, num_heads=2)
        images = torch.randn(2, 3, 224, 224)
        weather = torch.randn(2, 28, 9)

        first, _ = model(images, weather)
        second, _ = model(images, weather)
        self.assertEqual(first.shape[:2], (2, 28))
        loss = pretrain_mmst.nt_xent(first, second)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_prepare_pretrain_batch_flattens_six_by_grid_images(self):
        images = torch.zeros(1, 6, 2, 224, 224, 3, dtype=torch.uint8)
        weather = torch.zeros(1, 6, 2, 28, 9)
        flat_images, flat_weather = pretrain_mmst.prepare_pretrain_batch(images, weather)
        self.assertEqual(flat_images.shape, (12, 3, 224, 224))
        self.assertEqual(flat_weather.shape, (12, 28, 9))

    def test_prepare_pretrain_batch_accepts_chw_county_tensor(self):
        images = torch.zeros(6, 2, 3, 224, 224)
        weather = torch.zeros(6, 2, 28, 9)
        flat_images, flat_weather = pretrain_mmst.prepare_pretrain_batch(images, weather)
        self.assertEqual(flat_images.shape, (12, 3, 224, 224))
        self.assertEqual(flat_weather.shape, (12, 28, 9))

    def test_nt_xent_pairs_same_flattened_token_index_across_views(self):
        first = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        second = first.clone()
        expected = pretrain_mmst.nt_xent(first, second, temperature=0.5)
        flattened_expected = pretrain_mmst.nt_xent(
            first.reshape(-1, 2), second.reshape(-1, 2), temperature=0.5
        )

        self.assertAlmostEqual(expected.item(), flattened_expected.item())

    def test_rejects_non_positive_head_count(self):
        with self.assertRaisesRegex(ValueError, "num_heads must be positive"):
            WeatherRemotePatchPretrain(hidden_size=8, num_heads=0)

    def test_uses_pure_cross_attention_interface(self):
        tokens, weights = self.model(self.weather, self.patches)

        self.assertEqual(tokens.shape, (2, 28, 8))
        self.assertEqual(weights.shape, (2, 28, 50))


class PretrainProtocolTest(unittest.TestCase):
    def test_scheduler_warmup_and_cosine_endpoint(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=99.0)

        warmup_lr = pretrain_mmst.adjust_learning_rate(
            optimizer, 10.0, base_lr=0.2, min_lr=0.0,
            warmup_epochs=20, total_epochs=200,
        )
        endpoint_lr = pretrain_mmst.adjust_learning_rate(
            optimizer, 200.0, base_lr=0.2, min_lr=0.0,
            warmup_epochs=20, total_epochs=200,
        )

        self.assertAlmostEqual(warmup_lr, 0.1)
        self.assertAlmostEqual(endpoint_lr, 0.0)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0)

    def test_scheduler_rejects_invalid_configuration_and_allows_zero_warmup(self):
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1.0)
        for warmup, total in ((-1, 200), (200, 200), (201, 200), (0, 0)):
            with self.assertRaisesRegex(ValueError, "warmup_epochs|total_epochs"):
                pretrain_mmst.adjust_learning_rate(
                    optimizer, 0.0, base_lr=0.2, min_lr=0.0,
                    warmup_epochs=warmup, total_epochs=total,
                )

        lr = pretrain_mmst.adjust_learning_rate(
            optimizer, 0.0, base_lr=0.2, min_lr=0.0,
            warmup_epochs=0, total_epochs=200,
        )
        self.assertAlmostEqual(lr, 0.2)

    def test_scheduler_preserves_parameter_group_lr_scale(self):
        first = torch.nn.Parameter(torch.ones(1))
        second = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([
            {"params": [first]},
            {"params": [second], "lr_scale": 0.5},
        ], lr=1.0)

        lr = pretrain_mmst.adjust_learning_rate(
            optimizer, 20.0, base_lr=0.2, min_lr=0.0,
            warmup_epochs=20, total_epochs=200,
        )

        self.assertAlmostEqual(lr, 0.2)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.2)
        self.assertAlmostEqual(optimizer.param_groups[1]["lr"], 0.1)

    def test_manifest_selection_ignores_extra_states_and_requires_complete_years(self):
        states = sorted(pretrain_mmst.FIVE_STATES)
        records = [
            {"state": state, "year": year}
            for state in states
            for year in (2017, 2018, 2019, 2020)
        ]
        records.append({"state": "texas", "year": 2019})

        selected = pretrain_mmst.select_five_state_records(records)
        self.assertEqual(len(selected), 20)
        self.assertEqual(
            pretrain_mmst.validate_pretrain_records(selected),
            {state: 4 for state in states},
        )
        incomplete = [record for record in selected if not (
            record["state"] == states[-1] and record["year"] == 2020
        )]
        with self.assertRaisesRegex(ValueError, "cover 2017-2020"):
            pretrain_mmst.validate_pretrain_records(incomplete)
        with self.assertRaisesRegex(ValueError, "outside pretraining years"):
            pretrain_mmst.validate_pretrain_records(
                selected + [{"state": states[0], "year": 2021}]
            )

    def test_manifest_validation_reports_missing_state_and_year(self):
        with self.assertRaisesRegex(ValueError, "missing State/state"):
            pretrain_mmst.validate_pretrain_records([{"year": 2017}])
        with self.assertRaisesRegex(ValueError, "missing Year/year"):
            pretrain_mmst.validate_pretrain_records([{"state": "iowa"}])

    def test_defaults_use_repository_paths_and_batch_size_learning_rate(self):
        parser = pretrain_mmst.build_parser()
        args = parser.parse_args([])
        self.assertEqual(args.manifest, str(ROOT / "MMST-ViT/data/pretrain_2017_2020.json"))
        self.assertEqual(args.output, str(ROOT / "MMST-ViT/output_current/tft_pretrain.pth"))
        self.assertEqual(pretrain_mmst.resolve_learning_rate(args), 1e-3 * args.batch_size / 256)
        self.assertNotIn("effective_batch_size", vars(args))

    def test_legacy_launcher_and_explicit_learning_rate(self):
        launcher = (ROOT / "TFT_model/run_pretrain_mmst.sh").read_text()
        self.assertNotIn("pretrain_mmst.py", launcher)
        self.assertNotIn("tft_pretrain.pth", launcher)
        self.assertIn("MMST-ViT/run_pretrain.sh", launcher)

        parser = pretrain_mmst.build_parser()
        explicit = parser.parse_args(["--lr", "0.007"])
        self.assertEqual(pretrain_mmst.resolve_learning_rate(explicit), 0.007)

    def test_optimizer_uses_mmst_adamw_protocol(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(2, 2),
            torch.nn.LayerNorm(2),
        )
        optimizer = pretrain_mmst.create_optimizer(model, learning_rate=0.01)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.95))
        self.assertEqual(
            sorted(group["weight_decay"] for group in optimizer.param_groups),
            [0.0, 0.05],
        )
        no_decay_parameters = {
            id(parameter)
            for group in optimizer.param_groups
            if group["weight_decay"] == 0.0
            for parameter in group["params"]
        }
        self.assertEqual(
            {name for name, parameter in model.named_parameters() if id(parameter) in no_decay_parameters},
            {"0.bias", "1.weight", "1.bias"},
        )

    def test_county_scheduler_progress_matches_official_iteration(self):
        self.assertEqual(pretrain_mmst.county_progress(3, 2, 5), 3.4)
        optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=1.0)
        with patch.object(pretrain_mmst, "adjust_learning_rate", wraps=pretrain_mmst.adjust_learning_rate) as adjust:
            pretrain_mmst.schedule_county_iteration(
                optimizer, epoch=3, batch_index=2, loader_length=5,
                scheduler_config={
                    "base_lr": 0.2, "min_lr": 0.0,
                    "warmup_epochs": 20, "total_epochs": 200,
                },
            )
        adjust.assert_called_once_with(
            optimizer, 3.4, base_lr=0.2, min_lr=0.0,
            warmup_epochs=20, total_epochs=200,
        )

    def test_checkpoint_contains_scheduler_configuration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pth"
            pretrain_mmst.save_checkpoint(
                path,
                model_state={},
                optimizer_state={},
                epoch=3,
                scheduler_config={
                    "base_lr": 0.001,
                    "min_lr": 0.0,
                    "warmup_epochs": 20,
                    "total_epochs": 200,
                    "batch_size": 128,
                    "actual_lr": 0.0005,
                },
            )
            checkpoint = torch.load(path, weights_only=False)

        self.assertEqual(checkpoint["scheduler"]["base_lr"], 0.001)
        self.assertEqual(checkpoint["scheduler"]["min_lr"], 0.0)
        self.assertEqual(checkpoint["scheduler"]["warmup_epochs"], 20)
        self.assertEqual(checkpoint["scheduler"]["total_epochs"], 200)
        self.assertEqual(checkpoint["scheduler"]["batch_size"], 128)
        self.assertEqual(checkpoint["scheduler"]["actual_lr"], 0.0005)

    def test_checkpoint_contains_manifest_and_protocol_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pth"
            pretrain_mmst.save_checkpoint(
                path, model_state={}, optimizer_state={}, epoch=3,
                scheduler_config={"total_epochs": 200},
                manifest_path="manifest.json", manifest_hash="abc",
                config={"blr": 1e-3},
            )
            checkpoint = torch.load(path, weights_only=False)
        self.assertEqual(checkpoint["manifest_path"], "manifest.json")
        self.assertEqual(checkpoint["manifest_hash"], "abc")
        self.assertEqual(checkpoint["config"]["blr"], 1e-3)

    def test_resume_restores_model_optimizer_and_next_epoch(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pth"
            pretrain_mmst.save_checkpoint(
                path, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                epoch=7, scheduler_config={"total_epochs": 200},
            )
            restored = torch.nn.Linear(2, 1)
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.02)
            start_epoch, payload = pretrain_mmst.load_resume_checkpoint(
                path, restored, restored_optimizer
            )
        self.assertEqual(start_epoch, 8)
        self.assertEqual(payload["epoch"], 7)
        for expected, actual in zip(model.parameters(), restored.parameters()):
            self.assertTrue(torch.equal(expected, actual))

    def test_resume_rejects_manifest_or_scheduler_mismatch(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pth"
            pretrain_mmst.save_checkpoint(
                path, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                epoch=1, scheduler_config={"total_epochs": 200, "warmup_epochs": 20},
                manifest_hash="abc", config={"epochs": 200},
            )
            with self.assertRaises(ValueError):
                pretrain_mmst.load_resume_checkpoint(
                    path, model, optimizer, expected_manifest_hash="other",
                    expected_scheduler={"total_epochs": 200, "warmup_epochs": 20},
                    expected_config={"epochs": 200},
                )

    def test_checkpoint_scheduler_configuration_is_usable_by_scheduler(self):
        parameter = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.AdamW([parameter], lr=1.0)
        scheduler = {
            "base_lr": 0.001,
            "min_lr": 0.0,
            "warmup_epochs": 20,
            "total_epochs": 200,
        }

        pretrain_mmst.adjust_learning_rate(optimizer, 0.5, **scheduler)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.001 * 0.5 / 20)


if __name__ == "__main__":
    unittest.main()
