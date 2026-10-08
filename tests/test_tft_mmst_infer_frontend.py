import json
import math
import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]


class InferFormalContractTests(unittest.TestCase):
    @staticmethod
    def _formal_payload(model):
        from TFT_model.train_mmst_frontend import (
            DEFAULT_CHECKPOINT,
            DEFAULT_TRAIN_CONFIG,
        )

        return {
            "model": model.state_dict(),
            "optimizer": {"state": {}, "param_groups": []},
            "epoch": 100,
            "config": {
                "train_config": str(DEFAULT_TRAIN_CONFIG),
                "train_year": 2021,
                "epochs": 100,
                "outer_batch_size": 1,
                "pretrained": str(DEFAULT_CHECKPOINT),
                "hidden_size": 512,
                "heads": 4,
                "dropout": 0.1,
                "lr": 1e-3,
                "weight_decay": 0.05,
                "warmup_epochs": 5,
                "device": "not-used-for-inference",
                "seed": 999,
            },
            "schedule": {
                "name": "linear-warmup-cosine",
                "base_lr": 1e-3,
                "warmup_epochs": 5,
                "total_epochs": 100,
                "min_lr": 0.0,
            },
            "soil_stats": {
                "mean": [1.0] * 7,
                "std": [2.0] * 7,
                "source": {"year": 2021, "county_count": 195},
            },
            "source_checkpoint": str(DEFAULT_CHECKPOINT.resolve()),
        }

    def test_cli_can_be_run_directly_from_repository_root(self):
        result = subprocess.run(
            [sys.executable, "TFT_model/infer_mmst_frontend.py", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--checkpoint", result.stdout)

    def test_metrics_use_exp_restored_log_yields(self):
        from TFT_model.infer_mmst_frontend import metrics_from_log_pairs

        metrics = metrics_from_log_pairs(
            [
                {"FIPS": "17001", "yield_log": math.log(10.0), "log_mean": math.log(12.0)},
                {"FIPS": "17003", "yield_log": math.log(20.0), "log_mean": math.log(18.0)},
            ]
        )

        self.assertAlmostEqual(metrics["rmse"], math.sqrt(4.0), places=7)
        self.assertAlmostEqual(metrics["mae"], 2.0, places=7)
        self.assertAlmostEqual(metrics["r2"], 0.84, places=7)
        self.assertAlmostEqual(metrics["corr"], 1.0, places=7)

    def test_checkpoint_loader_requires_final_name_and_strictly_roundtrips_model(self):
        from TFT_model.infer_mmst_frontend import load_formal_checkpoint

        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 1)

        source = TinyModel()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "checkpoint-final.pth"
            payload = self._formal_payload(source)
            torch.save(payload, path)
            restored = TinyModel()
            with patch(
                "TFT_model.infer_mmst_frontend.MMSTTFTYieldModel",
                return_value=restored,
            ) as model_type:
                model, config, soil_stats = load_formal_checkpoint(
                    path, model_factory=model_type
                )
            model_type.assert_called_once_with(
                config["pretrained"], hidden_size=512, heads=4, dropout=0.1
            )
            self.assertIs(model, restored)
            for name, value in source.state_dict().items():
                self.assertTrue(torch.equal(value, restored.state_dict()[name]))
            self.assertEqual(soil_stats["source"]["county_count"], 195)

            wrong_name = directory / "checkpoint-latest.pth"
            torch.save(payload, wrong_name)
            with self.assertRaisesRegex(ValueError, "checkpoint-final.pth"):
                load_formal_checkpoint(wrong_name, model_factory=model_type)

    def test_checkpoint_loader_rejects_each_tampered_formal_field(self):
        from TFT_model.infer_mmst_frontend import load_formal_checkpoint

        model = nn.Linear(2, 1)
        valid = self._formal_payload(model)
        mutations = []
        for field in (
            "model", "optimizer", "epoch", "config", "schedule",
            "soil_stats", "source_checkpoint",
        ):
            payload = copy.deepcopy(valid)
            payload.pop(field)
            mutations.append((f"missing {field}", payload))
        for field, value in (("optimizer", None), ("epoch", 99)):
            payload = copy.deepcopy(valid)
            payload[field] = value
            mutations.append((f"invalid {field}", payload))

        config_values = {
            "train_config": "other.json",
            "train_year": 2020,
            "epochs": 99,
            "outer_batch_size": 2,
            "pretrained": "checkpoint-final.pth",
            "hidden_size": 256,
            "heads": 2,
            "dropout": 0.1,
            "lr": 2e-3,
            "weight_decay": 0.1,
            "warmup_epochs": 4,
        }
        for field, value in config_values.items():
            payload = copy.deepcopy(valid)
            payload["config"][field] = value
            mutations.append((f"config {field}", payload))

        schedule_values = {
            "name": "cosine",
            "base_lr": 2e-3,
            "warmup_epochs": 4,
            "total_epochs": 99,
            "min_lr": 1e-6,
        }
        for field, value in schedule_values.items():
            payload = copy.deepcopy(valid)
            payload["schedule"][field] = value
            mutations.append((f"schedule {field}", payload))

        for field, value in (("year", 2020), ("county_count", 194)):
            payload = copy.deepcopy(valid)
            payload["soil_stats"]["source"][field] = value
            mutations.append((f"soil source {field}", payload))

        payload = copy.deepcopy(valid)
        payload["source_checkpoint"] = str(ROOT / "other-checkpoint.pth")
        mutations.append(("source checkpoint", payload))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint-final.pth"
            for name, payload in mutations:
                with self.subTest(field=name):
                    torch.save(payload, path)
                    with self.assertRaises(ValueError):
                        load_formal_checkpoint(path, model_factory=lambda *args, **kwargs: model)

    def test_eval_path_uses_eval_wrapper_and_never_long_weather(self):
        source = (ROOT / "TFT_model/infer_mmst_frontend.py").read_text(encoding="utf-8")
        self.assertNotIn("matplotlib", source)
        self.assertNotIn("get_long_term_val", source)
        self.assertIn("DataWrapper(train=False)", source)
        self.assertIn("corn_val.json", source)

    def test_real_official_validation_manifest_has_239_fips(self):
        records = json.loads(
            (ROOT / "MMST-ViT/data/corn_val.json").read_text(encoding="utf-8")
        )
        self.assertEqual(len(records), 239)
        self.assertEqual(len({record["FIPS"] for record in records}), 239)

    @unittest.skipUnless(
        (Path("/data/raid0/hqx/Product_model_runtime/DataSrc").is_dir()
         and Path("/data/raid0/hqx/Product_model_runtime/DataSrc/soil/county_soil.json").is_file()
         and Path("/data/raid0/hqx/TFT_train/mmst_frontend_2021/checkpoint-final.pth").is_file()),
        "需要正式 239 县数据、土壤文件和 checkpoint-final.pth",
    )
    def test_one_real_eval_county_smoke_is_finite(self):
        from TFT_model.infer_mmst_frontend import (
            DEFAULT_CHECKPOINT,
            DEFAULT_DATA_ROOT,
            DEFAULT_EVAL_CONFIG,
            DEFAULT_SOIL_FILE,
            load_formal_checkpoint,
            transform_eval_images,
        )
        from TFT_model.mmst_data import MMSTCornDataset
        from TFT_model.train_mmst_frontend import zscore_soil

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model, _, stats = load_formal_checkpoint(DEFAULT_CHECKPOINT)
        dataset = MMSTCornDataset(DEFAULT_DATA_ROOT, DEFAULT_EVAL_CONFIG, DEFAULT_SOIL_FILE)
        self.addCleanup(dataset.close)
        images, weather, target_log, soil, fips, year = dataset[0]
        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std = torch.tensor(stats["std"], dtype=torch.float32)
        with torch.inference_mode():
            final_dist, _, _ = model.to(device).eval()(
                transform_eval_images(images.unsqueeze(0)).to(device),
                weather.unsqueeze(0).to(device),
                zscore_soil(soil.unsqueeze(0), mean, std).to(device),
            )
        self.assertEqual((fips, year), ("17001", 2022))
        self.assertTrue(torch.isfinite(final_dist).all())
        self.assertTrue(math.isfinite(math.exp(float(target_log))))


if __name__ == "__main__":
    unittest.main()
