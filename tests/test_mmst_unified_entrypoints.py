"""MMST-ViT 训练入口：统一参数、年份协议、final checkpoint 策略、输出契约和旧依赖静态扫描。"""

import json
import tempfile
import unittest
from pathlib import Path

import torch

from tests.test_mmst_unified_adapter import COUNTIES, MMST_DIR, load_mmst_adapter, load_mmst_module, make_cropnet_fixture
from train_dataset.dataset import CropNetDataset

LEGACY = ("hrrr_loader", "sentinel_loader", "usda_loader", "sentinel_wrapper",
          "corn_train.json", "corn_val.json", "Product_model_runtime")
TINY = ["--embed_dim", "32", "--num_heads", "4", "--batch_size", "16",
        "--device", "cpu", "--num_workers", "0", "--no_pvt_pretrained", "--epochs", "1", "--warmup_epochs", "0"]
RUN_FILES = ("config.json", "checkpoint-latest.pth", "checkpoint-final.pth", "train_metrics.jsonl",
             "run_metadata.json")


def load_entry(name):
    load_mmst_adapter()  # 把 MMST-ViT 目录加入 sys.path
    return load_mmst_module(name)


class MMSTEntrypointTests(unittest.TestCase):
    # ---------------------------------------------------------------- 任务 4
    def test_pretrain_parser_uses_only_unified_dataset_arguments(self):
        module = load_entry("main_pretrain_mmst_vit.py")
        parser = module.build_parser()
        args = parser.parse_args(["--dataset-jsonl", "dataset.jsonl", "--data-root", "/tmp/data"])
        self.assertEqual(args.dataset_jsonl, "dataset.jsonl")
        self.assertEqual(args.data_root, "/tmp/data")
        self.assertEqual(module.PRETRAIN_YEARS, (2017, 2018, 2019, 2020))
        for legacy in (["--data_file", "data/corn_pretrain_2017_2021.json"], ["--root_dir", "/tmp"]):
            with self.assertRaises(SystemExit):
                parser.parse_args(legacy)

    # ---------------------------------------------------------------- 任务 5
    def test_finetune_has_no_validation_manifest_or_best_metrics(self):
        module = load_entry("main_finetune_mmst_vit.py")
        parser = module.build_parser()
        args = parser.parse_args(["--dataset-jsonl", "dataset.jsonl", "--data-root", "/tmp/data"])
        self.assertEqual(module.FINETUNE_YEARS, (2021,))
        self.assertEqual(module.TEST_YEARS, (2022,))
        self.assertFalse(hasattr(args, "data_file_val"))
        self.assertFalse(hasattr(args, "data_file_train"))
        self.assertNotIn("best_metrics", module.__dict__)
        for legacy in (["--data_file_val", "x.json"], ["--root_dir", "/tmp"], ["--eval_year", "2021"]):
            with self.assertRaises(SystemExit):
                parser.parse_args(legacy)

    def test_final_evaluation_selects_only_2022(self):
        module = load_entry("main_finetune_mmst_vit.py")
        self.assertEqual(module.capture_evaluation_years("checkpoint-final.pth"), (2022,))
        self.assertEqual(module.capture_evaluation_years("/runs/x/checkpoint-final.pth"), (2022,))

    def test_final_evaluation_rejects_non_final_checkpoint(self):
        module = load_entry("main_finetune_mmst_vit.py")
        for name in ("checkpoint-latest.pth", "checkpoint-best.pth", "checkpoint-10.pth"):
            with self.assertRaisesRegex(ValueError, "checkpoint-final.pth"):
                module.capture_evaluation_years(name)

    # ---------------------------------------------------------------- 任务 6
    def test_mmst_run_scripts_have_no_legacy_paths_or_manifests(self):
        for name in ("run_pretrain.sh", "run_finetune.sh"):
            text = (MMST_DIR / name).read_text()
            for forbidden in ("Product_model_runtime", "data/corn_train.json", "data/corn_val.json", "--root_dir",
                              "--data_file"):
                self.assertNotIn(forbidden, text, f"{name}: {forbidden}")
            self.assertIn("--dataset-jsonl", text)
            self.assertIn("--data-root /data/raid0/hqx/DataSrc", text)
        self.assertIn("checkpoint-final.pth", (MMST_DIR / "run_finetune.sh").read_text())

    # ---------------------------------------------------------------- 任务 7
    def test_mmst_has_no_deleted_loader_or_manifest_references(self):
        files = [p for p in list(MMST_DIR.rglob("*.py")) + list(MMST_DIR.rglob("*.sh")) if "__pycache__" not in p.parts]
        for path in files:
            text = path.read_text(errors="ignore")
            for forbidden in LEGACY:
                self.assertNotIn(forbidden, text, f"{path.relative_to(MMST_DIR)}: {forbidden}")
        for name in ("hrrr_loader.py", "sentinel_loader.py", "usda_loader.py", "sentinel_wrapper.py"):
            self.assertFalse((MMST_DIR / "dataset" / name).exists(), name)


class MMSTEndToEndTests(unittest.TestCase):
    """合成 JSONL + HDF5 上 CPU 跑通预训练 -> 微调 -> final 测试，验证年份和输出文件。"""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.path = make_cropnet_fixture(cls.root, image_years=range(2017, 2023))
        cls.out = cls.root / "runs"
        common = ["--dataset-jsonl", str(cls.path), "--data-root", str(cls.root), "--output_root", str(cls.out)]
        cls.pretrain = load_entry("main_pretrain_mmst_vit.py")
        cls.pretrain_dir = cls.pretrain.main(cls.pretrain.build_parser().parse_args(common + TINY + ["--run_id", "p"]))
        cls.finetune = load_entry("main_finetune_mmst_vit.py")
        cls.finetune_dir = cls.finetune.main(cls.finetune.build_parser().parse_args(
            common + TINY + ["--depth", "1", "--heads", "2", "--run_id", "f", "--pvt_simclr", str(cls.pretrain_dir / "checkpoint-final.pth")]))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_pretrain_outputs_and_years(self):
        self.assertEqual(self.pretrain_dir, self.out / "pvt_simclr" / "p")
        for name in RUN_FILES:
            self.assertTrue((self.pretrain_dir / name).is_file(), name)
        meta = json.loads((self.pretrain_dir / "run_metadata.json").read_text())
        self.assertEqual(meta["train_years"], [2017, 2018, 2019, 2020])
        self.assertEqual(meta["train_samples"], 8)
        self.assertNotIn("test_years", meta)

    def test_finetune_outputs_and_years(self):
        self.assertEqual(self.finetune_dir, self.out / "mmst_vit" / "f")
        for name in RUN_FILES + ("test_metrics.json", "county_predictions.jsonl"):
            self.assertTrue((self.finetune_dir / name).is_file(), name)
        meta = json.loads((self.finetune_dir / "run_metadata.json").read_text())
        self.assertEqual(meta["train_years"], [2021])
        self.assertEqual(meta["test_years"], [2022])
        self.assertTrue(meta["test_checkpoint"].endswith("checkpoint-final.pth"))
        config = json.loads((self.finetune_dir / "config.json").read_text())
        self.assertNotIn("data_file_val", config)

    def test_final_metrics_and_county_predictions_cover_2022_in_dataset_order(self):
        metrics = json.loads((self.finetune_dir / "test_metrics.json").read_text())
        for key in ("rmse", "mae", "r2", "corr", "corr2", "count"):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["count"], 2)
        rows = [json.loads(line) for line in (self.finetune_dir / "county_predictions.jsonl").read_text().splitlines()]
        expected = CropNetDataset(self.path, (2022,), data_root=self.root).sample_ids
        self.assertEqual([r["sample_id"] for r in rows], expected)
        self.assertEqual({r["year"] for r in rows}, {2022})
        for row in rows:
            self.assertEqual(set(row), {"sample_id", "fips", "year", "y_true", "y_pred"})

    def test_final_checkpoint_has_model_and_no_validation_selection(self):
        checkpoint = torch.load(self.finetune_dir / "checkpoint-final.pth", map_location="cpu", weights_only=False)
        self.assertIn("model", checkpoint)
        self.assertEqual(checkpoint["train_years"], [2021])
        self.assertNotIn("best_metrics", checkpoint)

    def test_train_metrics_have_one_row_per_epoch(self):
        rows = (self.finetune_dir / "train_metrics.jsonl").read_text().splitlines()
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(rows[0])["epoch"], 0)


if __name__ == "__main__":
    unittest.main()
