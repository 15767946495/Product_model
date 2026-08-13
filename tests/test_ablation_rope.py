import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "TFT_model" / "ablation_rope.py"
SPEC = importlib.util.spec_from_file_location("ablation_rope", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
ABLATION_ROPE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ABLATION_ROPE)


class ParseVisibleDevicesTests(unittest.TestCase):
    def test_parses_comma_separated_devices(self):
        self.assertEqual(
            ABLATION_ROPE.parse_visible_devices("2, 3,4,5,6"),
            ["2", "3", "4", "5", "6"],
        )

    def test_rejects_empty_device_entries(self):
        with self.assertRaisesRegex(ValueError, "GPU"):
            ABLATION_ROPE.parse_visible_devices("2,,4")


class ParallelValidationTests(unittest.TestCase):
    def test_parallel_requires_constructed_mode(self):
        with self.assertRaisesRegex(ValueError, "constructed"):
            ABLATION_ROPE.validate_parallel_args(
                constructed=False,
                parallel=True,
                visible_devices=["2", "3", "4", "5", "6"],
                combo_count=5,
            )

    def test_parallel_requires_one_gpu_per_combo(self):
        with self.assertRaisesRegex(ValueError, "2"):
            ABLATION_ROPE.validate_parallel_args(
                constructed=True,
                parallel=True,
                visible_devices=["2"],
                combo_count=2,
            )

    def test_parallel_rejects_duplicate_combos(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ABLATION_ROPE.validate_unique_combos([0, 1, 1])

    def test_constructed_combos_cover_vsn_stage_and_spatial_pooling(self):
        self.assertEqual(
            [ABLATION_ROPE.constructed_combo(i) for i in range(4)],
            [
                {"use_constructed": True, "spatial_mode": "mean", "variable_selection_stage": "county"},
                {"use_constructed": True, "spatial_mode": "attention", "variable_selection_stage": "county"},
                {"use_constructed": True, "spatial_mode": "mean", "variable_selection_stage": "grid"},
                {"use_constructed": True, "spatial_mode": "attention", "variable_selection_stage": "grid"},
            ],
        )


if __name__ == "__main__":
    unittest.main()
