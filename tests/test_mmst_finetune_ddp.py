import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MMST_ROOT = ROOT / "MMST-ViT"
if str(MMST_ROOT) not in sys.path:
    sys.path.insert(0, str(MMST_ROOT))

from models_pvt_simclr import PVTSimCLR
from current_data import filter_complete_records, has_complete_inputs


class MMSTFinetuneDDPTests(unittest.TestCase):
    def test_unused_pvt_classification_head_is_frozen(self):
        model = PVTSimCLR(
            "pvt_tiny", out_dim=32, context_dim=9, num_head=4,
            pretrained=False,
        )

        self.assertFalse(model.backbone.head.weight.requires_grad)
        self.assertFalse(model.backbone.head.bias.requires_grad)
        self.assertTrue(model.proj.weight.requires_grad)

    def test_stale_manifest_records_are_filtered_before_training(self):
        records = [
            {"FIPS": "17031", "year": 2021},
            {"FIPS": "17033", "year": 2021},
        ]

        with patch("current_data.has_complete_inputs", side_effect=[True, False]):
            valid, skipped = filter_complete_records(records)

        self.assertEqual([record["FIPS"] for record in valid], ["17031"])
        self.assertEqual([record["FIPS"] for record in skipped], ["17033"])

    def test_incomplete_daily_weather_grid_is_rejected(self):
        record = {"FIPS": "36043", "year": 2022, "state": "new_york"}

        with patch("current_data.weather_files_exist", return_value=True), \
             patch("current_data.sentinel_fips_exist", return_value=True), \
             patch("current_data.short_weather_is_complete", return_value=False), \
             patch("current_data.long_weather_is_complete", return_value=True):
            self.assertFalse(has_complete_inputs(record))


if __name__ == "__main__":
    unittest.main()
