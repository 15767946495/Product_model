import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
TFT_DIR = ROOT / "TFT_model"
sys.path.insert(0, str(TFT_DIR))
SPEC = importlib.util.spec_from_file_location("error_report", TFT_DIR / "error_report.py")
assert SPEC is not None and SPEC.loader is not None
error_report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(error_report)


class ErrorReportTests(unittest.TestCase):
    def test_prediction_records_keep_county_identity_and_error(self):
        records = error_report.prediction_records(
            states=["iowa"],
            years=[2022],
            fips=["19001"],
            counties=["Adair"],
            predictions=[101.5],
            labels=[100.0],
        )

        self.assertEqual(records, [{
            "state": "iowa",
            "year": 2022,
            "fips": "19001",
            "county": "Adair",
            "prediction": 101.5,
            "label": 100.0,
            "error": 1.5,
            "abs_error": 1.5,
        }])

    def test_metrics_by_group_reports_rmse_mae_bias_and_r2(self):
        groups = np.array(["a", "a", "b"])
        predictions = np.array([1.0, 3.0, 5.0])
        labels = np.array([2.0, 2.0, 4.0])

        report = error_report.metrics_by_group(groups, predictions, labels)

        self.assertEqual(report["a"]["n"], 2)
        self.assertAlmostEqual(report["a"]["rmse"], 1.0)
        self.assertAlmostEqual(report["a"]["mae"], 1.0)
        self.assertAlmostEqual(report["a"]["bias"], 0.0)
        self.assertEqual(report["b"]["r2"], None)

    def test_metrics_by_group_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            error_report.metrics_by_group(["a"], [1.0, 2.0], [1.0])


if __name__ == "__main__":
    unittest.main()
