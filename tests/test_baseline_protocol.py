import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
COMMON_DIR = ROOT / "BaseLine_Model" / "common"
sys.path.insert(0, str(COMMON_DIR.parent))
SPEC = importlib.util.spec_from_file_location("baseline_data", COMMON_DIR / "data.py")
assert SPEC is not None and SPEC.loader is not None
baseline_data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline_data)


class BaselineProtocolTests(unittest.TestCase):
    def test_split_years_keeps_2022_out_of_training_and_validation(self):
        samples = [{"year": year} for year in (2019, 2020, 2021, 2022)]
        train, val, test = baseline_data.split_years(samples, val_year=2021, test_year=2022)
        self.assertEqual([s["year"] for s in train], [2019, 2020])
        self.assertEqual([s["year"] for s in val], [2021])
        self.assertEqual([s["year"] for s in test], [2022])

    def test_split_years_rejects_non_chronological_protocol(self):
        with self.assertRaisesRegex(ValueError, "test_year"):
            baseline_data.split_years([], val_year=2022, test_year=2021)


if __name__ == "__main__":
    unittest.main()
