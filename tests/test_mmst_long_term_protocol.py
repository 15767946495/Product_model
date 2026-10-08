import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "MMST-ViT" / "data"


class MMSTLongTermProtocolTests(unittest.TestCase):
    def test_long_term_weather_uses_previous_four_years(self):
        expected = {
            "corn_train.json": (2021, (2017, 2018, 2019, 2020)),
            "corn_val.json": (2022, (2018, 2019, 2020, 2021)),
        }
        for filename, (target_year, history_years) in expected.items():
            records = json.loads((DATA / filename).read_text(encoding="utf-8"))
            with self.subTest(filename=filename):
                self.assertTrue(records)
                self.assertTrue(all(record["year"] == target_year for record in records))
                self.assertTrue(all(
                    tuple(int(paths[0].split("/")[2]) for paths in record["data"]["HRRR"]["long_term"])
                    == history_years
                    for record in records
                ))


if __name__ == "__main__":
    unittest.main()
