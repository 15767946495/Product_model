import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MMST_ROOT = ROOT / "MMST-ViT"
sys.path.insert(0, str(MMST_ROOT))

from current_data import normalize_state_name, validate_pretrain_records


def record(state, year=2019, fips="19001"):
    return {"state": state, "year": year, "FIPS": fips, "data": {}}


class MMSTPretrainPipelineTests(unittest.TestCase):
    def test_normalize_state_name_accepts_new_york_spellings(self):
        self.assertEqual(normalize_state_name("New York"), "new_york")
        self.assertEqual(normalize_state_name("new_york"), "new_york")

    def test_validate_pretrain_records_accepts_exact_four_state_subset(self):
        expected_states = {
            "illinois",
            "iowa",
            "louisiana",
            "mississippi",
        }
        records = [
            record(state, year=year, fips=f"{index:05d}")
            for index, (state, year) in enumerate(
                (state, year)
                for state in sorted(expected_states)
                for year in (2017, 2018, 2019, 2020)
            )
        ]

        counts = validate_pretrain_records(records)

        self.assertEqual(set(counts), expected_states)
        self.assertEqual(counts, {state: 4 for state in expected_states})

    def test_validate_pretrain_records_rejects_empty_manifest(self):
        with self.assertRaisesRegex(ValueError, r"empty"):
            validate_pretrain_records([])

    def test_validate_pretrain_records_rejects_out_of_scope_state(self):
        with self.assertRaisesRegex(
            ValueError, r"outside the four-state protocol.*texas"
        ):
            validate_pretrain_records([record("Texas")])

    def test_validate_pretrain_records_rejects_out_of_scope_year(self):
        with self.assertRaisesRegex(
            ValueError, r"outside pretraining years.*2021"
        ):
            validate_pretrain_records([record("Iowa", year=2021)])

    def test_run_pretrain_uses_main_entrypoint(self):
        script = (MMST_ROOT / "run_pretrain.sh").read_text(encoding="utf-8")
        self.assertIn('exec "$PYTHON" main_pretrain_mmst_vit.py "$@"', script)

    def test_pretrain_entrypoint_validates_the_four_state_protocol(self):
        source = (MMST_ROOT / "main_pretrain_mmst_vit.py").read_text(encoding="utf-8")

        self.assertIn("validate_pretrain_records(records)", source)
        self.assertNotIn("len(states) < 6", source)


if __name__ == "__main__":
    unittest.main()
