import os
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_cropnet_dataset import make_rows, write_jsonl
from tests.test_cropnet_schema import write_h5
from train_dataset import migration_check as mc

PASS = lambda: {"returncode": 0}  # noqa: E731
FAIL = lambda: {"returncode": 1}  # noqa: E731


class MigrationCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.old, self.new = base / "old", base / "new"
        (self.old / "DataSrc" / "Sentinel-2 Imagery").mkdir(parents=True)
        (self.old / "DataSrc" / "Sentinel-2 Imagery" / "a.h5").write_bytes(b"x" * 10)
        rows = make_rows()
        for row in rows:
            write_h5(self.new, row)
        self.target = base / "raid" / "dataset.jsonl"
        self.target.parent.mkdir()
        write_jsonl(self.target, rows)
        self.link = base / "proj" / "dataset.jsonl"
        self.link.parent.mkdir()
        self.link.symlink_to(self.target)
        (self.new / "download_audit.jsonl").write_text(
            "".join(json.dumps({"fips": f, "status": "retained", "reason": ""}) + "\n" for f in ("19001", "19003"))
        )

    def tearDown(self):
        self.tmp.cleanup()

    def report(self, old=None, jsonl=None, **kw):
        args = dict(link=self.link, expected_target=self.target, run_tests=PASS)
        args.update(kw)
        return mc.migration_report(old or self.old, self.new, jsonl or self.link, **args)

    def test_all_checks_pass(self):
        report = self.report()
        self.assertTrue(report["ready_to_delete"], report["checks"])
        self.assertEqual(report["dataset"]["counties"], 2)
        self.assertTrue(report["year_county_sets"]["all_equal"])
        self.assertEqual(report["year_county_sets"]["years"]["2017"]["count"], 2)
        self.assertEqual(report["hdf5_sampling"]["sample_count"], 6)
        self.assertEqual(report["hdf5_sampling"]["years"], [2017, 2018, 2019, 2020, 2021, 2022])
        self.assertTrue(report["hdf5_sampling"]["ok"])
        self.assertEqual(report["file_counts"]["old"]["sentinel"]["files"], 1)  # 嵌套在 DataSrc/ 下也能识别
        self.assertGreater(report["file_counts"]["new"]["sentinel"]["files"], 0)
        self.assertEqual(len(report["jsonl_sha256"]), 64)

    def test_year_county_sets_report_counts_and_differences(self):
        path = self.new / "year-sets.jsonl"
        rows = [
            {"year": 2017, "fips": "19001"},
            {"year": 2017, "fips": "19003"},
            {"year": 2018, "fips": "19001"},
            {"year": 2018, "fips": "19005"},
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))

        report = mc.compare_year_county_sets(path, years=(2017, 2018))

        self.assertFalse(report["all_equal"])
        self.assertEqual(report["reference_year"], 2017)
        self.assertEqual(report["years"]["2017"], {
            "count": 2, "missing_from_year": [], "extra_in_year": [],
        })
        self.assertEqual(report["years"]["2018"], {
            "count": 2, "missing_from_year": ["19003"], "extra_in_year": ["19005"],
        })

    def test_report_summarizes_download_audit(self):
        audit = self.new / "download_audit.jsonl"
        rows = [
            {"fips": "19001", "year": 2017, "status": "retained", "reason": ""},
            {"fips": "19001", "year": 2018, "status": "retained", "reason": ""},
            {"fips": "19003", "year": 2017, "status": "excluded", "reason": "weather: missing day; remote: missing image"},
            {"fips": "19003", "year": 2018, "status": "excluded", "reason": "county incomplete in years [2017]"},
        ]
        audit.write_text("".join(json.dumps(row) + "\n" for row in rows))

        report = self.report()

        self.assertEqual(report["download_audit"], {
            "path": str(audit),
            "candidates": 2,
            "retained_counties": 1,
            "excluded_reasons": {"remote": 1, "weather": 1},
        })

    def test_missing_old_root_is_explicitly_reported(self):
        report = self.report(old=self.old / "missing")
        self.assertEqual(report["old_root_status"], "missing_unable_to_record_file_counts")
        self.assertFalse(report["checks"]["file_counts_recorded"])
        self.assertFalse(report["ready_to_delete"])

    def test_delete_gate_requires_all_checks(self):
        self.assertFalse(self.report(run_tests=FAIL)["ready_to_delete"])
        self.assertFalse(self.report(old=self.old / "missing")["checks"]["file_counts_recorded"])
        self.assertFalse(self.report(jsonl=self.new / "none.jsonl")["checks"]["jsonl_exists"])
        self.assertFalse(self.report(expected_target=self.new / "other.jsonl")["checks"]["symlink_ok"])
        h5 = next(self.new.rglob("*.h5"))
        h5.rename(h5.with_suffix(".bak"))
        failed = self.report()
        self.assertFalse(failed["checks"]["hdf5_refs_ok"])
        self.assertIn("file not found", failed["dataset_error"])
        h5.with_suffix(".bak").rename(h5)
        self.target.write_text(self.target.read_text().splitlines()[0] + "\n")
        self.assertFalse(self.report()["checks"]["schema_ok"])
        shutil.rmtree(self.new)
        self.assertFalse(self.report()["checks"]["new_root_exists"])

    def test_never_deletes(self):
        with mock.patch.object(shutil, "rmtree") as rmtree, mock.patch.object(os, "remove") as remove, \
                mock.patch.object(os, "unlink") as unlink:
            self.report()
        rmtree.assert_not_called()
        remove.assert_not_called()
        unlink.assert_not_called()
        self.assertTrue((self.old / "DataSrc" / "Sentinel-2 Imagery" / "a.h5").is_file())
        source = Path(mc.__file__).read_text()
        for banned in ("rmtree", "os.remove", "unlink(", "rm -rf"):
            self.assertNotIn(banned, source)


if __name__ == "__main__":
    unittest.main()
