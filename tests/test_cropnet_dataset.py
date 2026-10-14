import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import h5py
import numpy as np
import torch

from tests.test_cropnet_schema import make_sample, write_h5
from train_dataset import dataset as dataset_module
from train_dataset.dataset import (
    FINETUNE_YEARS, PRETRAIN_YEARS, SUPERVISED_YEARS, TEST_YEARS, YEARS,
    CropNetDataset, build_gnn_windows, load_jsonl, select_years,
)

COUNTY_COORDS = {"19001": ((41.0, -94.5), (41.1, -94.4)), "19003": ((40.9, -94.7),)}
NEIGHBORS = {"19001": ("19003",), "19003": ("19001",)}


def make_rows(years=YEARS, counties=COUNTY_COORDS):
    return [make_sample(fips=f, year=y, coords=counties[f], neighbors=NEIGHBORS[f], label=150.0 + y - 2017)
            for y in years for f in sorted(counties)]


def write_jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return Path(path)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.path = write_jsonl(self.root / "dataset.jsonl", make_rows())

    def tearDown(self):
        self.tmp.cleanup()

    def test_duplicate_sample_id_is_rejected(self):
        rows = make_rows()
        rows.insert(1, rows[0])
        with self.assertRaisesRegex(ValueError, "duplicate sample_id 19001_2017"):
            load_jsonl(write_jsonl(self.root / "dup.jsonl", rows))

    def test_file_invariants(self):
        rows = make_rows()
        with self.assertRaisesRegex(ValueError, "sorted"):
            load_jsonl(write_jsonl(self.root / "a.jsonl", [rows[1], rows[0], *rows[2:]]))
        with self.assertRaisesRegex(ValueError, "expected years"):
            load_jsonl(write_jsonl(self.root / "b.jsonl", [r for r in rows if r["sample_id"] != "19003_2019"]))
        bad = [dict(r, neighbor_fips=[]) if r["sample_id"] == "19001_2020" else r for r in rows]
        with self.assertRaisesRegex(ValueError, "neighbor_fips differs"):
            load_jsonl(write_jsonl(self.root / "c.jsonl", bad))
        asym = [dict(r, neighbor_fips=[]) if r["fips"] == "19003" else r for r in rows]
        with self.assertRaisesRegex(ValueError, "not symmetric"):
            load_jsonl(write_jsonl(self.root / "d.jsonl", asym))
        (self.root / "e.jsonl").write_text(json.dumps(rows[0]) + "\n{oops\n")
        with self.assertRaisesRegex(ValueError, "e.jsonl:2: invalid JSON"):
            load_jsonl(self.root / "e.jsonl")

    def test_check_dataset_file_uses_streaming_invariants(self):
        with mock.patch.object(dataset_module, "check_dataset_invariants", side_effect=AssertionError("list invariant path used")):
            summary = dataset_module.check_dataset_file(self.path)

        self.assertEqual(summary["samples"], 12)
        self.assertEqual(summary["counties"], 2)

    def test_select_years_matches_protocol(self):
        samples = load_jsonl(self.path)
        self.assertEqual(PRETRAIN_YEARS, (2017, 2018, 2019, 2020))
        self.assertEqual(FINETUNE_YEARS, (2021,))
        self.assertEqual(SUPERVISED_YEARS, (2017, 2018, 2019, 2020, 2021))
        self.assertEqual(TEST_YEARS, (2022,))
        pretrain = select_years(samples, PRETRAIN_YEARS)
        self.assertEqual([s["sample_id"] for s in pretrain],
                         [f"{f}_{y}" for y in PRETRAIN_YEARS for f in ("19001", "19003")])
        self.assertEqual([s["sample_id"] for s in select_years(reversed(samples), TEST_YEARS)],
                         ["19001_2022", "19003_2022"])
        with self.assertRaises(ValueError):
            select_years(samples, (2023,))

    def test_dataset_items_without_images(self):
        with mock.patch.object(h5py, "File") as opened:
            ds = CropNetDataset(self.path, FINETUNE_YEARS, data_root=self.root)
            item = ds[0]
        opened.assert_not_called()
        self.assertEqual(ds.sample_ids, ["19001_2021", "19003_2021"])
        self.assertEqual(item["weather"].shape, (2, 168, 11))
        self.assertEqual(item["weather"].dtype, torch.float32)
        self.assertEqual(item["coords"].shape, (2, 2))
        self.assertEqual(item["soil"].shape, (7,))
        self.assertAlmostEqual(item["label"].item(), 154.0)
        self.assertEqual(item["long_term"].shape, (2, 4, 12, 11))
        self.assertEqual(len(item["remote_sensing"]), 6)
        self.assertNotIn("images", item)
        self.assertIsNone(CropNetDataset(self.path, (2017,))[0]["long_term"])

    def test_images_are_loaded_lazily(self):
        for row in make_rows(years=(2021,)):
            write_h5(self.root, row)
        real = h5py.File
        with mock.patch.object(h5py, "File", side_effect=real) as opened:
            ds = CropNetDataset(self.path, (2021,), load_images=True, data_root=self.root)
            opened.assert_not_called()
            item = ds[0]
            self.assertGreater(opened.call_count, 0)
        images = item["images"]
        self.assertEqual(images.shape, (6, 2, 3, 224, 224))
        self.assertEqual(images.dtype, torch.float32)
        self.assertTrue(torch.isfinite(images).all())
        self.assertAlmostEqual(images[0, 0].mean().item(), 100 / 255, places=4)
        self.assertAlmostEqual(images[5, 1].mean().item(), 101 / 255, places=4)

    def test_image_errors_report_sample_and_path(self):
        ds = CropNetDataset(self.path, (2022,), load_images=True, data_root=self.root)
        with self.assertRaisesRegex(ValueError, r"19001_2022\.19001_grid_0000\.remote_sensing\[0\] Sentinel-2"):
            ds[0]

    def test_gnn_windows_are_exactly_five_years(self):
        windows = build_gnn_windows(load_jsonl(self.path))
        self.assertEqual([[s["year"] for s in w] for w in windows["train"]], [[2017, 2018, 2019, 2020, 2021]] * 2)
        self.assertEqual([[s["year"] for s in w] for w in windows["test"]], [[2018, 2019, 2020, 2021, 2022]] * 2)
        self.assertEqual([w[0]["fips"] for w in windows["test"]], ["19001", "19003"])
        partial = [s for s in load_jsonl(self.path) if s["sample_id"] != "19003_2022"]
        with self.assertRaisesRegex(ValueError, r"19003: GNN test window missing years \[2022\]"):
            build_gnn_windows(partial)


if __name__ == "__main__":
    unittest.main()
