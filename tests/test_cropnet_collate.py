import json
import tempfile
import unittest
from pathlib import Path

import torch

from tests.test_cropnet_dataset import make_rows, write_jsonl
from tests.test_cropnet_schema import write_h5
from train_dataset.collate import collate_county_year, collate_gnn_window, monthly_weather_view
from train_dataset.dataset import CropNetDataset, build_gnn_windows


class CollateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # 19001 有 2 个网格，19003 有 3 个网格
        counties = {"19001": ((41.0, -94.5), (41.1, -94.4)),
                    "19003": ((40.7, -94.7), (40.8, -94.7), (40.9, -94.7))}
        self.rows = make_rows(counties=counties)
        self.path = write_jsonl(self.root / "dataset.jsonl", self.rows)

    def tearDown(self):
        self.tmp.cleanup()

    def test_county_year_collate_pads_grids_with_mask(self):
        ds = CropNetDataset(self.path, (2021,))
        first = ds[0]
        first["weather"] = torch.zeros_like(first["weather"])  # 真实零值网格
        batch = collate_county_year([first, ds[1]])
        self.assertEqual(batch["weather"].shape, (2, 3, 168, 11))
        self.assertEqual(batch["coords"].shape, (2, 3, 2))
        self.assertEqual(batch["soil"].shape, (2, 7))
        self.assertEqual(batch["labels"].shape, (2,))
        self.assertEqual(batch["grid_mask"].tolist(), [[True, True, False], [True, True, True]])
        self.assertEqual(batch["long_term"].shape, (2, 3, 4, 12, 11))
        self.assertEqual(batch["long_term_mask"].tolist(), batch["grid_mask"].tolist())
        self.assertEqual(batch["sample_id"], ["19001_2021", "19003_2021"])
        pre = collate_county_year([CropNetDataset(self.path, (2018,))[0]])
        self.assertFalse(pre["long_term_mask"].any())
        self.assertTrue(pre["grid_mask"].all())

    def test_image_collate_preserves_month_and_grid_masks(self):
        for row in self.rows:
            if row["year"] == 2021:
                write_h5(self.root, row)
        ds = CropNetDataset(self.path, (2021,), load_images=True, data_root=self.root)
        batch = collate_county_year([ds[0], ds[1]])
        self.assertEqual(batch["images"].shape, (2, 6, 3, 3, 224, 224))
        self.assertEqual(batch["image_mask"].shape, (2, 6, 3))
        self.assertEqual(batch["image_mask"][0, :, 2].tolist(), [False] * 6)
        self.assertTrue(batch["image_mask"][1].all())
        self.assertEqual(batch["images"][0, :, 2].abs().sum().item(), 0.0)

    def test_monthly_weather_view(self):
        weather = torch.arange(2 * 3 * 168 * 11, dtype=torch.float32).reshape(2, 3, 168, 11)
        view = monthly_weather_view(weather)
        self.assertEqual(view.shape, (2, 6, 3, 28, 11))
        self.assertTrue(torch.equal(view[1, 2, 0, 5], weather[1, 0, 2 * 28 + 5]))

    def test_gnn_window_collate_has_five_year_axis(self):
        ds = CropNetDataset(self.path, range(2017, 2023))
        windows = build_gnn_windows([ds[i] for i in range(len(ds))])
        batch = collate_gnn_window(list(reversed(windows["train"])))
        self.assertEqual(batch["fips"], ["19001", "19003"])
        self.assertEqual(batch["weather"].shape, (2, 5, 168, 11))
        self.assertEqual(batch["soil"].shape, (2, 7))
        self.assertEqual(batch["years"].tolist(), [2017, 2018, 2019, 2020, 2021])
        self.assertEqual(batch["targets"].tolist(), [154.0, 154.0])
        self.assertEqual(sorted(map(tuple, batch["edge_index"].t().tolist())),
                         [(0, 0), (0, 1), (1, 0), (1, 1)])
        self.assertNotIn("labels", batch)
        test = collate_gnn_window(windows["test"])
        self.assertEqual(test["years"].tolist(), [2018, 2019, 2020, 2021, 2022])
        self.assertEqual(test["fips"], batch["fips"])
        # 修改标签不得改变节点特征
        changed = [[dict(item, label=torch.tensor(0.0)) for item in w] for w in windows["train"]]
        self.assertTrue(torch.equal(collate_gnn_window(changed)["weather"], batch["weather"]))


if __name__ == "__main__":
    unittest.main()
