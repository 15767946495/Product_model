"""公共模型接口 smoke test：只验证统一 Dataset/collate 能产出各模型所需输入，不实例化神经网络。

默认使用合成 JSONL；设置 CROPNET_INTERFACE_JSONL 与 CROPNET_INTERFACE_DATA_ROOT 可改用真实数据。
"""

import os
import tempfile
import unittest
from pathlib import Path

import torch

from tests.test_cropnet_dataset import make_rows, write_jsonl
from tests.test_cropnet_schema import write_h5
from train_dataset.collate import collate_county_year, collate_gnn_window, monthly_weather_view
from train_dataset.dataset import (
    FINETUNE_YEARS, SUPERVISED_YEARS, TEST_YEARS, YEARS, CropNetDataset,
    build_gnn_windows, load_jsonl, select_years,
)

COUNTIES = {"19001": ((41.0, -94.5), (41.1, -94.4)), "19003": ((40.7, -94.7), (40.8, -94.7), (40.9, -94.7))}


class ModelInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        real = os.environ.get("CROPNET_INTERFACE_JSONL")
        if real:
            cls.tmp = None
            cls.path = Path(real)
            cls.root = Path(os.environ.get("CROPNET_INTERFACE_DATA_ROOT", "/data/raid0/hqx/DataSrc"))
            return
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        rows = make_rows(counties=COUNTIES)
        for row in rows:
            if row["year"] in (2021, 2022):
                write_h5(cls.root, row)
        cls.path = write_jsonl(cls.root / "dataset.jsonl", rows)

    @classmethod
    def tearDownClass(cls):
        if cls.tmp:
            cls.tmp.cleanup()

    def batch(self, years, load_images=False, n=2):
        ds = CropNetDataset(self.path, years, load_images=load_images, data_root=self.root)
        return ds, collate_county_year([ds[i] for i in range(min(n, len(ds)))])

    def test_multimodal_interface_shapes(self):
        """PhenoYieldNet / MMST-ViT：images (B,6,G,3,224,224)、weather (B,6,G,28,11)。"""
        _, batch = self.batch(FINETUNE_YEARS, load_images=True)
        B, G = batch["grid_mask"].shape
        self.assertEqual(batch["images"].shape, (B, 6, G, 3, 224, 224))
        self.assertEqual(monthly_weather_view(batch["weather"]).shape, (B, 6, G, 28, 11))
        self.assertEqual(batch["labels"].shape, (B,))
        self.assertEqual(batch["images"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(batch["images"]).all())

    def test_tft_interface_has_grid_weather_and_soil(self):
        """TFT：原始网格逐日气象、坐标、土壤、标签；不依赖 grid_cache.pt。"""
        ds, batch = self.batch(SUPERVISED_YEARS)
        B, G = batch["grid_mask"].shape
        self.assertEqual(batch["weather"].shape, (B, G, 168, 11))
        self.assertEqual(batch["coords"].shape, (B, G, 2))
        self.assertEqual(batch["soil"].shape, (B, 7))
        self.assertEqual(batch["labels"].shape, (B,))
        self.assertEqual({s["year"] for s in ds.samples}, set(SUPERVISED_YEARS))
        self.assertNotIn("grid_cache", str(vars(ds)))

    def test_gnn_interface_has_five_year_county_graph(self):
        ds = CropNetDataset(self.path, YEARS, data_root=self.root)
        windows = build_gnn_windows([ds[i] for i in range(len(ds))])
        train, test = collate_gnn_window(windows["train"]), collate_gnn_window(windows["test"])
        self.assertEqual(train["years"].tolist(), [2017, 2018, 2019, 2020, 2021])
        self.assertEqual(test["years"].tolist(), [2018, 2019, 2020, 2021, 2022])
        self.assertEqual(train["fips"], test["fips"])
        self.assertTrue(torch.equal(train["edge_index"], test["edge_index"]))
        zeroed = [[dict(item, label=torch.tensor(-1.0)) for item in w] for w in windows["test"]]
        self.assertTrue(torch.equal(collate_gnn_window(zeroed)["edge_index"], test["edge_index"]))
        self.assertEqual(train["weather"].shape[1:], (5, 168, 11))

    def test_unet_convlstm_interface_is_single_county_year(self):
        _, batch = self.batch(TEST_YEARS, load_images=True)
        B, G = batch["grid_mask"].shape
        self.assertEqual(len(set(batch["sample_id"])), B)
        self.assertEqual(set(batch["year"].tolist()), {2022})
        self.assertEqual(batch["images"].shape[:3], (B, 6, G))
        self.assertEqual(monthly_weather_view(batch["weather"]).shape[:3], (B, 6, G))
        self.assertEqual(batch["image_mask"].shape, (B, 6, G))

    def test_all_models_share_2022_sample_ids(self):
        expected = [s["sample_id"] for s in select_years(load_jsonl(self.path), TEST_YEARS)]
        county_year = CropNetDataset(self.path, TEST_YEARS, data_root=self.root).sample_ids
        full = CropNetDataset(self.path, YEARS, data_root=self.root)
        gnn = [w[-1]["sample_id"] for w in build_gnn_windows([full[i] for i in range(len(full))])["test"]]
        self.assertEqual(county_year, expected)
        self.assertEqual(gnn, expected)


if __name__ == "__main__":
    unittest.main()
