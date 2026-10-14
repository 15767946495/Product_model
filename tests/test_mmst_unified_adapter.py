"""MMST-ViT 统一 CropNet adapter：年份协议、形状、9 维上下文、80 网格 padding 与 mask。"""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tests.test_cropnet_dataset import write_jsonl
from tests.test_cropnet_schema import make_sample, write_h5
from train_dataset.dataset import WEATHER_FEATURES, YEARS, CropNetDataset

MMST_DIR = Path(__file__).resolve().parents[1] / "MMST-ViT"


def load_mmst_module(name):
    """MMST-ViT 目录名含连字符，按文件路径导入；模块内部使用目录内绝对导入。"""
    if str(MMST_DIR) not in sys.path:
        sys.path.insert(0, str(MMST_DIR))
    spec = importlib.util.spec_from_file_location(f"mmst_{Path(name).stem}", MMST_DIR / name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_mmst_adapter():
    return load_mmst_module("dataset_adapter.py")


COUNTIES = {"19001": ((41.0, -94.5), (41.1, -94.4), (41.2, -94.3)), "19003": ((40.8, -94.7), (40.9, -94.7))}
NEIGHBORS = {"19001": ("19003",), "19003": ("19001",)}


def make_cropnet_fixture(root, counties=COUNTIES, image_years=(2021, 2022)):
    """统一 Dataset 要求每县覆盖全部六年，因此夹具写满 2017--2022，再由 adapter 按年份选择。"""
    rows = [make_sample(fips=f, year=y, coords=counties[f], neighbors=[n for n in NEIGHBORS[f] if n in counties],
                        label=150.0 + y - 2017 + 10.0 * i)
            for y in YEARS for i, f in enumerate(sorted(counties))]
    for row in rows:
        if row["year"] in image_years:
            write_h5(root, row)
    return write_jsonl(Path(root) / "dataset.jsonl", rows)


def make_batch_with_grid_count(count, sample_id="19001_2021"):
    """直接构造 MMST 布局 batch（不读文件），用于 padding 上限测试。"""
    return {
        "images": torch.rand((1, 6, count, 3, 4, 4)),
        "image_mask": torch.ones((1, 6, count), dtype=torch.bool),
        "weather": torch.rand((1, 6, count, 28, 11)),
        "weather_context": torch.rand((1, 6, count, 28, 9)),
        "coords": torch.rand((1, count, 2)),
        "grid_mask": torch.ones((1, count), dtype=torch.bool),
        "long_term": torch.rand((1, 4, 12, 9)),
        "labels": torch.tensor([150.0]),
        "sample_id": [sample_id], "fips": [sample_id[:5]], "year": torch.tensor([2021]),
    }


class MMSTAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = load_mmst_adapter()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.path = make_cropnet_fixture(cls.root, counties={"19001": COUNTIES["19001"]})
        cls.root2 = cls.root / "two"
        cls.root2.mkdir()
        cls.path2 = make_cropnet_fixture(cls.root2)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ---------------------------------------------------------------- 任务 1
    def test_protocol_year_constants(self):
        self.assertEqual(self.adapter.MMST_YEARS_PRETRAIN, (2017, 2018, 2019, 2020))
        self.assertEqual(self.adapter.MMST_YEARS_FINETUNE, (2021,))
        self.assertEqual(self.adapter.MMST_YEARS_TEST, (2022,))

    def test_adapter_reads_unified_jsonl_for_three_protocol_years(self):
        adapter = self.adapter
        batch_2017 = adapter.make_batch(self.path, self.root, years=(2017,), load_images=False)
        batch_2021 = adapter.make_batch(self.path, self.root, years=(2021,), load_images=True)
        batch_2022 = adapter.make_batch(self.path, self.root, years=(2022,), load_images=True)

        self.assertIsNone(batch_2017["long_term"])
        self.assertNotIn("images", batch_2017)
        self.assertEqual(tuple(batch_2021["images"].shape), (1, 6, 3, 3, 224, 224))
        self.assertEqual(tuple(batch_2021["weather"].shape), (1, 6, 3, 28, 11))
        self.assertEqual(batch_2021["grid_mask"].tolist(), [[True, True, True]])
        self.assertEqual(tuple(batch_2021["long_term"].shape), (1, 4, 12, 9))
        self.assertEqual(batch_2022["sample_id"], ["19001_2022"])
        self.assertEqual(batch_2021["labels"].dtype, torch.float32)
        for key in ("images", "weather", "weather_context", "long_term", "grid_mask", "image_mask",
                    "labels", "sample_id", "fips", "year", "coords"):
            self.assertIn(key, batch_2021)

    def test_monthly_weather_matches_daily_order(self):
        ds = CropNetDataset(self.path, (2021,), data_root=self.root)
        daily = ds[0]["weather"]  # (G,168,11)
        batch = self.adapter.make_batch(self.path, self.root, years=(2021,))
        for month in range(6):
            torch.testing.assert_close(batch["weather"][0, month], daily[:, month * 28:(month + 1) * 28])

    def test_dataloader_preserves_dataset_sample_order(self):
        loader = self.adapter.build_mmst_dataloader(self.path2, self.root2, (2022,), batch_size=1,
                                                    shuffle=False, load_images=True)
        ids = [sid for batch in loader for sid in batch["sample_id"]]
        self.assertEqual(ids, CropNetDataset(self.path2, (2022,), data_root=self.root2).sample_ids)

    def test_dataloader_rejects_years_outside_protocol(self):
        with self.assertRaisesRegex(ValueError, "years outside protocol"):
            self.adapter.build_mmst_dataloader(self.path, self.root, (2016,), batch_size=1, shuffle=False)

    def test_missing_image_error_keeps_sample_and_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_cropnet_fixture(tmp, counties={"19001": COUNTIES["19001"]}, image_years=(2021,))
            with self.assertRaisesRegex(ValueError, r"19001_2022\.19001_grid_0000\.remote_sensing\[0\].*\.h5"):
                self.adapter.make_batch(path, tmp, years=(2022,), load_images=True)

    def test_mixed_long_term_batch_is_rejected(self):
        ds = CropNetDataset(self.path, (2020, 2021), data_root=self.root)
        with self.assertRaisesRegex(ValueError, "long-term weather"):
            self.adapter.collate_mmst([ds[0], ds[1]])

    # ---------------------------------------------------------------- 任务 2
    def test_context_features_are_selected_by_name_and_exclude_uv(self):
        adapter = self.adapter
        values = np.arange(2 * 11, dtype=np.float32).reshape(2, 11)
        names = list(WEATHER_FEATURES)
        selected = adapter.select_context_features(values, names)
        expected = [names.index(name) for name in adapter.MMST_CONTEXT_FEATURES]
        np.testing.assert_array_equal(selected, values[:, expected])
        self.assertEqual(len(adapter.MMST_CONTEXT_FEATURES), 9)
        self.assertNotIn("u_wind_m_s", adapter.MMST_CONTEXT_FEATURES)
        self.assertNotIn("v_wind_m_s", adapter.MMST_CONTEXT_FEATURES)

    def test_context_features_follow_names_not_positions(self):
        adapter = self.adapter
        names = list(reversed(WEATHER_FEATURES))
        values = np.arange(11, dtype=np.float32)[None]
        selected = adapter.select_context_features(values, names)
        expected = [names.index(name) for name in adapter.MMST_CONTEXT_FEATURES]
        np.testing.assert_array_equal(selected, values[:, expected])

    def test_context_features_accept_tensors(self):
        values = torch.arange(11, dtype=torch.float32).reshape(1, 11)
        selected = self.adapter.select_context_features(values, list(WEATHER_FEATURES))
        self.assertIsInstance(selected, torch.Tensor)
        self.assertEqual(tuple(selected.shape), (1, 9))

    def test_context_features_reject_missing_or_duplicate_names(self):
        adapter = self.adapter
        values = np.zeros((1, 11), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "missing context feature"):
            adapter.select_context_features(np.zeros((1, 10), dtype=np.float32), list(WEATHER_FEATURES[:-1]))
        with self.assertRaisesRegex(ValueError, "duplicate weather feature"):
            adapter.select_context_features(values, list(WEATHER_FEATURES[:-1]) + [WEATHER_FEATURES[0]])
        with self.assertRaisesRegex(ValueError, "last dimension"):
            adapter.select_context_features(np.zeros((1, 9), dtype=np.float32), list(WEATHER_FEATURES))

    def test_batch_context_is_selected_from_weather(self):
        batch = self.adapter.make_batch(self.path, self.root, years=(2021,))
        index = [WEATHER_FEATURES.index(n) for n in self.adapter.MMST_CONTEXT_FEATURES]
        self.assertEqual(tuple(batch["weather_context"].shape), (1, 6, 3, 28, 9))
        torch.testing.assert_close(batch["weather_context"], batch["weather"][..., index])

    # ---------------------------------------------------------------- 任务 3
    def test_78_grids_are_padded_to_80_and_masked(self):
        padded = self.adapter.pad_mmst_grids(make_batch_with_grid_count(78), max_grids=80)
        self.assertEqual(tuple(padded["grid_mask"].shape), (1, 80))
        self.assertTrue(padded["grid_mask"][0, :78].all())
        self.assertFalse(padded["grid_mask"][0, 78:].any())
        self.assertEqual(padded["weather"].shape[2], 80)
        self.assertEqual(padded["weather_context"].shape[2], 80)
        self.assertEqual(padded["images"].shape[2], 80)
        self.assertEqual(tuple(padded["image_mask"].shape), (1, 6, 80))
        self.assertEqual(tuple(padded["coords"].shape), (1, 80, 2))
        self.assertEqual(float(padded["images"][:, :, 78:].abs().sum()), 0.0)

    def test_more_than_80_grids_is_rejected_without_county_drop(self):
        with self.assertRaisesRegex(ValueError, "19001_2021.*81.*80"):
            self.adapter.pad_mmst_grids(make_batch_with_grid_count(81), max_grids=80)

    def test_dataloader_pads_to_80_grids(self):
        loader = self.adapter.build_mmst_dataloader(self.path2, self.root2, (2021,), batch_size=2,
                                                    shuffle=False, load_images=True)
        batch = next(iter(loader))
        self.assertEqual(tuple(batch["images"].shape[:3]), (2, 6, 80))
        self.assertEqual(batch["grid_mask"].sum(dim=1).tolist(), [3, 2])
        self.assertFalse(batch["image_mask"][:, :, 3:].any())

    def test_long_term_is_grid_masked_mean(self):
        ds = CropNetDataset(self.path2, (2021,), data_root=self.root2)
        items = [ds[0], ds[1]]
        index = [WEATHER_FEATURES.index(n) for n in self.adapter.MMST_CONTEXT_FEATURES]
        items[1] = dict(items[1], long_term=items[1]["long_term"] * torch.arange(1, 3).view(2, 1, 1, 1))
        batch = self.adapter.collate_mmst(items)
        torch.testing.assert_close(batch["long_term"][0], items[0]["long_term"].mean(0)[..., index])
        torch.testing.assert_close(batch["long_term"][1], items[1]["long_term"].mean(0)[..., index])


class MMSTModelMaskTests(unittest.TestCase):
    """任务 3：num_grid=80，padding 网格不进入 PVT、空间注意力和输出。"""

    @classmethod
    def setUpClass(cls):
        load_mmst_adapter()  # 把 MMST-ViT 目录加入 sys.path
        cls.models = load_mmst_module("models_mmst_vit.py")
        simclr = load_mmst_module("models_pvt_simclr.py")
        torch.manual_seed(0)
        pvt = simclr.PVTSimCLR("pvt_tiny", out_dim=32, context_dim=9, num_head=4, pretrained=False)
        cls.model = cls.models.MMST_ViT(out_dim=1, pvt_backbone=pvt, context_dim=9, dim=32, heads=2,
                                        dim_head=16, depth=1, batch_size=8).eval()

    def inputs(self, G):
        g = torch.Generator().manual_seed(1)
        return (torch.rand((1, 6, G, 3, 32, 32), generator=g), torch.rand((1, 6, G, 28, 9), generator=g),
                torch.rand((1, 4, 12, 9), generator=g))

    def test_default_num_grid_is_80(self):
        self.assertEqual(self.models.MMST_ViT.__init__.__defaults__[1], 80)
        self.assertEqual(self.model.pos_embedding.shape[2], 81)  # 80 网格 + 空间 cls token

    @torch.no_grad()
    def test_padding_grids_do_not_change_output(self):
        x, ys, yl = self.inputs(3)
        reference = self.model(x, ys=ys, yl=yl)
        mask = torch.zeros((1, 80), dtype=torch.bool)
        mask[:, :3] = True
        xp, ysp = torch.rand((1, 6, 80, 3, 32, 32)), torch.rand((1, 6, 80, 28, 9))
        xp[:, :, :3], ysp[:, :, :3] = x, ys
        torch.testing.assert_close(self.model(xp, ys=ysp, yl=yl, grid_mask=mask), reference)

    @torch.no_grad()
    def test_more_than_80_grids_is_rejected(self):
        x, ys, yl = self.inputs(81)
        with self.assertRaisesRegex(ValueError, "81.*80"):
            self.model(x, ys=ys, yl=yl)

    def test_missing_long_term_is_rejected(self):
        x, ys, _ = self.inputs(2)
        with self.assertRaisesRegex(ValueError, "long-term"):
            self.model(x, ys=ys, yl=None)

    def test_masked_grid_pool(self):
        features = torch.tensor([[[1.0], [3.0], [100.0]]])
        mask = torch.tensor([[True, True, False]])
        torch.testing.assert_close(self.models.masked_grid_pool(features, mask), torch.tensor([[2.0]]))


@unittest.skipUnless(os.environ.get("CROPNET_INTERFACE_JSONL"), "real unified dataset not configured")
class MMSTRealDataSmokeTests(unittest.TestCase):
    """任务 8：真实统一 JSONL + HDF5 读取 smoke。"""

    @classmethod
    def setUpClass(cls):
        cls.adapter = load_mmst_adapter()
        cls.path = Path(os.environ["CROPNET_INTERFACE_JSONL"])
        cls.root = Path(os.environ.get("CROPNET_INTERFACE_DATA_ROOT", "/data/raid0/hqx/DataSrc"))
        cls.datasets = {year: CropNetDataset(cls.path, (year,), load_images=year != 2017, data_root=cls.root)
                        for year in (2017, 2021, 2022)}

    def test_real_samples_for_three_protocol_years(self):
        for year, ds in self.datasets.items():
            batch = self.adapter.collate_mmst([ds[0]], max_grids=self.adapter.MMST_MAX_GRIDS)
            G = int(batch["grid_mask"].sum())
            self.assertEqual(batch["sample_id"], [ds.sample_ids[0]])
            self.assertEqual(tuple(batch["weather"].shape), (1, 6, 80, 28, 11))
            self.assertEqual(tuple(batch["weather_context"].shape), (1, 6, 80, 28, 9))
            self.assertTrue(torch.isfinite(batch["weather"]).all())
            if year == 2017:
                self.assertIsNone(batch["long_term"])
                continue
            self.assertEqual(tuple(batch["long_term"].shape), (1, 4, 12, 9))
            self.assertEqual(tuple(batch["images"].shape), (1, 6, 80, 3, 224, 224))
            self.assertTrue(batch["image_mask"][0, :, :G].all())
            self.assertGreater(float(batch["images"][0, :, :G].std()), 0.0)

    def test_real_2022_sample_order_matches_dataset(self):
        ds = self.datasets[2022]
        self.assertEqual(ds.sample_ids, sorted(ds.sample_ids))
        self.assertEqual(len(ds), 384)
        self.assertTrue(all(sid.endswith("_2022") for sid in ds.sample_ids))


if __name__ == "__main__":
    unittest.main()
