"""真实 RAID 数据单县 smoke test；数据未准备时显式跳过。

  export CROPNET_SMOKE_FIPS=<五位FIPS>
  python train_dataset/build_dataset.py --only-fips "$CROPNET_SMOKE_FIPS" --output /tmp/opencode/cropnet-smoke.jsonl
"""

import os
import unittest
from pathlib import Path

import torch

from train_dataset.dataset import DATA_ROOT, YEARS, CropNetDataset, check_dataset_file, load_jsonl

FIPS = os.environ.get("CROPNET_SMOKE_FIPS", "")
JSONL = Path(os.environ.get("CROPNET_SMOKE_JSONL", "/tmp/opencode/cropnet-smoke.jsonl"))


@unittest.skipUnless(DATA_ROOT.is_dir(), f"{DATA_ROOT} not present")
@unittest.skipUnless(FIPS, "CROPNET_SMOKE_FIPS not set")
class RealCropNetSmokeTests(unittest.TestCase):
    def setUp(self):
        if not JSONL.is_file():
            self.skipTest(f"{JSONL} not built yet")

    def test_smoke_jsonl_has_six_years(self):
        rows = load_jsonl(JSONL)
        self.assertEqual([r["year"] for r in rows], list(YEARS))
        self.assertEqual({r["fips"] for r in rows}, {FIPS})
        self.assertGreater(len(rows[0]["grids"]), 0)
        check_dataset_file(JSONL, check_files=True, data_root=DATA_ROOT)

    def test_loads_one_real_image(self):
        ds = CropNetDataset(JSONL, (2021,), load_images=True, data_root=DATA_ROOT)
        item = ds[0]
        images = item["images"]
        self.assertEqual(images.shape, (6, item["weather"].shape[0], 3, 224, 224))
        self.assertEqual(images.dtype, torch.float32)
        self.assertTrue(torch.isfinite(images).all())
        self.assertGreaterEqual(images.min().item(), 0.0)
        self.assertLessEqual(images.max().item(), 1.0)


if __name__ == "__main__":
    unittest.main()
