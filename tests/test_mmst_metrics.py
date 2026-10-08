import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MMST_ROOT = ROOT / "MMST-ViT"
if str(MMST_ROOT) not in sys.path:
    sys.path.insert(0, str(MMST_ROOT))

from util.metrics import R2_Score


class MMSTMetricsTests(unittest.TestCase):
    def test_r2_uses_one_minus_sse_over_sst(self):
        actual = np.asarray([1.0, 2.0, 3.0])
        predicted = np.asarray([1.0, 2.0, 2.0])

        self.assertAlmostEqual(R2_Score(actual, predicted), 0.5)


if __name__ == "__main__":
    unittest.main()
