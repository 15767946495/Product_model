import unittest

import numpy as np

from train_dataset.metrics import regression_metrics


class MetricsTests(unittest.TestCase):
    def test_r2_is_one_minus_sse_over_sst(self):
        result = regression_metrics(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 2.0]))
        self.assertAlmostEqual(result["r2"], 0.5)
        self.assertAlmostEqual(result["rmse"], np.sqrt(1 / 3))
        self.assertAlmostEqual(result["mae"], 1 / 3)
        self.assertEqual(result["count"], 3)

    def test_corr2_and_constant_prediction(self):
        y = np.array([1.0, 2.0, 4.0, 3.0])
        p = np.array([1.5, 2.5, 3.0, 3.5])
        result = regression_metrics(y, p)
        self.assertAlmostEqual(result["corr"], np.corrcoef(y, p)[0, 1])
        self.assertAlmostEqual(result["corr2"], result["corr"] ** 2)
        const = regression_metrics(y, np.full(4, 2.5))
        self.assertEqual(const["corr"], 0.0)
        self.assertEqual(const["corr2"], 0.0)
        self.assertAlmostEqual(const["r2"], 0.0)  # 预测均值时 SSE == SST
        shifted = regression_metrics(y, np.full(4, 10.0))
        self.assertLess(shifted["r2"], 0)  # 标准 R2 可为负，不是 Corr2

    def test_invalid_inputs_rejected(self):
        for y, p in (([1.0, 2.0], [1.0]), ([], []), ([1.0, np.nan], [1.0, 2.0]),
                     ([1.0, 2.0], [1.0, np.inf]), ([2.0, 2.0], [1.0, 2.0])):
            with self.assertRaises(ValueError):
                regression_metrics(y, p)


if __name__ == "__main__":
    unittest.main()
