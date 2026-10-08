import unittest

import torch

from TFT_model.models import RemoteGridWeatherCrossAttention


class RemoteGridWeatherCrossAttentionTests(unittest.TestCase):
    def test_remote_query_reads_only_the_corresponding_month(self):
        module = RemoteGridWeatherCrossAttention(hidden_size=4, num_heads=2, dropout=0.0).eval()
        remote = torch.zeros(1, 2, 6, 4)
        weather = torch.zeros(1, 2, 168, 4)
        weather[0, 0, :28] = 3.0
        weather[0, 1, :28] = 9.0

        fused, weights = module(remote, weather)

        self.assertEqual(fused.shape, (1, 2, 6, 4))
        self.assertEqual(weights.shape, (1, 2, 6, 28))
        self.assertFalse(torch.allclose(fused[0, 0], fused[0, 1]))
        self.assertTrue(torch.allclose(weights.sum(dim=-1), torch.ones(1, 2, 6)))

    def test_remote_query_isolated_by_grid_and_masked_days_are_finite(self):
        module = RemoteGridWeatherCrossAttention(hidden_size=4, num_heads=2, dropout=0.0).eval()
        remote = torch.zeros(1, 2, 6, 4)
        weather = torch.zeros(1, 2, 168, 4)
        weather_mask = torch.ones(1, 2, 168, dtype=torch.bool)
        weather_mask[:, :, 28:56] = False
        base, _ = module(remote, weather, weather_mask)
        weather[0, 1, :28] = 100.0
        changed, weights = module(remote, weather, weather_mask)

        self.assertTrue(torch.allclose(base[0, 0], changed[0, 0], atol=1e-6))
        self.assertTrue(torch.isfinite(changed).all())
        self.assertTrue(torch.isfinite(weights).all())

    def test_rejects_non_monthly_weather_length(self):
        module = RemoteGridWeatherCrossAttention(hidden_size=4, num_heads=2, dropout=0.0)
        with self.assertRaisesRegex(ValueError, "6 months of 28"):
            module(torch.zeros(1, 1, 6, 4), torch.zeros(1, 1, 28, 4))

    def test_all_weather_mask_returns_zero_fused_tokens_and_weights(self):
        module = RemoteGridWeatherCrossAttention(hidden_size=4, num_heads=2, dropout=0.0).eval()
        fused, weights = module(
            torch.randn(1, 2, 6, 4),
            torch.randn(1, 2, 168, 4),
            torch.zeros(1, 2, 168, dtype=torch.bool),
        )
        self.assertTrue(torch.equal(fused, torch.zeros_like(fused)))
        self.assertTrue(torch.equal(weights, torch.zeros_like(weights)))


if __name__ == "__main__":
    unittest.main()
