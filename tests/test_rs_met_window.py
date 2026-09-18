import unittest

import torch

from TFT_model.models import RS_WINDOW_SIZE, _build_met_windows


class RemoteSensingMeteorologyWindowTest(unittest.TestCase):
    def test_window_includes_rs_day_and_uses_following_14_days(self):
        temporal_feat = torch.arange(20, dtype=torch.float32).view(1, 20, 1)
        seq_lens = torch.tensor([20])
        day_indices = torch.tensor([0, 2])

        windows, window_mask = _build_met_windows(
            temporal_feat, seq_lens, day_indices, RS_WINDOW_SIZE
        )

        self.assertEqual(windows.shape, (1, 2, 14, 1))
        self.assertEqual(window_mask.shape, (1, 2, 14))
        self.assertTrue(torch.equal(windows[0, 0, :, 0], torch.arange(14.0)))
        self.assertTrue(torch.equal(windows[0, 1, :, 0], torch.arange(2.0, 16.0)))
        self.assertTrue(window_mask[0].all())

    def test_window_is_truncated_by_sequence_length(self):
        temporal_feat = torch.arange(20, dtype=torch.float32).view(1, 20, 1)
        seq_lens = torch.tensor([10])
        day_indices = torch.tensor([8])

        windows, window_mask = _build_met_windows(
            temporal_feat, seq_lens, day_indices, RS_WINDOW_SIZE
        )

        self.assertTrue(torch.equal(windows[0, 0, :2, 0], torch.tensor([8.0, 9.0])))
        self.assertTrue(window_mask[0, 0, :2].all())
        self.assertFalse(window_mask[0, 0, 2:].any())


if __name__ == "__main__":
    unittest.main()
