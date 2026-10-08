import unittest

import torch

from TFT_model.infer_mmst_frontend import transform_eval_images
from TFT_model.train_mmst_frontend import transform_training_images


class TFTMMSTInputScaleTests(unittest.TestCase):
    def test_uint8_images_match_mmst_pretraining_unit_scale_without_normalize(self):
        images = torch.full((1, 6, 1, 384, 384, 3), 255, dtype=torch.uint8)

        torch.manual_seed(0)
        train_images = transform_training_images(images)
        eval_images = transform_eval_images(images)

        self.assertGreaterEqual(float(train_images.min()), 0.0)
        self.assertLessEqual(float(train_images.max()), 1.0)
        self.assertGreaterEqual(float(eval_images.min()), 0.0)
        self.assertLessEqual(float(eval_images.max()), 1.0)
        self.assertTrue(torch.allclose(eval_images, torch.ones_like(eval_images)))


if __name__ == "__main__":
    unittest.main()
