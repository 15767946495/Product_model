import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "MMST-ViT" / "main_finetune_mmst_vit.py"


class MMSTFinetuneFreezeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SOURCE_PATH.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)

    def test_validation_loaders_use_validation_samplers(self):
        self.assertIn("dataset_sentinel_val, sampler=sampler_sentinel_val", self.source)
        self.assertIn("dataset_hrrr_val, sampler=sampler_hrrr_val", self.source)
        self.assertIn("dataset_usda_val, sampler=sampler_usda_val", self.source)
        self.assertIn("shuffle=False", self.source)

    def test_eval_mode_uses_validation_loaders(self):
        eval_block = self.source[self.source.index("if args.eval:"):self.source.index("print(f\"Start training", self.source.index("if args.eval:"))]
        self.assertIn("data_loader_sentinel_val", eval_block)
        self.assertIn("data_loader_hrrr_val", eval_block)
        self.assertIn("data_loader_usda_val", eval_block)
        self.assertNotIn("data_loader_sentinel_train", eval_block)
        self.assertIn("model.eval()", self.source[self.source.index("def evaluate("):])

    def test_pretrained_pvt_and_mm_mha_are_finetuned_end_to_end(self):
        self.assertNotIn("for parameter in pvt.parameters():", self.source)
        self.assertNotIn("parameter.requires_grad = False", self.source)
        self.assertNotIn("pvt_backbone.eval()", self.source)

    def test_finetune_supervises_only_yield_output(self):
        self.assertIn("MMST_ViT(out_dim=1", self.source)
        self.assertIn("z = z[0][..., -1:]", self.source)


if __name__ == "__main__":
    unittest.main()
