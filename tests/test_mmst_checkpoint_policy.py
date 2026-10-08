import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MMST_ROOT = ROOT / "MMST-ViT"


class MMSTCheckpointPolicyTests(unittest.TestCase):
    def test_save_model_uses_fixed_latest_checkpoint(self):
        source = (MMST_ROOT / "util" / "misc.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "save_model"
        )
        text = ast.get_source_segment(source, function)
        self.assertIn("checkpoint-latest.pth", text)
        self.assertNotIn("checkpoint-%s.pth", text)

    def test_finetune_launcher_uses_single_gpu_and_latest_pretrain_checkpoint(self):
        source = (MMST_ROOT / "run_finetune.sh").read_text(encoding="utf-8")
        self.assertIn("main_finetune_mmst_vit.py", source)
        self.assertIn("--pvt_simclr ./output_dir/pvt_simclr/checkpoint-latest.pth", source)
        self.assertIn("--device cuda:0", source)
        self.assertNotIn("torch.distributed.run", source)


if __name__ == "__main__":
    unittest.main()
