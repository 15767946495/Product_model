import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).parents[1]
PREPARE_PATH = ROOT / "train_dataset" / "prepare_grid.py"
MODELS_PATH = ROOT / "TFT_model" / "models.py"
DATA_PATH = ROOT / "TFT_model" / "data.py"
sys.path.insert(0, str(PREPARE_PATH.parent))
sys.path.insert(0, str(MODELS_PATH.parent))

prepare_spec = importlib.util.spec_from_file_location("prepare_grid", PREPARE_PATH)
assert prepare_spec is not None and prepare_spec.loader is not None
prepare_grid = importlib.util.module_from_spec(prepare_spec)
prepare_spec.loader.exec_module(prepare_grid)

models_spec = importlib.util.spec_from_file_location("models", MODELS_PATH)
assert models_spec is not None and models_spec.loader is not None
models = importlib.util.module_from_spec(models_spec)
models_spec.loader.exec_module(models)

data_spec = importlib.util.spec_from_file_location("model_data", DATA_PATH)
assert data_spec is not None and data_spec.loader is not None
model_data = importlib.util.module_from_spec(data_spec)
data_spec.loader.exec_module(model_data)


class CenterCoordinateTests(unittest.TestCase):
    def test_uses_grid_boundary_midpoint(self):
        row = {
            "Lat (llcrnr)": "10.0",
            "Lon (llcrnr)": "20.0",
            "Lat (urcrnr)": "10.2",
            "Lon (urcrnr)": "20.4",
        }
        self.assertEqual(prepare_grid.center_coords(row), (10.1, 20.2))

    def test_rejects_old_corner_coordinate_cache(self):
        with tempfile.NamedTemporaryFile(suffix=".pt") as tmp:
            torch.save({"version": 2, "entries": []}, tmp.name)
            with self.assertRaisesRegex(ValueError, "version 3"):
                model_data.load_grid_cache(tmp.name)


class SpatialAttentionContractTests(unittest.TestCase):
    def test_query_only_attention_returns_grid_weights(self):
        aggregator = models.SpatialAttentionAggregator(
            hidden_size=8, dropout=0.0, spatial_encoding="rope"
        )
        tokens = torch.randn(2, 3, 4, 8)
        coords = torch.randn(2, 3, 2)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        output, weights = aggregator.forward_weights(tokens, coords, mask)
        self.assertEqual(output.shape, (2, 4, 8))
        self.assertEqual(weights.shape, (2, 4, 3))
        self.assertTrue(torch.allclose(weights[0, :, 2], torch.zeros(4)))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(2, 4)))

    def test_additive_position_is_spatial_only(self):
        aggregator = models.SpatialAttentionAggregator(
            hidden_size=8, dropout=0.0, spatial_encoding="additive"
        )
        coords = torch.randn(2, 3, 2)
        position = aggregator._spatial_pe(coords)
        self.assertEqual(position.shape, (2, 3, 8))

    def test_temporal_attention_has_no_rope_option(self):
        with self.assertRaises(TypeError):
            models.CausalScaledDotProductAttention(8, num_heads=1, use_rope=True)

    def test_rope_coordinates_affect_initial_attention(self):
        torch.manual_seed(0)
        aggregator = models.SpatialAttentionAggregator(
            hidden_size=8, dropout=0.0, spatial_encoding="rope"
        )
        tokens = torch.randn(1, 3, 2, 8)
        mask = torch.ones(1, 3, dtype=torch.bool)
        coords_a = torch.zeros(1, 3, 2)
        coords_b = torch.tensor([[[10.0, 20.0], [11.0, 21.0], [12.0, 22.0]]])
        _, weights_a = aggregator.forward_weights(tokens, coords_a, mask)
        _, weights_b = aggregator.forward_weights(tokens, coords_b, mask)
        self.assertFalse(torch.allclose(weights_a, weights_b))

    def test_infer_contract_requires_spatial_encoding(self):
        self.assertTrue(hasattr(models, "MODEL_CONTRACT_VERSION"))
        self.assertEqual(models.MODEL_CONTRACT_VERSION, 3)


if __name__ == "__main__":
    unittest.main()
