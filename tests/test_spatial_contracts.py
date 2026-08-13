import importlib.util
import inspect
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
        aggregator = models.SpatialAttentionAggregator(hidden_size=8, dropout=0.0)
        tokens = torch.randn(2, 3, 4, 8)
        coords = torch.randn(2, 3, 2)
        mask = torch.tensor([[True, True, False], [True, True, True]])
        output, weights = aggregator.forward_weights(tokens, coords, mask)
        self.assertEqual(output.shape, (2, 4, 8))
        self.assertEqual(weights.shape, (2, 4, 3))
        self.assertTrue(torch.allclose(weights[0, :, 2], torch.zeros(4)))
        self.assertTrue(torch.allclose(weights.sum(-1), torch.ones(2, 4)))

    def test_weatherformer_four_slot_encoding(self):
        aggregator = models.SpatialAttentionAggregator(hidden_size=8, dropout=0.0)
        coords = torch.tensor([[[10.0, 20.0]]])
        t_idx = torch.tensor([0, 1], dtype=torch.long)
        position = aggregator._st_pe(coords, t_idx)
        freq = 10000.0 ** torch.tensor([0.0, -0.5])
        lat = torch.tensor(10.0 * torch.pi / 180.0)
        lon = torch.tensor(20.0 * torch.pi / 180.0)
        expected = torch.empty(1, 1, 2, 8)
        expected[..., 0::4] = torch.sin(t_idx.float().view(1, 1, 2, 1) * freq)
        expected[..., 1::4] = torch.cos(t_idx.float().view(1, 1, 2, 1) * freq)
        expected[..., 2::4] = torch.sin(lat * freq)
        expected[..., 3::4] = torch.cos(lon * freq)
        self.assertTrue(torch.allclose(position, expected, atol=1e-6))

    def test_hidden_size_must_be_divisible_by_four(self):
        with self.assertRaisesRegex(ValueError, "4"):
            models.SpatialAttentionAggregator(hidden_size=6, dropout=0.0)

    def test_cls_coordinate_is_masked_grid_center_mean(self):
        aggregator = models.SpatialAttentionAggregator(hidden_size=8, dropout=0.0)
        coords = torch.tensor([[[10.0, 20.0], [14.0, 24.0], [100.0, 200.0]]])
        mask = torch.tensor([[True, True, False]])
        cls_coords = aggregator._cls_coords(coords, mask)
        self.assertTrue(torch.allclose(cls_coords, torch.tensor([[12.0, 22.0]])))

    def test_time_changes_cls_and_grid_position(self):
        aggregator = models.SpatialAttentionAggregator(hidden_size=8, dropout=0.0)
        grid_coords = torch.tensor([[[10.0, 20.0], [12.0, 22.0]]])
        cls_coords = torch.tensor([[[11.0, 21.0]]])
        t_idx = torch.tensor([0, 1], dtype=torch.long)
        grid_pe = aggregator._st_pe(grid_coords, t_idx)
        cls_pe = aggregator._st_pe(cls_coords, t_idx)
        self.assertFalse(torch.allclose(grid_pe[:, :, 0], grid_pe[:, :, 1]))
        self.assertFalse(torch.allclose(cls_pe[:, :, 0], cls_pe[:, :, 1]))

    def test_value_path_does_not_include_position(self):
        aggregator = models.SpatialAttentionAggregator(hidden_size=8, dropout=0.0)
        with torch.no_grad():
            aggregator.W_q.weight.zero_()
            aggregator.W_q.bias.zero_()
            aggregator.W_k.weight.zero_()
            aggregator.W_k.bias.zero_()
            aggregator.W_v.weight.copy_(torch.eye(8))
            aggregator.W_v.bias.zero_()
        tokens = torch.randn(1, 2, 3, 8)
        mask = torch.ones(1, 2, dtype=torch.bool)
        coords_a = torch.tensor([[[10.0, 20.0], [11.0, 21.0]]])
        coords_b = torch.tensor([[[30.0, 40.0], [31.0, 41.0]]])
        output_a, _ = aggregator.forward_weights(tokens, coords_a, mask)
        output_b, _ = aggregator.forward_weights(tokens, coords_b, mask)
        self.assertTrue(torch.allclose(output_a, output_b, atol=1e-6))

    def test_temporal_attention_has_no_rope_option(self):
        with self.assertRaises(TypeError):
            models.CausalScaledDotProductAttention(8, num_heads=1, use_rope=True)

    def test_model_contract_version_is_six(self):
        self.assertTrue(hasattr(models, "MODEL_CONTRACT_VERSION"))
        self.assertEqual(models.MODEL_CONTRACT_VERSION, 6)

    def test_model_has_no_spatial_encoding_parameter(self):
        signature = inspect.signature(models.TFTEncoderForYieldPrediction)
        self.assertNotIn("spatial_encoding", signature.parameters)

    def test_attention_model_uses_weatherformer_aggregator(self):
        model = models.TFTEncoderForYieldPrediction(
            soil_dim=7,
            dynamic_feature_names=["temperature"],
            hidden_size=8,
            num_lstm_layers=1,
            dropout=0.0,
            output_size=1,
            num_heads=1,
            spatial_mode="attention",
        )
        self.assertIsInstance(model.spatial_agg, models.SpatialAttentionAggregator)

    def test_model_uses_grid_local_vsn_before_spatial_pooling(self):
        model = models.TFTEncoderForYieldPrediction(
            soil_dim=7,
            dynamic_feature_names=["temperature", "precipitation"],
            hidden_size=8,
            num_lstm_layers=1,
            dropout=0.0,
            output_size=1,
            num_heads=1,
            spatial_mode="attention",
        )
        self.assertIsInstance(model.grid_vsn, models.VariableSelectionNetwork)
        self.assertFalse(hasattr(model, "temporal_vsn"))

    def test_grid_vsn_context_changes_grid_token_selection(self):
        torch.manual_seed(0)
        model = models.TFTEncoderForYieldPrediction(
            soil_dim=7,
            dynamic_feature_names=["temperature", "precipitation"],
            hidden_size=8,
            num_lstm_layers=1,
            dropout=0.0,
            output_size=1,
            num_heads=1,
            spatial_mode="mean",
        ).eval()
        grid_feats = torch.randn(1, 2, 3, 2)
        coords = torch.randn(1, 2, 2)
        mask = torch.ones(1, 2, dtype=torch.bool)
        soil = torch.randn(1, 7)
        seq_lens = torch.tensor([3])
        with torch.no_grad():
            _, _, aux_a = model(grid_feats, coords, mask, soil, seq_lens)
            _, _, aux_b = model(grid_feats, coords, mask, soil + 2.0, seq_lens)
        self.assertFalse(torch.allclose(aux_a["grid_vsn_weights"], aux_b["grid_vsn_weights"]))

    def test_county_vsn_is_available_after_featurewise_spatial_pooling(self):
        model = models.TFTEncoderForYieldPrediction(
            soil_dim=7,
            dynamic_feature_names=["temperature", "precipitation"],
            hidden_size=8,
            num_lstm_layers=1,
            dropout=0.0,
            output_size=1,
            num_heads=1,
            spatial_mode="attention",
            variable_selection_stage="county",
        )
        self.assertIsNone(getattr(model, "grid_vsn", None))
        self.assertIsInstance(model.county_vsn, models.VariableSelectionNetwork)

    def test_county_vsn_weights_change_with_static_context(self):
        torch.manual_seed(0)
        model = models.TFTEncoderForYieldPrediction(
            soil_dim=7,
            dynamic_feature_names=["temperature", "precipitation"],
            hidden_size=8,
            num_lstm_layers=1,
            dropout=0.0,
            output_size=1,
            num_heads=1,
            spatial_mode="mean",
            variable_selection_stage="county",
        ).eval()
        grid_feats = torch.randn(1, 2, 3, 2)
        coords = torch.randn(1, 2, 2)
        mask = torch.ones(1, 2, dtype=torch.bool)
        seq_lens = torch.tensor([3])
        with torch.no_grad():
            _, _, aux_a = model(grid_feats, coords, mask, torch.randn(1, 7), seq_lens)
            _, _, aux_b = model(grid_feats, coords, mask, torch.randn(1, 7) + 2.0, seq_lens)
        self.assertFalse(torch.allclose(aux_a["county_vsn_weights"], aux_b["county_vsn_weights"]))


if __name__ == "__main__":
    unittest.main()
