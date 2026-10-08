import importlib.util
import inspect
import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import unittest
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
MMST_ROOT = ROOT / "MMST-ViT"
DATA_ROOT = Path("/data/raid0/hqx/Product_model_runtime/DataSrc")
SOIL_FILE = DATA_ROOT / "soil/county_soil.json"
SOIL_FEATURES = [
    "clay_pct", "sand_pct", "silt_pct", "om_pct", "ph",
    "bulk_density", "awc",
]


def load_official_class(module_name, relative_path, class_name):
    spec = importlib.util.spec_from_file_location(
        module_name, MMST_ROOT / relative_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载测试官方模块: {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


Sentinel_Dataset = load_official_class(
    "test_mmst_official_sentinel", "dataset/sentinel_loader.py", "Sentinel_Dataset",
)
HRRR_Dataset = load_official_class(
    "test_mmst_official_hrrr", "dataset/hrrr_loader.py", "HRRR_Dataset",
)
USDA_Dataset = load_official_class(
    "test_mmst_official_usda", "dataset/usda_loader.py", "USDA_Dataset",
)


class MMSTDataAdapterTests(unittest.TestCase):
    def setUp(self):
        self.warning_context = warnings.catch_warnings()
        self.warning_context.__enter__()
        warnings.filterwarnings(
            "ignore",
            message=r"unclosed file <_io.TextIOWrapper name='.*config\.json'.*",
            category=ResourceWarning,
        )
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.fips = "17001"
        self.year = 2021
        self._write_fixture()

    def tearDown(self):
        self.directory.cleanup()
        self.warning_context.__exit__(None, None, None)

    def _write_fixture(self):
        sentinel_paths = []
        for quarter, dates in enumerate((("04-01", "04-15", "05-01", "05-15", "06-01", "06-15"),
                                         ("07-01", "07-15", "08-01", "08-15", "09-01", "09-15"))):
            path = self.root / f"sentinel-{quarter}.h5"
            sentinel_paths.append(path.name)
            with h5py.File(path, "w") as handle:
                county = handle.create_group(self.fips)
                for offset, date in enumerate(dates):
                    group = county.create_group(date)
                    group.create_dataset(
                        "data",
                        data=np.full((2, 2, 2, 3), quarter * 20 + offset, dtype=np.uint8),
                    )

        weather_columns = [
            "Avg Temperature (K)", "Max Temperature (K)", "Min Temperature (K)",
            "Precipitation (kg m**-2)", "Relative Humidity (%)", "Wind Gust (m s**-1)",
            "Wind Speed (m s**-1)", "Downward Shortwave Radiation Flux (W m**-2)",
            "Vapor Pressure Deficit (kPa)",
        ]
        weather_paths = []
        for month in range(4, 10):
            path = self.root / f"weather-{month}.csv"
            weather_paths.append(path.name)
            rows = []
            for grid in range(2):
                for day in range(1, 29):
                    row = {
                        "FIPS Code": self.fips,
                        "Daily/Monthly": "Daily",
                        "Month": month,
                        "Grid Index": grid,
                        "Day": day,
                    }
                    row.update({name: float(month * 100 + grid * 10 + day + index)
                                for index, name in enumerate(weather_columns)})
                    rows.append(row)
                monthly = {
                    "FIPS Code": self.fips,
                    "Daily/Monthly": "Monthly",
                    "Month": month,
                    "Grid Index": grid,
                    "Day": 1,
                }
                monthly.update({name: float(month + index) for index, name in enumerate(weather_columns)})
                rows.append(monthly)
            pd.DataFrame(rows).to_csv(path, index=False)

        usda_path = self.root / "USDA Crop Dataset/data/Corn/2021/usda.csv"
        usda_path.parent.mkdir(parents=True)
        pd.DataFrame([{
            "state_ansi": 17,
            "county_ansi": 1,
            "PRODUCTION, MEASURED IN BU": 1000.0,
            "YIELD, MEASURED IN BU / ACRE": 200.0,
        }]).to_csv(usda_path, index=False)

        record = {
            "FIPS": self.fips,
            "year": self.year,
            "state_ansi": "17",
            "county_ansi": "001",
            "data": {
                "sentinel": sentinel_paths,
                "HRRR": {
                    "short_term": weather_paths,
                    "long_term": [weather_paths],
                },
                "USDA": usda_path.relative_to(self.root).as_posix(),
            },
        }
        self.config = self.root / "config.json"
        self.config.write_text(json.dumps([record]), encoding="utf-8")
        self.soil = self.root / "soil.json"
        self.soil.write_text(json.dumps({
            self.fips: {name: float(index + 1) for index, name in enumerate(SOIL_FEATURES)},
        }), encoding="utf-8")

    def test_adapter_matches_direct_official_outputs_exactly(self):
        from TFT_model.mmst_data import MMSTCornDataset

        direct_images, image_fips, image_year = Sentinel_Dataset(self.root, self.config)[0]
        direct_weather = HRRR_Dataset(self.root, self.config, num_workers=1)
        self.addCleanup(direct_weather.executor.shutdown, wait=True)
        direct_short, _, weather_fips, weather_year = direct_weather[0]
        direct_target, target_fips, target_year = USDA_Dataset(
            self.root, self.config, crop_type="Corn",
        )[0]

        dataset = MMSTCornDataset(self.root, self.config, self.soil)
        self.addCleanup(dataset.close)
        sample = dataset[0]

        self.assertEqual(len(sample), 6)
        images, short_weather, yield_log, soil, fips, year = sample
        self.assertTrue(torch.equal(images, direct_images))
        self.assertTrue(torch.equal(short_weather, direct_short))
        self.assertTrue(torch.equal(yield_log, direct_target[-1]))
        self.assertTrue(torch.equal(soil, torch.arange(1, 8, dtype=torch.float32)))
        self.assertEqual((fips, year), (image_fips, image_year))
        self.assertEqual((fips, year), (weather_fips, weather_year))
        self.assertEqual((fips, year), (target_fips, target_year))

    def test_adapter_does_not_access_long_weather(self):
        from TFT_model.mmst_data import MMSTCornDataset

        dataset = MMSTCornDataset(self.root, self.config, self.soil)
        self.addCleanup(dataset.close)
        dataset.weather_dataset.get_long_term_val = lambda *args: self.fail("访问了 long weather")
        sample = dataset[0]
        self.assertEqual(len(sample), 6)

    def test_alignment_mismatch_fails_before_data_access(self):
        from TFT_model.mmst_data import MMSTCornDataset

        dataset = MMSTCornDataset(self.root, self.config, self.soil)
        self.addCleanup(dataset.close)
        dataset.weather_dataset.years[0] = self.year + 1
        with self.assertRaisesRegex(ValueError, "alignment"):
            dataset.validate_alignment()

    def test_missing_or_incomplete_soil_fails_for_config_fips(self):
        from TFT_model.mmst_data import MMSTCornDataset

        for payload in ({}, {self.fips: {name: 1.0 for name in SOIL_FEATURES[:-1]}}):
            self.soil.write_text(json.dumps(payload), encoding="utf-8")
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "soil"):
                MMSTCornDataset(self.root, self.config, self.soil)

    def test_rejects_non_corn_usda_path_before_official_loaders_are_created(self):
        from TFT_model.mmst_data import MMSTCornDataset

        records = json.loads(self.config.read_text(encoding="utf-8"))
        records[0]["data"]["USDA"] = (
            "USDA Crop Dataset/data/Soybean/2021/USDA_Soybean_County_2021.csv"
        )
        self.config.write_text(json.dumps(records), encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Corn"):
            MMSTCornDataset(self.root, self.config, self.soil)

    def test_close_and_context_manager_shutdown_hrrr_executor_idempotently(self):
        from TFT_model.mmst_data import MMSTCornDataset

        dataset = MMSTCornDataset(self.root, self.config, self.soil)
        executor = dataset.weather_dataset.executor
        dataset.close()
        dataset.close()
        with self.assertRaises(RuntimeError):
            executor.submit(lambda: None)

        with MMSTCornDataset(self.root, self.config, self.soil) as managed:
            managed_executor = managed.weather_dataset.executor
            self.assertIsInstance(managed_executor, ThreadPoolExecutor)
        with self.assertRaises(RuntimeError):
            managed_executor.submit(lambda: None)

    def test_county_loader_forces_outer_batch_size_one(self):
        from TFT_model.mmst_data import MMSTCornDataset, MMSTCountyLoader

        dataset = MMSTCornDataset(self.root, self.config, self.soil)
        self.addCleanup(dataset.close)
        loader = MMSTCountyLoader(dataset, shuffle=False, num_workers=0)
        self.assertEqual(loader.batch_size, 1)
        self.assertEqual(next(iter(loader))[0].shape[0], 1)
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            MMSTCountyLoader(dataset, batch_size=2)

    def test_import_isolation_does_not_pollute_models_or_dataset_modules(self):
        before = {name: sys.modules.get(name) for name in ("models", "dataset")}
        from TFT_model import mmst_data

        self.assertIs(mmst_data.Sentinel_Dataset, mmst_data._OFFICIAL_SENTINEL_MODULE.Sentinel_Dataset)
        self.assertIs(mmst_data.HRRR_Dataset, mmst_data._OFFICIAL_HRRR_MODULE.HRRR_Dataset)
        self.assertIs(mmst_data.USDA_Dataset, mmst_data._OFFICIAL_USDA_MODULE.USDA_Dataset)
        self.assertEqual({name: sys.modules.get(name) for name in before}, before)


class MMSTSpatialFrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from TFT_model import mmst_frontend

        cls.frontend_module = mmst_frontend
        cls.PVTSimCLR = mmst_frontend.PVTSimCLR
        cls.SpatialTransformer = mmst_frontend.SpatialTransformer

    def _checkpoint(self, directory, *, remove_key=None, include_spatial=False):
        source = self.PVTSimCLR("pvt_tiny", out_dim=512, context_dim=9, pretrained=False)
        state = source.state_dict()
        if remove_key is not None:
            state.pop(remove_key)
        if include_spatial:
            state["space_token"] = torch.randn(1, 1, 512)
        path = Path(directory) / "checkpoint.pth"
        torch.save({"model": state}, path)
        return path, source

    def _write_state(self, directory, state):
        path = Path(directory) / "checkpoint.pth"
        torch.save({"model": state}, path)
        return path

    def test_uses_the_official_pvt_and_spatial_transformer_classes(self):
        self.assertIs(
            self.frontend_module.PVTSimCLR,
            self.frontend_module._OFFICIAL_SIMCLR_MODULE.PVTSimCLR,
        )
        self.assertIs(
            self.frontend_module.SpatialTransformer,
            self.frontend_module._OFFICIAL_ATTENTION_MODULE.SpatialTransformer,
        )

    def test_official_top_level_import_aliases_do_not_leak(self):
        for name in ("models_pvt", "models_pvt_simclr", "attention"):
            self.assertNotIn(name, sys.modules)

    def test_strictly_loads_complete_pvt_simclr_state_and_ignores_spatial_keys(self):
        from TFT_model.mmst_frontend import MMSTSpatialFrontend

        with tempfile.TemporaryDirectory() as directory:
            path, source = self._checkpoint(directory, include_spatial=True)
            frontend = MMSTSpatialFrontend(path)
            for name, parameter in source.state_dict().items():
                self.assertTrue(torch.equal(parameter, frontend.pvt_backbone.state_dict()[name]))
            self.assertEqual(len(frontend.loaded_pvt_keys), 195)
            self.assertEqual(frontend.ignored_spatial_keys, ["space_token"])
            self.assertEqual(frontend.ignored_unknown_keys, [])
            self.assertTrue(all(
                not parameter.requires_grad
                for parameter in frontend.pvt_backbone.parameters()
            ))
            self.assertFalse(frontend.pvt_backbone.training)

    def test_accepts_all_module_prefix_and_rejects_mixed_or_duplicate_prefixes(self):
        from TFT_model.mmst_frontend import MMSTSpatialFrontend

        with tempfile.TemporaryDirectory() as directory:
            source = self.PVTSimCLR("pvt_tiny", out_dim=512, context_dim=9, pretrained=False)
            state = {"module." + name: value for name, value in source.state_dict().items()}
            frontend = MMSTSpatialFrontend(self._write_state(directory, state))
            self.assertEqual(len(frontend.loaded_pvt_keys), 195)

            mixed = dict(state)
            first = next(iter(source.state_dict()))
            mixed[first] = source.state_dict()[first]
            with self.assertRaisesRegex(ValueError, "mixed"):
                MMSTSpatialFrontend(self._write_state(directory, mixed))

            duplicate = dict(state)
            duplicate["module." + first] = source.state_dict()[first]
            duplicate[first] = source.state_dict()[first]
            with self.assertRaisesRegex(ValueError, "duplicate"):
                MMSTSpatialFrontend(self._write_state(directory, duplicate))

    def test_records_unknown_and_official_spatial_extra_keys(self):
        from TFT_model.mmst_frontend import MMSTSpatialFrontend

        with tempfile.TemporaryDirectory() as directory:
            source = self.PVTSimCLR("pvt_tiny", out_dim=512, context_dim=9, pretrained=False)
            state = source.state_dict()
            state["pos_embedding"] = torch.randn(1, 6, 64, 512)
            state["space_transformer.norm.weight"] = torch.randn(512)
            state["unexpected.extra"] = torch.randn(1)
            frontend = MMSTSpatialFrontend(self._write_state(directory, state))
            self.assertEqual(frontend.ignored_spatial_keys, [
                "pos_embedding", "space_transformer.norm.weight",
            ])
            self.assertEqual(frontend.ignored_unknown_keys, ["unexpected.extra"])

    def test_missing_any_pvt_simclr_key_fails(self):
        from TFT_model.mmst_frontend import MMSTSpatialFrontend

        with tempfile.TemporaryDirectory() as directory:
            source = self.PVTSimCLR("pvt_tiny", out_dim=512, context_dim=9, pretrained=False)
            missing = next(iter(source.state_dict()))
            path, _ = self._checkpoint(directory, remove_key=missing)
            with self.assertRaisesRegex(RuntimeError, "missing"):
                MMSTSpatialFrontend(path)

    def test_forward_accepts_only_batched_monthly_chw_images(self):
        from TFT_model.mmst_frontend import MMSTSpatialFrontend

        with tempfile.TemporaryDirectory() as directory:
            path, _ = self._checkpoint(directory)
            frontend = MMSTSpatialFrontend(path, batch_size=2).eval()
            images = torch.randn(1, 6, 2, 3, 224, 224)
            weather = torch.randn(1, 6, 2, 28, 9)
            with torch.no_grad():
                output = frontend(images, weather)
            self.assertEqual(output.shape, (1, 6, 512))
            with self.assertRaisesRegex(ValueError, "CHW"):
                frontend(images.permute(0, 1, 2, 4, 5, 3), weather)
            with self.assertRaisesRegex(ValueError, "28, 9"):
                frontend(images, torch.randn(1, 6, 2, 27, 9))
            with self.assertRaisesRegex(ValueError, "224"):
                frontend(torch.randn(1, 6, 2, 3, 223, 224), weather)

    def test_grid_capacity_includes_spatial_cls_token(self):
        from TFT_model.mmst_frontend import MMSTSpatialFrontend

        with tempfile.TemporaryDirectory() as directory:
            path, _ = self._checkpoint(directory)
            frontend = MMSTSpatialFrontend(path)
            self.assertEqual(tuple(frontend.pos_embedding.shape), (1, 6, 64, 512))
            images = torch.empty(1, 6, 64, 3, 224, 224)
            weather = torch.empty(1, 6, 64, 28, 9)
            with self.assertRaisesRegex(ValueError, "num_grid=64.*63"):
                frontend(images, weather)

    def test_batch_size_one_tail_is_duplicated_and_matches_direct_official_call(self):
        from TFT_model.mmst_frontend import MMSTSpatialFrontend

        with tempfile.TemporaryDirectory() as directory:
            path, _ = self._checkpoint(directory)
            frontend = MMSTSpatialFrontend(path, batch_size=5).eval()
            images = torch.randn(1, 6, 1, 3, 224, 224)
            weather = torch.randn(1, 6, 1, 28, 9)
            with torch.no_grad():
                actual = frontend.forward_features(images, weather)
                direct = frontend.pvt_backbone(
                    torch.cat((images.reshape(6, 3, 224, 224)[-1:],) * 2),
                    context=torch.cat((weather.reshape(6, 28, 9)[-1:],) * 2),
                )[:1]
                direct = torch.cat((
                    frontend.pvt_backbone(
                        images.reshape(6, 3, 224, 224)[:5],
                        context=weather.reshape(6, 28, 9)[:5],
                    ),
                    direct,
                ))
            torch.testing.assert_close(actual, direct, rtol=1e-5, atol=1e-5)
            self.assertEqual(actual.shape, (6, 512))

    def test_batch_size_one_duplicates_each_singleton_chunk(self):
        from TFT_model.mmst_frontend import MMSTSpatialFrontend

        with tempfile.TemporaryDirectory() as directory:
            path, _ = self._checkpoint(directory)
            frontend = MMSTSpatialFrontend(path, batch_size=1).eval()
            images = torch.randn(1, 6, 1, 3, 224, 224)
            weather = torch.randn(1, 6, 1, 28, 9)
            with torch.no_grad():
                actual = frontend.forward_features(images, weather)
                flat_images = images.reshape(6, 3, 224, 224)
                flat_weather = weather.reshape(6, 28, 9)
                expected = torch.cat([
                    frontend.pvt_backbone(
                        torch.cat((image.unsqueeze(0), image.unsqueeze(0))),
                        context=torch.cat((context.unsqueeze(0), context.unsqueeze(0))),
                    )[:1]
                    for image, context in zip(flat_images, flat_weather)
                ])
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
            self.assertEqual(actual.shape, (6, 512))

    def test_concurrent_imports_reuse_stable_official_classes(self):
        import importlib

        def load():
            module = importlib.import_module("TFT_model.mmst_frontend")
            return module._load_official_frontend_classes()[:2]

        with ThreadPoolExecutor(max_workers=8) as pool:
            classes = list(pool.map(lambda _: load(), range(8)))
        self.assertTrue(all(pair == classes[0] for pair in classes))
        for name in ("models_pvt", "models_pvt_simclr", "attention"):
            self.assertNotIn(name, sys.modules)

    def test_spatial_output_matches_official_mmst_path(self):
        from einops import rearrange, repeat
        from TFT_model.mmst_frontend import MMSTSpatialFrontend

        with tempfile.TemporaryDirectory() as directory:
            path, source = self._checkpoint(directory)
            frontend = MMSTSpatialFrontend(path, batch_size=2).eval()
            official = self.frontend_module.MMST_ViT(
                pvt_backbone=source,
                dim=512,
                num_grid=3,
                num_short_term_seq=6,
                batch_size=2,
            ).eval()
            official.space_token.data.copy_(frontend.space_token.data)
            official.pos_embedding.data.copy_(frontend.pos_embedding.data[:, :, :3])
            official.space_transformer.load_state_dict(frontend.space_transformer.state_dict())
            images = torch.randn(1, 6, 2, 3, 224, 224)
            weather = torch.randn(1, 6, 2, 28, 9)
            with torch.no_grad():
                expected = official.forward_features(images, weather)
                expected = rearrange(expected, "(b t g) d -> b t g d", b=1, t=6, g=2)
                expected = torch.cat((repeat(official.space_token, "() g d -> b t g d", b=1, t=6), expected), dim=2)
                expected = expected + official.pos_embedding[:, :, :3]
                expected = official.dropout(expected)
                expected = rearrange(expected, "b t g d -> (b t) g d")
                expected = official.space_transformer(expected)
                expected = rearrange(expected[:, 0], "(b t) ... -> b t ...", b=1)
                actual = frontend(images, weather)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _CheapSpatialFrontend(nn.Module):
    def __init__(self, checkpoint_path, dim=512, num_grid=64, batch_size=64):
        super().__init__()
        self.projection = nn.Linear(1, dim, bias=False)

    def forward(self, images, short_weather):
        image_token = images.mean(dim=(2, 3, 4, 5), keepdim=False).unsqueeze(-1)
        weather_token = short_weather.mean(dim=(2, 3, 4), keepdim=False).unsqueeze(-1)
        return self.projection(image_token + weather_token)


class MMSTTFTYieldModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.frontend_patch = patch(
            "TFT_model.mmst_tft_model.MMSTSpatialFrontend",
            _CheapSpatialFrontend,
        )
        self.frontend_patch.start()
        self.addCleanup(self.frontend_patch.stop)

        from TFT_model.mmst_tft_model import MMSTTFTYieldModel

        self.model = MMSTTFTYieldModel(
            "unused-checkpoint.pth", hidden_size=8, heads=2, dropout=0.0,
        ).eval()
        self.images = torch.randn(2, 6, 2, 3, 4, 4)
        self.weather = torch.randn(2, 6, 2, 28, 9)
        self.soil = torch.randn(2, 7)

    def test_reuses_tft_components_and_excludes_legacy_frontend_modules(self):
        from TFT_model.mmst_tft_model import MMSTTFTYieldModel
        from TFT_model.models import (
            CausalScaledDotProductAttention,
            GatedResidualNetwork,
            LSTMEncoder,
            SoilStaticEncoder,
        )

        self.assertIsInstance(self.model.soil_static_encoder, SoilStaticEncoder)
        self.assertIsInstance(self.model.lstm_encoder, LSTMEncoder)
        self.assertIsInstance(self.model.temporal_grn, GatedResidualNetwork)
        self.assertIsInstance(self.model.attention, CausalScaledDotProductAttention)
        self.assertIsInstance(self.model.pred_grn, GatedResidualNetwork)
        forbidden = {
            "per_feature_linear", "grid_vsn", "weather_remote_attn", "spatial_agg",
        }
        self.assertTrue(forbidden.isdisjoint(dict(self.model.named_modules())))
        signature = inspect.signature(MMSTTFTYieldModel.forward)
        self.assertEqual(
            list(signature.parameters),
            ["self", "images", "short_weather", "soil", "month_mask"],
        )
        self.assertNotIn("long", str(signature).lower())

    def test_soil_never_enters_frontend_but_changes_final_distribution(self):
        with torch.no_grad():
            first, _, first_aux = self.model(self.images, self.weather, self.soil)
            second, _, second_aux = self.model(
                self.images, self.weather, self.soil.flip(0),
            )

        torch.testing.assert_close(
            first_aux["frontend_tokens"], second_aux["frontend_tokens"], rtol=0, atol=0,
        )
        self.assertFalse(torch.allclose(first, second))
        self.assertEqual(first_aux["frontend_tokens"].shape, (2, 6, 512))
        self.assertEqual(first.shape, (2, 1))
        self.assertEqual(first_aux["pred_dist_all"].shape, (2, 6, 1))
        self.assertTrue(hasattr(self.model, "yield_head"))
        self.assertFalse(hasattr(self.model, "variance_head"))

    def test_soil_conditions_single_remote_variable_before_lstm(self):
        self.assertTrue(hasattr(self.model, "remote_variable_grn"))
        self.assertEqual(self.model.remote_variable_grn.context_size, 8)

    def test_temporal_decoder_restores_tft_gate_add_norm_stages(self):
        self.assertTrue(hasattr(self.model, "attention_gate"))
        self.assertFalse(hasattr(self.model, "output_gate"))

    def test_grn_residual_flag_controls_skip_connection(self):
        from TFT_model.models import GatedResidualNetwork

        residual = GatedResidualNetwork(4, 4, 4, dropout=0.0, residual=True).eval()
        no_residual = GatedResidualNetwork(4, 4, 4, dropout=0.0, residual=False).eval()
        no_residual.load_state_dict(residual.state_dict())
        inputs = torch.randn(2, 3, 4)
        skip = torch.randn(2, 3, 4)

        with torch.no_grad():
            residual_output = residual(inputs, skip=skip)
            no_residual_output = no_residual(inputs, skip=skip)

        self.assertFalse(torch.allclose(residual_output, no_residual_output))

    def test_glu_applies_dropout_before_linear_gate(self):
        from TFT_model.models import GatedLinearUnit

        module = GatedLinearUnit(4, hidden_size=4, dropout=0.1)
        self.assertIsNotNone(module.dropout)

    def test_formal_tft_dropout_default_is_zero(self):
        source = (ROOT / "TFT_model" / "train_mmst_frontend.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('parser.add_argument("--dropout", type=float, default=0.0)', source)

    def test_soil_static_encoder_uses_seven_variable_selection_weights(self):
        from TFT_model.models import SoilStaticEncoder

        encoder = SoilStaticEncoder(soil_dim=7, hidden_size=8, dropout=0.0)
        self.assertEqual(len(encoder.variable_grns), 7)
        self.assertEqual(encoder.selection_grn.output_size, 7)
        features = torch.randn(2, 7)
        _, _, _, _, weights = encoder(features, return_weights=True)
        self.assertEqual(weights.shape, (2, 7))
        torch.testing.assert_close(
            weights.sum(dim=-1), torch.ones(2), rtol=1e-6, atol=1e-6,
        )

    def test_future_months_cannot_change_earlier_distributions(self):
        changed_images = self.images.clone()
        changed_weather = self.weather.clone()
        changed_images[:, 4:] += 100.0
        changed_weather[:, 4:] -= 100.0

        with torch.no_grad():
            _, first_weights, first_aux = self.model(
                self.images, self.weather, self.soil,
            )
            _, _, changed_aux = self.model(
                changed_images, changed_weather, self.soil,
            )

        torch.testing.assert_close(
            first_aux["pred_dist_all"][:, :4],
            changed_aux["pred_dist_all"][:, :4],
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertTrue(torch.equal(first_weights.triu(1), torch.zeros_like(first_weights)))

    def test_last_distribution_uses_last_valid_month_with_gaps(self):
        month_mask = torch.tensor([
            [True, False, True, False, False, False],
            [False, True, False, True, False, True],
        ])
        with torch.no_grad():
            final_dist, _, aux = self.model(
                self.images, self.weather, self.soil, month_mask,
            )
        expected = torch.stack((aux["pred_dist_all"][0, 2], aux["pred_dist_all"][1, 5]))
        torch.testing.assert_close(final_dist, expected)
        self.assertTrue(torch.equal(aux["month_valid"], month_mask))

    def test_last_true_index_handles_sparse_and_all_invalid_masks(self):
        from TFT_model.mmst_tft_model import last_true_index

        mask = torch.tensor([
            [True, False, True, False, False, False],
            [False, True, False, True, False, True],
            [False, False, False, False, False, False],
        ])
        indices, sample_valid = last_true_index(mask)
        self.assertTrue(torch.equal(indices, torch.tensor([2, 5, 0])))
        self.assertTrue(torch.equal(sample_valid, torch.tensor([True, True, False])))

    def test_all_invalid_months_are_finite_and_zero(self):
        month_mask = torch.zeros(2, 6, dtype=torch.bool)
        with torch.no_grad():
            final_dist, weights, aux = self.model(
                self.images, self.weather, self.soil, month_mask,
            )
        self.assertTrue(torch.isfinite(final_dist).all())
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.isfinite(aux["pred_dist_all"]).all())
        self.assertTrue(torch.equal(final_dist, torch.zeros_like(final_dist)))
        self.assertTrue(torch.equal(weights, torch.zeros_like(weights)))
        self.assertTrue(torch.equal(
            aux["pred_dist_all"], torch.zeros_like(aux["pred_dist_all"]),
        ))

    def test_finite_backward_reaches_frontend_soil_temporal_and_heads(self):
        self.model.train()
        final_dist, _, aux = self.model(
            self.images, self.weather, self.soil,
            torch.tensor([[True, True, True, True, True, True],
                          [True, False, True, True, False, True]]),
        )
        loss = final_dist.square().mean() + aux["pred_dist_all"].square().mean()
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        modules = {
            "soil": self.model.soil_static_encoder,
            "lstm": self.model.lstm_encoder,
            "temporal_grn": self.model.temporal_grn,
            "attention": self.model.attention,
            "pred_grn": self.model.pred_grn,
            "yield_head": self.model.yield_head,
        }
        for name, module in modules.items():
            with self.subTest(module=name):
                gradients = [
                    parameter.grad
                    for parameter in module.parameters()
                    if parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(all(
                    torch.isfinite(gradient).all() for gradient in gradients
                ))
                self.assertGreater(
                    sum(gradient.abs().sum().item() for gradient in gradients),
                    0.0,
                )


class MMSTRealConfigTests(unittest.TestCase):
    def setUp(self):
        self.warning_context = warnings.catch_warnings()
        self.warning_context.__enter__()
        warnings.filterwarnings(
            "ignore",
            message=r"unclosed file <_io.TextIOWrapper name='.*/MMST-ViT/data/corn_(?:train|val)\.json'.*",
            category=ResourceWarning,
        )

    def tearDown(self):
        self.warning_context.__exit__(None, None, None)

    def test_real_train_and_eval_counts_order_and_soil_coverage(self):
        self.assertTrue(SOIL_FILE.is_file(), f"真实土壤文件不存在: {SOIL_FILE}")

        from TFT_model.mmst_data import MMSTCornDataset

        for filename, expected_count in (("corn_train.json", 195), ("corn_val.json", 239)):
            config = MMST_ROOT / "data" / filename
            records = json.loads(config.read_text(encoding="utf-8"))
            dataset = MMSTCornDataset(DATA_ROOT, config, SOIL_FILE)
            self.addCleanup(dataset.close)
            expected = [(record["FIPS"], record["year"]) for record in records]
            actual = list(zip(dataset.fips_codes, dataset.years))
            self.assertEqual(len(dataset), expected_count)
            self.assertEqual(actual, expected)

    def test_real_first_train_short_weather_has_fixed_official_value(self):
        from TFT_model.mmst_data import MMSTCornDataset

        config = MMST_ROOT / "data/corn_train.json"
        self.assertTrue(DATA_ROOT.is_dir(), f"真实数据根目录不存在: {DATA_ROOT}")
        self.assertTrue(SOIL_FILE.is_file(), f"真实土壤文件不存在: {SOIL_FILE}")
        dataset = MMSTCornDataset(DATA_ROOT, config, SOIL_FILE)
        self.addCleanup(dataset.close)

        _, short_weather, _, _, fips, year = dataset[0]
        expected = torch.tensor([
            275.7619934082031,
            280.447998046875,
            271.7650146484375,
            0.0,
            51.0,
            8.354999542236328,
            7.236000061035156,
            13607.2001953125,
            0.3889999985694885,
        ], dtype=torch.float32)
        self.assertEqual((fips, year), ("17005", 2021))
        torch.testing.assert_close(
            short_weather[0, 0, 0], expected, rtol=1e-6, atol=1e-6,
        )


class MMSTFormalTrainingTests(unittest.TestCase):
    def test_parser_is_the_fixed_single_gpu_corn_protocol(self):
        from TFT_model.train_mmst_frontend import (
            DEFAULT_CHECKPOINT,
            build_parser,
            validate_args,
        )

        args = build_parser().parse_args([])
        validate_args(args)
        self.assertEqual(args.epochs, 100)
        self.assertEqual(args.lr, 1e-3)
        self.assertEqual(args.weight_decay, 0.05)
        self.assertEqual(args.warmup_epochs, 5)
        self.assertEqual(args.outer_batch_size, 1)
        self.assertEqual(args.train_year, 2021)
        self.assertEqual(Path(args.train_config).name, "corn_train.json")
        self.assertEqual(Path(args.pretrained).resolve(), DEFAULT_CHECKPOINT.resolve())
        self.assertEqual(args.device, "cuda:0")
        self.assertFalse(hasattr(args, "val_config"))
        self.assertFalse(hasattr(args, "early_stop"))

        args.pretrained = str(DEFAULT_CHECKPOINT.with_name("checkpoint-final.pth"))
        with self.assertRaisesRegex(ValueError, "checkpoint-latest"):
            validate_args(args)

    def test_training_transform_matches_isolated_official_data_wrapper(self):
        from TFT_model.train_mmst_frontend import (
            DataWrapper,
            transform_training_images,
        )

        raw = torch.arange(
            1 * 6 * 2 * 240 * 240 * 3, dtype=torch.uint8,
        ).reshape(1, 6, 2, 240, 240, 3).remainder(255)
        torch.manual_seed(123)
        actual = transform_training_images(raw)

        flat = raw.reshape(-1, 240, 240, 3).permute(0, 3, 1, 2)
        torch.manual_seed(123)
        expected, _ = DataWrapper(train=True)(flat)
        expected = expected.reshape(1, 6, 2, 3, 224, 224)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(actual.shape, (1, 6, 2, 3, 224, 224))
        self.assertEqual(actual.dtype, torch.float32)
        self.assertNotIn("dataset", sys.modules)

    def test_soil_stats_and_zscore_use_only_training_counties(self):
        from TFT_model.train_mmst_frontend import compute_soil_stats, zscore_soil

        class SoilOnlyDataset:
            soil_by_fips = {
                "a": torch.tensor([1.0, 2.0]),
                "b": torch.tensor([3.0, 6.0]),
                "c": torch.tensor([5.0, 10.0]),
            }
            fips_codes = ["a", "b", "c"]
            years = [2021, 2021, 2022]

        mean, std, source = compute_soil_stats(SoilOnlyDataset(), train_year=2021)
        torch.testing.assert_close(mean, torch.tensor([2.0, 4.0]))
        torch.testing.assert_close(std, torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(
            zscore_soil(torch.tensor([[1.0, 6.0]]), mean, std),
            torch.tensor([[-1.0, 1.0]]),
        )
        self.assertEqual(source, {"year": 2021, "county_count": 2})

    def test_loss_is_single_head_mse(self):
        prediction = torch.tensor([[1.5]], requires_grad=True)
        target = torch.tensor([[2.5]])
        loss = torch.nn.MSELoss()(prediction, target)
        self.assertAlmostEqual(loss.item(), 1.0, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_learning_rate_is_linear_warmup_then_cosine(self):
        from TFT_model.train_mmst_frontend import adjust_learning_rate

        parameter = nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        self.assertEqual(adjust_learning_rate(
            optimizer, 0.0, base_lr=1e-3, warmup_epochs=5, total_epochs=100,
        ), 0.0)
        self.assertAlmostEqual(adjust_learning_rate(
            optimizer, 2.5, base_lr=1e-3, warmup_epochs=5, total_epochs=100,
        ), 5e-4)
        self.assertAlmostEqual(adjust_learning_rate(
            optimizer, 5.0, base_lr=1e-3, warmup_epochs=5, total_epochs=100,
        ), 1e-3)
        self.assertAlmostEqual(adjust_learning_rate(
            optimizer, 100.0, base_lr=1e-3, warmup_epochs=5, total_epochs=100,
        ), 0.0, places=12)

    def test_optimizer_is_fixed_adamw_with_bias_and_norm_excluded_from_decay(self):
        from TFT_model.train_mmst_frontend import create_optimizer

        model = nn.Sequential(nn.Linear(2, 2), nn.LayerNorm(2))
        optimizer = create_optimizer(model, lr=1e-3, weight_decay=0.05)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.999))
        self.assertEqual([group["weight_decay"] for group in optimizer.param_groups], [0.05, 0.0])
        self.assertEqual(optimizer.defaults["lr"], 1e-3)

    def test_atomic_checkpoint_is_one_complete_file(self):
        from TFT_model.train_mmst_frontend import atomic_save_checkpoint

        model = nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint-final.pth"
            atomic_save_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                epoch=100,
                config={"epochs": 100},
                schedule={"name": "linear-warmup-cosine", "warmup_epochs": 5},
                soil_stats={"mean": [1.0], "std": [2.0], "source": {"county_count": 195}},
                source_checkpoint="checkpoint-latest.pth",
            )
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(set(payload), {
                "model", "optimizer", "epoch", "config", "schedule",
                "soil_stats", "source_checkpoint",
            })
            self.assertEqual(payload["epoch"], 100)
            self.assertEqual(payload["soil_stats"]["source"]["county_count"], 195)
            self.assertEqual([item.name for item in Path(directory).iterdir()], [path.name])

    def test_formal_entrypoint_has_no_plotting_or_legacy_training_imports(self):
        source = (ROOT / "TFT_model/train_mmst_frontend.py").read_text(encoding="utf-8")
        self.assertNotIn("matplotlib", source)
        self.assertNotIn("from .train import", source)
        self.assertNotIn("from train import", source)
        self.assertNotIn("from .data import", source)
        self.assertNotIn("from data import", source)
        self.assertNotIn("from .models import", source)
        self.assertNotIn("from models import", source)
        self.assertNotIn("DataParallel", source)
        self.assertNotIn("DistributedDataParallel", source)

    def test_launcher_points_only_to_fixed_corn_entrypoint_on_cuda_zero(self):
        launcher = (ROOT / "TFT_model/run_finetune.sh").read_text(encoding="utf-8")
        self.assertIn("CUDA_VISIBLE_DEVICES=\"${CUDA_VISIBLE_DEVICES:-0}\"", launcher)
        self.assertIn("train_mmst_frontend.py", launcher)
        self.assertIn("corn_train.json", launcher)
        self.assertIn("checkpoint-latest.pth", launcher)
        self.assertNotIn("train.py", launcher.replace("train_mmst_frontend.py", ""))
        self.assertNotIn("torch.distributed", launcher)

        environment = dict(os.environ)
        environment["PYTHON"] = "echo"
        result = subprocess.run(
            [
                "bash", str(ROOT / "TFT_model/run_finetune.sh"),
                "--hidden-size", "8", "--heads", "1", "--dropout", "0.9",
                "--seed", "999",
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        arguments = result.stdout.split()
        for option, expected in (
            ("--hidden-size", "512"),
            ("--heads", "4"),
            ("--dropout", "0.1"),
            ("--seed", "42"),
        ):
            positions = [index for index, value in enumerate(arguments) if value == option]
            self.assertEqual(arguments[positions[-1] + 1], expected)

    @unittest.skipUnless(
        torch.cuda.is_available() and DATA_ROOT.is_dir() and SOIL_FILE.is_file()
        and (MMST_ROOT / "output_dir/pvt_simclr/checkpoint-latest.pth").is_file(),
        "需要 CUDA、正式 MMST 数据和 checkpoint",
    )
    def test_one_real_county_formal_frontend_forward_backward_is_finite(self):
        from TFT_model.mmst_data import MMSTCornDataset
        from TFT_model.mmst_tft_model import MMSTTFTYieldModel
        from TFT_model.train_mmst_frontend import (
            compute_soil_stats, transform_training_images, zscore_soil,
        )

        checkpoint = MMST_ROOT / "output_dir/pvt_simclr/checkpoint-latest.pth"
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"unclosed file <_io.TextIOWrapper name='.*/corn_train\.json'.*",
                category=ResourceWarning,
            )
            dataset = MMSTCornDataset(
                DATA_ROOT, MMST_ROOT / "data/corn_train.json", SOIL_FILE,
            )
        self.addCleanup(dataset.close)
        images, weather, target, soil, _, _ = dataset[0]
        grid_limit = min(1, images.shape[1])
        images = transform_training_images(images[None, :, :grid_limit]).cuda()
        weather = weather[None, :, :grid_limit].cuda()
        mean, std, _ = compute_soil_stats(dataset, train_year=2021)
        soil = zscore_soil(soil[None], mean, std).cuda()
        target = target.reshape(1, 1).cuda()

        model = MMSTTFTYieldModel(checkpoint).cuda().train()
        prediction, _, aux = model(images, weather, soil)
        loss = torch.nn.MSELoss()(prediction, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        modules = {
            "frontend": model.frontend,
            "soil": model.soil_static_encoder,
            "lstm": model.lstm_encoder,
            "attention": model.attention,
            "head": model.yield_head,
        }
        for name, module in modules.items():
            with self.subTest(module=name):
                gradients = [
                    parameter.grad for parameter in module.parameters()
                    if parameter.grad is not None
                ]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                self.assertGreater(
                    sum(gradient.abs().sum().item() for gradient in gradients), 0.0,
                )


class LSTMEncoderSparseMaskTests(unittest.TestCase):
    def test_sparse_valid_mask_returns_the_actual_last_true_step(self):
        from TFT_model.models import LSTMEncoder

        torch.manual_seed(31)
        encoder = LSTMEncoder(
            input_size=4, hidden_size=4, dropout=0.0, have_context=True,
        ).eval()
        inputs = torch.randn(3, 6, 4)
        valid = torch.tensor([
            [True, False, True, False, False, False],
            [False, True, False, True, False, True],
            [False, False, False, False, False, False],
        ])
        zeros = torch.zeros(3, 4)
        features, last_hidden, enc_h_last, enc_c_last = encoder(
            inputs, valid.sum(dim=1), c_c=zeros, c_h=zeros, valid_mask=valid,
        )

        torch.testing.assert_close(last_hidden[0], features[0, 2])
        torch.testing.assert_close(last_hidden[1], features[1, 5])
        self.assertTrue(torch.equal(last_hidden[2], torch.zeros_like(last_hidden[2])))
        self.assertTrue(torch.equal(enc_h_last[2], torch.zeros_like(enc_h_last[2])))
        self.assertTrue(torch.equal(enc_c_last[2], torch.zeros_like(enc_c_last[2])))


if __name__ == "__main__":
    unittest.main()
