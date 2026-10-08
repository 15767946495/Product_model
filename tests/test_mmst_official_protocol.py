import json
import math
import multiprocessing as mp
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
MMST_ROOT = ROOT / "MMST-ViT"
sys.path.insert(0, str(MMST_ROOT))

import current_data
import main_finetune_mmst_vit as finetune
import main_pretrain_mmst_vit as pretrain
from dataset.data_wrapper import DataWrapper


def _gloo_worker(rank, world_size, init_method, queue):
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world_size)
    class CountingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.projection = torch.nn.Linear(3, 4)
            self.calls = 0

        def forward(self, images, weather):
            self.calls += 1
            return self.projection(images.flatten(1)[:, :3])

    model = torch.nn.parallel.DistributedDataParallel(CountingModel())
    with torch.no_grad():
        model.module.projection.weight.copy_(torch.eye(4, 3))
        model.module.projection.bias.zero_()
    initial = model.module.projection.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    args = type("Args", (), {
        "batch_size": 4, "accum_iter": 1, "distributed": True,
        "lr": 0.01, "min_lr": 0.01, "warmup_epochs": 0, "epochs": 1,
    })()
    image = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.]]).view(4, 3, 1, 1)
    batches = [(image, image.roll(1, 0), torch.zeros(4, 1, 9))]
    if rank == 0:
        batches.append((image * 2, image.roll(2, 0), torch.zeros(4, 1, 9)))
    county = [(torch.zeros(1, len(batches) * 4, 1, 1, 1, 3), torch.zeros(1, len(batches) * 4, 1, 1, 9))]
    with patch.object(pretrain.sentinel_wrapper, "get_data_loader", return_value=batches):
        stats = pretrain.train_one_epoch(model, county, optimizer, torch.device("cpu"), 0, None, args=args)
    queue.put((rank, model.module.calls, stats["optimizer_steps"],
               (initial - model.module.projection.weight.detach()).tolist()))
    dist.destroy_process_group()


class MMSTOfficialProtocolTests(unittest.TestCase):
    def test_run_scripts_only_exec_main_python_entrypoints(self):
        pretrain = (MMST_ROOT / "run_pretrain.sh").read_text()
        finetune = (MMST_ROOT / "run_finetune.sh").read_text()
        self.assertIn('PYTHON="${PYTHON:-python}"', pretrain)
        self.assertIn('PYTHON="${PYTHON:-python}"', finetune)
        self.assertIn('NPROC_PER_NODE="${NPROC_PER_NODE:-8}"', pretrain)
        self.assertIn('NPROC_PER_NODE="${NPROC_PER_NODE:-8}"', finetune)
        self.assertIn('-m torch.distributed.run', pretrain)
        self.assertIn('-m torch.distributed.run', finetune)
        self.assertIn('exec "$PYTHON" -m torch.distributed.run', pretrain)
        self.assertIn('exec "$PYTHON" -m torch.distributed.run', finetune)
        self.assertIn('--nproc_per_node="$NPROC_PER_NODE"', pretrain)
        self.assertIn('--nproc_per_node="$NPROC_PER_NODE"', finetune)
        self.assertIn('main_pretrain_mmst_vit.py "$@"', pretrain)
        self.assertIn('main_finetune_mmst_vit.py "$@"', finetune)
        for source in (pretrain, finetune):
            self.assertNotIn("current_data.py", source)
            self.assertNotIn("pip install", source)

    def test_eval_sampler_does_not_pad_non_divisible_dataset(self):
        shards = [finetune.eval_indices_for_rank(5, 3, rank) for rank in range(3)]
        self.assertEqual(sorted(index for shard in shards for index in shard), [0, 1, 2, 3, 4])
        self.assertEqual(len(set(index for shard in shards for index in shard)), 5)

    def test_training_distributed_sampler_uses_official_padding_duplicates(self):
        dataset = list(range(5))
        shards = [
            list(torch.utils.data.DistributedSampler(
                dataset, num_replicas=3, rank=rank, shuffle=False, drop_last=False,
            ))
            for rank in range(3)
        ]
        flattened = [index for shard in shards for index in shard]
        self.assertEqual(len(flattened), 6)
        self.assertEqual(len(set(flattened)), 5)

    def test_eval_only_loads_eval_manifest_and_does_not_save_checkpoint(self):
        args = finetune.get_args_parser().parse_args(["--eval", "--device", "cpu"])
        with patch.object(finetune, "_load_records", side_effect=[[{"year": 2022}], AssertionError("train loaded")]) as load_records:
            finetune.select_records_for_mode(args)
        load_records.assert_called_once_with(args.data_file_val, finetune.EVAL_YEAR)

    def test_eval_only_does_not_save_checkpoint(self):
        args = finetune.get_args_parser().parse_args(["--eval", "--device", "cpu"])
        self.assertFalse(finetune.should_save_checkpoint(args))

    def test_checkpoint_contains_complete_resume_state(self):
        args = finetune.get_args_parser().parse_args([])
        args.epochs = 3
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scaler = finetune._make_scaler(torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            path = finetune.save_checkpoint(model, optimizer, scaler, epoch=2, args=args, output=Path(directory))
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["epoch"], 2)
        self.assertIn("optimizer", checkpoint)
        self.assertIn("scaler", checkpoint)
        self.assertIn("args", checkpoint)
        self.assertEqual(checkpoint["protocol"]["crop"], "corn")

    def test_resume_restores_model_optimizer_scaler_and_next_epoch(self):
        args = finetune.get_args_parser().parse_args([])
        source = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(source.parameters(), lr=0.1)
        scaler = finetune._make_scaler(torch.device("cpu"))
        with tempfile.TemporaryDirectory() as directory:
            finetune.save_checkpoint(source, optimizer, scaler, epoch=4, args=args, output=Path(directory))
            target = torch.nn.Linear(2, 2)
            target_optimizer = torch.optim.SGD(target.parameters(), lr=0.01)
            target_scaler = finetune._make_scaler(torch.device("cpu"))
            args.resume = str(Path(directory) / "checkpoint-4.pth")
            finetune.load_checkpoint(args, target, target_optimizer, target_scaler)
        self.assertEqual(args.start_epoch, 5)
        for expected, actual in zip(source.parameters(), target.parameters()):
            self.assertTrue(torch.equal(expected, actual))

    def test_checkpoint_epoch_is_last_completed_zero_based_epoch(self):
        self.assertEqual(finetune.last_completed_epoch(0, 200), 199)
        self.assertEqual(finetune.last_completed_epoch(5, 10), 9)
        self.assertEqual(finetune.last_completed_epoch(5, 5), 4)

    def test_eval_only_does_not_prepare_training_state(self):
        args = finetune.get_args_parser().parse_args(["--eval"])
        model = torch.nn.Linear(2, 2)
        with patch.object(finetune, "build_optimizer", side_effect=AssertionError("optimizer built")) as build, \
             patch.object(finetune, "_make_scaler", side_effect=AssertionError("scaler built")) as scaler:
            optimizer, grad_scaler = finetune.prepare_training_state(args, model, torch.device("cpu"))
        self.assertIsNone(optimizer)
        self.assertIsNone(grad_scaler)
        build.assert_not_called()
        scaler.assert_not_called()

    def test_backend_follows_requested_device(self):
        self.assertEqual(finetune.backend_for_device("cpu"), "gloo")
        with patch.object(torch.cuda, "is_available", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                finetune.backend_for_device("cuda")

    def test_eval_uses_requested_input_size(self):
        args = finetune.get_args_parser().parse_args(["--input_size", "96"])
        with patch.object(finetune, "DataWrapper") as wrapper:
            wrapper.return_value = lambda images: (images, images)
            model = torch.nn.Linear(1, 2)
            model.forward = lambda images, ys=None, yl=None: torch.zeros(images.shape[0], 2)
            loader = [(torch.zeros(1, 1, 1, 2, 2, 3), torch.zeros(1, 1, 1, 1, 9),
                       torch.zeros(1, 1, 1, 1, 9), torch.zeros(1, 2), {})]
            with patch.object(finetune, "metrics_from_prediction_pairs", return_value={"rmse": 0, "r2": 0, "corr": 0}):
                finetune.evaluate(model, loader, torch.device("cpu"), input_size=args.input_size)
        wrapper.assert_called_once_with(img_size=96, train=False)

    def test_supervised_uint8_images_use_unit_scale_before_train_and_eval_transforms(self):
        images = torch.full((1, 1, 1, 224, 224, 3), 255, dtype=torch.uint8)
        train = finetune._prepare_images(images, DataWrapper(img_size=224, train=True))
        eval_image = finetune._prepare_images(images, DataWrapper(img_size=224, train=False))
        already_scaled = finetune._prepare_images(
            images.float().div(255), DataWrapper(img_size=224, train=False),
        )
        expected = torch.tensor([
            (1.0 - 0.466) / 0.195,
            (1.0 - 0.471) / 0.194,
            (1.0 - 0.380) / 0.192,
        ])
        self.assertTrue(torch.allclose(eval_image[0, 0, 0, :, 0, 0], expected, atol=1e-5))
        self.assertTrue(torch.allclose(eval_image, already_scaled, atol=1e-5))
        self.assertLess(float(train.abs().max()), 6.0)
        self.assertLess(float(eval_image.abs().max()), 6.0)

    def test_manifest_crop_is_corn_and_wrong_crop_is_rejected(self):
        record = {"state": "iowa", "year": 2022, "crop": "soybean", "production": 1, "yield_per_acre": 1}
        with self.assertRaisesRegex(ValueError, "crop"):
            current_data.validate_records([record], {2022}, require_targets=True)

    def test_lr_uses_official_effective_batch_with_county_outer_batch_one(self):
        args = finetune.get_args_parser().parse_args(["--batch_size", "64", "--accum_iter", "2"])
        self.assertEqual(finetune.effective_batch_size(args, world_size=3), 384)
        self.assertEqual(finetune.effective_learning_rate(args, world_size=3), args.blr * 384 / 256)

    def test_finetune_defaults_match_official_source(self):
        args = finetune.get_args_parser().parse_args([])
        self.assertEqual(args.epochs, 200)
        self.assertEqual(args.batch_size, 64)
        self.assertEqual(args.warmup_epochs, 40)
        self.assertEqual(args.blr, 1e-3)
        self.assertEqual(args.beta1, 0.9)
        self.assertEqual(args.beta2, 0.95)
        self.assertEqual(args.data_file_train, "./data/finetune_2021.json")
        self.assertEqual(args.data_file_val, "./data/eval_2022.json")

    def test_finetune_protocol_uses_final_evaluation_only(self):
        self.assertEqual(finetune.TRAIN_YEAR, 2021)
        self.assertEqual(finetune.EVAL_YEAR, 2022)
        self.assertFalse(finetune.SELECT_CHECKPOINT_ON_EVAL)

    def test_finetune_optimizer_uses_official_betas(self):
        model = torch.nn.Linear(3, 2)
        args = finetune.get_args_parser().parse_args([])
        args.lr = 0.01
        optimizer = finetune.build_optimizer(model, args)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.95))

    def test_finetune_model_has_two_targets(self):
        class FakeBackbone(torch.nn.Module):
            def forward(self, x, context=None):
                return x

        model = finetune.build_model(FakeBackbone(), embed_dim=3, batch_size=2)
        self.assertEqual(model.mlp_head[-1].out_features, 1)

    def test_final_evaluation_gathers_pairs_before_metrics(self):
        predictions = torch.tensor([[math.log(11.0), math.log(21.0)]])
        labels = torch.tensor([[math.log(10.0), math.log(20.0)]])
        gathered = finetune.collect_prediction_pairs(predictions, labels, distributed=False)
        self.assertEqual(gathered[0].shape, (1, 2))
        result = finetune.metrics_from_prediction_pairs(*gathered)
        self.assertAlmostEqual(result["rmse"], 1.0, places=5)

    def test_accumulation_boundaries_include_tail_window(self):
        self.assertEqual(finetune.accumulation_window_sizes(5, 2), [2, 2, 1])

    def test_finetune_training_steps_tail_window(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))

            def forward(self, images, ys=None, yl=None):
                return images.mean(dim=tuple(range(1, images.ndim))).unsqueeze(1).repeat(1, 2) * self.weight

        model = FakeModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        args = finetune.get_args_parser().parse_args([])
        args.device = "cpu"
        args.lr = 0.1
        args.min_lr = 0.1
        args.epochs = 1
        args.warmup_epochs = 0
        args.accum_iter = 2
        batches = [
            (torch.ones(1, 1, 1, 2, 2, 3), torch.zeros(1, 1, 1, 1, 9),
             torch.zeros(1, 1, 1, 1, 9), torch.zeros(1, 2), {}),
            (torch.ones(1, 1, 1, 2, 2, 3), torch.zeros(1, 1, 1, 1, 9),
             torch.zeros(1, 1, 1, 1, 9), torch.zeros(1, 2), {}),
            (torch.ones(1, 1, 1, 2, 2, 3), torch.zeros(1, 1, 1, 1, 9),
             torch.zeros(1, 1, 1, 1, 9), torch.zeros(1, 2), {}),
        ]
        with patch.object(finetune, "_prepare_images", side_effect=lambda images, wrapper: images):
            stats = finetune.train_one_epoch(model, batches, optimizer, torch.device("cpu"), 0, args)
        self.assertEqual(stats["optimizer_steps"], 2)
        self.assertIsNone(model.weight.grad)

    def test_pretrain_defaults_match_official_source(self):
        args = pretrain.get_args_parser().parse_args([])
        self.assertEqual(args.epochs, 200)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.warmup_epochs, 20)
        self.assertEqual(args.weight_decay, 0.05)
        self.assertEqual(args.blr, 1e-3)
        self.assertEqual(args.data_file, "./data/pretrain_2017_2020.json")
        self.assertTrue(args.imagenet_pretrained)

    def test_add_weight_decay_groups_bias_norm_and_matrix_parameters(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(3, 2),
            torch.nn.LayerNorm(2),
        )
        model[0].bias.requires_grad = False

        groups = pretrain.add_weight_decay(model, 0.05)

        self.assertEqual([group["weight_decay"] for group in groups], [0.0, 0.05])
        self.assertEqual(
            {id(parameter) for parameter in groups[0]["params"]},
            {id(model[1].weight), id(model[1].bias)},
        )
        self.assertEqual(
            {id(parameter) for parameter in groups[1]["params"]},
            {id(model[0].weight)},
        )

    def test_add_weight_decay_has_no_missing_or_duplicate_trainable_parameters(self):
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 3),
            torch.nn.LayerNorm(3),
            torch.nn.Linear(3, 2, bias=False),
        )

        groups = pretrain.add_weight_decay(model, 0.1)
        grouped = [parameter for group in groups for parameter in group["params"]]
        expected = [parameter for parameter in model.parameters() if parameter.requires_grad]

        self.assertEqual(len(grouped), len(expected))
        self.assertEqual(len({id(parameter) for parameter in grouped}), len(grouped))
        self.assertEqual({id(parameter) for parameter in grouped}, {id(parameter) for parameter in expected})
        self.assertIs(finetune.add_weight_decay, pretrain.add_weight_decay)

    def test_pretrain_protocol_years_are_fixed(self):
        self.assertEqual(pretrain.PRETRAIN_YEARS, (2017, 2018, 2019, 2020))

    def test_pretrain_checkpoint_is_complete_and_strictly_loadable(self):
        args = pretrain.get_args_parser().parse_args([])
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scaler = type("Scaler", (), {
            "state_dict": lambda self: {"scale": 1},
        })()
        with tempfile.TemporaryDirectory() as directory:
            path = pretrain.save_checkpoint(
                model, optimizer, scaler, epoch=4, args=args,
                output=Path(directory), filename="checkpoint-final.pth",
            )
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            target = torch.nn.Linear(3, 2)
            finetune._load_pretrained(target, path)
        self.assertEqual(checkpoint["epoch"], 4)
        self.assertIn("optimizer", checkpoint)
        self.assertIn("scaler", checkpoint)
        self.assertIn("args", checkpoint)
        self.assertEqual(checkpoint["protocol"]["states"], finetune.PROTOCOL_STATES)
        self.assertEqual(checkpoint["protocol"]["crop"], "corn")
        self.assertEqual(checkpoint["protocol"]["pretrain_years"], list(pretrain.PRETRAIN_YEARS))
        for expected_parameter, actual_parameter in zip(model.parameters(), target.parameters()):
            self.assertTrue(torch.equal(expected_parameter, actual_parameter))

    def test_pretrain_zero_epochs_does_not_write_final_checkpoint(self):
        args = pretrain.get_args_parser().parse_args(["--epochs", "0"])
        self.assertIsNone(pretrain.final_completed_epoch(args.start_epoch, args.epochs))

    def test_entrypoints_support_manifest_rebuild_data_check(self):
        pretrain_args = pretrain.get_args_parser().parse_args([
            "--rebuild_manifest", "--data_check",
        ])
        finetune_args = finetune.get_args_parser().parse_args([
            "--rebuild_manifest", "--data_check",
        ])
        self.assertTrue(pretrain_args.rebuild_manifest)
        self.assertTrue(pretrain_args.data_check)
        self.assertTrue(finetune_args.rebuild_manifest)
        self.assertTrue(finetune_args.data_check)

    def test_pretrain_data_check_exits_before_training_imports(self):
        args = pretrain.get_args_parser().parse_args(["--data_check"])
        records = [{"state": state, "year": 2017} for state in current_data.FOUR_STATES]
        with patch.object(pretrain, "ensure_manifest", return_value=records), \
             patch.object(pretrain, "validate_pretrain_records"):
            pretrain.main(args)

    def test_pretrain_cpu_entrypoint_has_no_torch_six_startup_blocker(self):
        command = [
            sys.executable,
            "main_pretrain_mmst_vit.py",
            "--device", "cpu",
            "--epochs", "0",
            "--num_workers", "0",
            "--data_file", "data/pretrain_2017_2020.json",
            "--output_dir", "",
            "--log_dir", "",
            "--no_imagenet_pretrained",
        ]
        result = subprocess.run(
            command,
            cwd=MMST_ROOT,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertNotIn("torch._six", output)
        self.assertIn("Not using distributed mode", output)
        self.assertEqual(result.returncode, 0, output)
        self.assertIn("Training time", output)

    def test_finetune_eval_data_check_has_no_torch_six_startup_blocker(self):
        result = subprocess.run(
            [
                sys.executable,
                "main_finetune_mmst_vit.py",
                "--eval",
                "--data_check",
                "--device", "cpu",
                "--num_workers", "0",
            ],
            cwd=MMST_ROOT,
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        output = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, output)
        self.assertNotIn("torch._six", output)
        self.assertIn('"eval_records": 245', output)

    def test_ensure_manifest_rebuilds_only_when_requested(self):
        records = [{"state": state, "year": 2017} for state in current_data.FOUR_STATES]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            with patch.object(pretrain, "build_manifest", return_value=records) as build:
                self.assertEqual(pretrain.ensure_manifest(path, pretrain.PRETRAIN_YEARS), records)
                build.assert_called_once()
            path.write_text(json.dumps(records), encoding="utf-8")
            with patch.object(pretrain, "build_manifest") as build:
                self.assertEqual(pretrain.ensure_manifest(path, pretrain.PRETRAIN_YEARS), records)
                build.assert_not_called()

    def test_synchronized_inner_batches_marks_small_batch_as_padding(self):
        batches = [
            (torch.zeros(4, 3, 2, 2), torch.zeros(4, 3, 2, 2), torch.zeros(4, 9)),
            (torch.zeros(1, 3, 2, 2), torch.zeros(1, 3, 2, 2), torch.zeros(1, 9)),
        ]
        with patch.object(pretrain.sentinel_wrapper, "get_data_loader", return_value=batches):
            result = list(pretrain.synchronized_inner_batches(
                torch.zeros(1), torch.zeros(1), batch_size=4, distributed=False
            ))
        self.assertEqual([item[-1] for item in result], [True, False])

    def test_cpu_smoke_updates_fake_model_once_without_weights(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(3, 4)

            def forward(self, images, weather):
                return self.projection(images.flatten(1)[:, :3])

        model = FakeModel()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        before = model.projection.weight.detach().clone()
        args = type("Args", (), {
            "batch_size": 4, "accum_iter": 1, "distributed": False,
            "lr": 0.01, "min_lr": 0.0, "warmup_epochs": 0, "epochs": 1,
        })()
        county = [(torch.zeros(1, 12, 1, 2, 2, 3), torch.zeros(1, 12, 1, 9))]
        inner = [(torch.ones(4, 3, 2, 2), torch.ones(4, 3, 2, 2), torch.zeros(4, 9))]
        with patch.object(pretrain.sentinel_wrapper, "get_data_loader", return_value=inner):
            stats = pretrain.train_one_epoch(
                model, county, optimizer, torch.device("cpu"), 0, None, args=args
            )
        self.assertTrue(torch.isfinite(torch.tensor(stats["loss"])))
        self.assertFalse(torch.equal(before, model.projection.weight.detach()))

    def test_accumulation_flushes_tail_and_clears_gradients(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(3, 4)

            def forward(self, images, weather):
                return self.projection(images.flatten(1)[:, :3])

        model = FakeModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        args = type("Args", (), {
            "batch_size": 4, "accum_iter": 2, "distributed": False,
            "lr": 0.01, "min_lr": 0.0, "warmup_epochs": 0, "epochs": 1,
            "total_active_slots": 3,
        })()
        county = [(torch.zeros(1, 12, 1, 2, 2, 3), torch.zeros(1, 12, 1, 9))]
        inner = [
            (torch.ones(4, 3, 2, 2), torch.ones(4, 3, 2, 2), torch.zeros(4, 9)),
            (torch.ones(4, 3, 2, 2) * 2, torch.ones(4, 3, 2, 2) * 2, torch.zeros(4, 9)),
            (torch.ones(4, 3, 2, 2) * 3, torch.ones(4, 3, 2, 2) * 3, torch.zeros(4, 9)),
        ]
        with patch.object(pretrain.sentinel_wrapper, "get_data_loader", return_value=inner), \
             patch.object(pretrain, "_adjust_learning_rate", wraps=pretrain._adjust_learning_rate) as adjust:
            stats = pretrain.train_one_epoch(
                model, county, optimizer, torch.device("cpu"), 0, None, args=args
            )
        self.assertEqual(stats["optimizer_steps"], 2)
        self.assertEqual(adjust.call_count, 2)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_accumulation_tail_uses_actual_window_size(self):
        class FakeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = torch.nn.Linear(3, 3, bias=False)

            def forward(self, images, weather):
                return self.projection(images.flatten(1)[:, :3])

        def run(inner, accum_iter=2):
            model = FakeModel()
            with torch.no_grad():
                model.projection.weight.copy_(torch.eye(3))
            optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
            args = type("Args", (), {
                "batch_size": 4, "accum_iter": accum_iter, "distributed": False,
                "lr": 1.0, "min_lr": 1.0, "warmup_epochs": 0, "epochs": 1,
            })()
            county = [(torch.zeros(1, len(inner) * 4, 1, 2, 2, 3), torch.zeros(1, len(inner) * 4, 1, 9))]
            with patch.object(pretrain.sentinel_wrapper, "get_data_loader", return_value=inner):
                pretrain.train_one_epoch(model, county, optimizer, torch.device("cpu"), 0, None, args=args)
            return model.projection.weight.detach().clone()

        image = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.]]).view(4, 3, 1, 1)
        batch = (image, image.roll(1, 0), torch.zeros(4, 9))
        tail = run([batch], accum_iter=2)
        reference = run([batch], accum_iter=1)
        self.assertGreater(torch.linalg.vector_norm(torch.eye(3) - reference).item(), 1e-6)
        self.assertTrue(torch.allclose(tail, reference, atol=1e-6))

    def test_update_boundary_and_window_helpers_cover_tail(self):
        self.assertEqual([pretrain._window_size(i, 3, 2) for i in range(3)], [2, 2, 1])
        self.assertEqual(
            [pretrain._should_update(i, 3, 2) for i in range(1, 4)],
            [False, True, True],
        )

    def test_padding_rank_still_runs_two_forwards(self):
        class CountingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))
                self.calls = 0

            def forward(self, images, weather):
                self.calls += 1
                return images.flatten(1).mean(1, keepdim=True) * self.weight

        model = CountingModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        args = type("Args", (), {
            "batch_size": 4, "accum_iter": 1, "distributed": False,
            "lr": 0.01, "min_lr": 0.0, "warmup_epochs": 0, "epochs": 1,
        })()
        inner = [(torch.ones(4, 3, 2, 2), torch.ones(4, 3, 2, 2), torch.zeros(4, 9))]
        with patch.object(pretrain.sentinel_wrapper, "get_data_loader", return_value=inner):
            pretrain.train_one_epoch(
                model, [(torch.zeros(1, 4, 1, 2, 2, 3), torch.zeros(1, 4, 1, 9))],
                optimizer, torch.device("cpu"), 0, None, args=args
            )
        self.assertEqual(model.calls, 2)

    def test_cpu_gloo_mixed_active_padding_has_equal_collectives(self):
        with tempfile.TemporaryDirectory() as directory:
            init_file = Path(directory) / "ddp-init"
            queue = mp.get_context("spawn").Queue()
            processes = [
                mp.get_context("spawn").Process(
                    target=_gloo_worker,
                    args=(rank, 2, f"file://{init_file}", queue),
                )
                for rank in range(2)
            ]
            for process in processes:
                process.start()
            for process in processes:
                process.join(15)
            self.assertTrue(all(not process.is_alive() for process in processes))
            self.assertTrue(all(process.exitcode == 0 for process in processes))
            results = sorted(queue.get(timeout=3) for _ in processes)
        self.assertEqual([result[1] for result in results], [4, 4])
        self.assertEqual([result[2] for result in results], [2, 2])
        self.assertTrue(torch.isfinite(torch.tensor(results[0][3])).all())
        self.assertTrue(torch.allclose(torch.tensor(results[0][3]), torch.tensor(results[1][3])))
        # The mixed slot must match a single-rank update on the active batch.
        image = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 0.]]).view(4, 3, 1, 1)
        from loss.contrastive_loss import ContrastiveLoss
        reference = torch.nn.Linear(3, 4)
        with torch.no_grad():
            reference.weight.copy_(torch.eye(4, 3))
            reference.bias.zero_()
        before = reference.weight.detach().clone()
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
        for first, second in ((image, image.roll(1, 0)), (image * 2, image.roll(2, 0))):
            loss = ContrastiveLoss(4, "cpu")(reference(first.flatten(1)), reference(second.flatten(1)))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        self.assertGreater(torch.linalg.vector_norm(before - reference.weight).item(), 1e-6)
        self.assertTrue(torch.allclose(torch.tensor(results[0][3]), before - reference.weight, atol=1e-6))

    def test_empty_county_padding_has_chw_and_weather_sequence(self):
        class ShapeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(()))
                self.calls = 0

            def forward(self, images, weather):
                self.calls += 1
                assert images.shape == (2, 3, 8, 8)
                assert weather.shape == (2, 28, 9)
                return images.mean(dim=(1, 2, 3)).unsqueeze(1) * self.weight

        county = [(torch.zeros(1, 1, 1, 8, 8, 3), torch.zeros(1, 1, 1, 28, 9))]
        first, second, weather = pretrain._synthetic_batch(*county[0])
        model = ShapeModel()
        model(first, weather)
        model(second, weather)
        self.assertEqual(model.calls, 2)

    def test_total_active_slots_aligns_same_position_across_ranks(self):
        args = type("Args", (), {"batch_size": 4, "accum_iter": 4})()
        records = [{"FIPS": "00001", "grid_count": 2}, {"FIPS": "00002", "grid_count": 2}]
        rank0_sampler = [0]
        rank1_sampler = [1]
        with patch.object(pretrain, "_distributed_max", return_value=3) as distributed_max:
            total = pretrain._set_total_active_slots(
                args, records, sampler=rank0_sampler, device=torch.device("cpu"), distributed=True
            )
        self.assertEqual(total, 3)
        self.assertEqual(distributed_max.call_count, 1)
        self.assertEqual([call.args[0] for call in distributed_max.call_args_list], [3])

        with patch.object(pretrain, "_distributed_max", return_value=3):
            rank1_total = pretrain._set_total_active_slots(
                args, records, sampler=rank1_sampler, device=torch.device("cpu"), distributed=True
            )
        self.assertEqual(rank1_total, 3)

    def test_total_active_slots_maxes_by_sampler_position_and_tail(self):
        args = type("Args", (), {"batch_size": 4, "accum_iter": 4})()
        records = [{"FIPS": "00001", "grid_count": 2}, {"FIPS": "00002", "grid_count": 0}]
        with patch.object(pretrain, "_record_grid_count", return_value=0), \
             patch.object(pretrain, "_distributed_max", side_effect=[3, 3]) as distributed_max:
            total = pretrain._set_total_active_slots(
                args, records, sampler=[0, 1], device=torch.device("cpu"), distributed=True
            )
        self.assertEqual(total, 6)
        self.assertEqual(math.ceil(total / args.accum_iter), 2)
        self.assertEqual([call.args[0] for call in distributed_max.call_args_list], [3, 0])

    def test_synthetic_padding_is_finite_and_zero_backward(self):
        images = torch.full((1, 1, 1, 2, 2, 3), 255, dtype=torch.uint8)
        weather = torch.zeros(1, 1, 1, 4, 9)
        first, second, weather_batch = pretrain._synthetic_batch(images, weather)
        self.assertTrue(torch.isfinite(first).all())
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)
        self.assertTrue(torch.isfinite(weather_batch).all())
        parameter = torch.nn.Parameter(torch.ones(()))
        zero_loss = (first.sum() + second.sum() + weather_batch.sum()) * parameter * 0.0
        zero_loss.backward()
        self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertEqual(float(parameter.grad), 0.0)

    def test_epoch_stream_does_not_retain_previous_county(self):
        iterations = []

        class Stream:
            def __iter__(self):
                iterations.append(1)
                for _ in range(3):
                    images = torch.zeros(1, 1, 1, 2, 2, 3)
                    yield images, torch.zeros(1, 1, 1, 1, 9)

        model = torch.nn.Linear(3, 4)
        class FakeModel(torch.nn.Module):
            def forward(self, images, weather):
                return model(images.flatten(1)[:, :3])
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        args = type("Args", (), {"batch_size": 4, "accum_iter": 2, "distributed": False,
                                   "lr": 0.01, "min_lr": 0., "warmup_epochs": 0, "epochs": 1,
                                   "total_active_slots": 1})()
        with patch.object(pretrain.sentinel_wrapper, "get_data_loader", return_value=[]):
            pretrain.train_one_epoch(FakeModel(), Stream(), optimizer, torch.device("cpu"), 0, None, args=args)
        self.assertEqual(iterations, [1])

    def test_small_batch_uses_safe_batch_two_for_model_forward(self):
        class CountingModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.parameter = torch.nn.Parameter(torch.ones(()))
                self.batch_sizes = []

            def forward(self, images, weather):
                self.batch_sizes.append(images.shape[0])
                return images.flatten(1).mean(1, keepdim=True) * self.parameter

        model = CountingModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        args = type("Args", (), {
            "batch_size": 4, "accum_iter": 1, "distributed": False,
            "lr": 0.01, "min_lr": 0.0, "warmup_epochs": 0, "epochs": 1,
        })()
        county = [(torch.zeros(1, 1, 1, 2, 2, 3), torch.zeros(1, 1, 1, 9))]
        inner = [(torch.ones(1, 3, 2, 2), torch.ones(1, 3, 2, 2), torch.zeros(1, 9))]
        with patch.object(pretrain.sentinel_wrapper, "get_data_loader", return_value=inner):
            stats = pretrain.train_one_epoch(
                model, county, optimizer, torch.device("cpu"), 0, None, args=args
            )
        self.assertEqual(stats["real_microbatches"], 0)
        self.assertEqual(stats["optimizer_steps"], 0)
        self.assertEqual(model.batch_sizes, [2, 2])
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_zero_sync_loss_is_connected_to_model_parameters(self):
        model = torch.nn.Linear(3, 2)
        loss = pretrain._zero_sync_loss(model)
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        self.assertTrue(all(torch.count_nonzero(parameter.grad) == 0 for parameter in model.parameters()))

    def test_protocol_uses_exact_four_states(self):
        self.assertEqual(current_data.FOUR_STATES, frozenset({
            "illinois", "iowa", "louisiana", "mississippi",
        }))

    def test_validate_records_rejects_new_york(self):
        records = [{"state": "new_york", "year": 2021, "crop": "corn", "production": 1, "yield_per_acre": 1}]
        with self.assertRaisesRegex(ValueError, "outside the four-state protocol"):
            current_data.validate_records(records, {2021}, require_targets=True)

    def test_validate_pretrain_records_requires_all_four_states(self):
        records = [
            {"state": state, "year": 2020, "crop": "corn"}
            for state in ("illinois", "iowa", "louisiana")
        ]
        with self.assertRaisesRegex(ValueError, "missing states.*mississippi"):
            current_data.validate_pretrain_records(records)

    def test_validate_pretrain_records_requires_all_four_years(self):
        records = [
            {"state": state, "year": year, "crop": "corn"}
            for state in current_data.FOUR_STATES
            for year in (2017, 2018, 2019)
        ]
        with self.assertRaisesRegex(ValueError, "missing years.*2020"):
            current_data.validate_pretrain_records(records)

    def test_supervised_dataset_returns_official_two_log_targets(self):
        record = {
            "FIPS": "19001", "year": 2021, "state": "iowa",
            "production": 1000.0, "yield_per_acre": 200.0,
            "data": {"sentinel": []},
        }
        dataset = current_data.CurrentMMSTDataset([record], train=True, official_protocol=True)
        with patch.object(dataset, "_load_images", return_value=torch.zeros(6, 1, 224, 224, 3)), \
             patch("current_data.load_short_term", return_value=torch.zeros(6, 1, 28, 9)), \
             patch("current_data.load_long_term", return_value=torch.zeros(5, 12, 9)):
            _, _, _, target, _ = dataset[0]
        self.assertEqual(target.shape, (2,))
        self.assertTrue(torch.allclose(target.exp(), torch.tensor([1000.0, 200.0])))

    def test_supervised_dataset_default_keeps_legacy_six_item_scalar_contract(self):
        record = {
            "FIPS": "19001", "year": 2021, "state": "iowa",
            "production": 1000.0, "yield_per_acre": 200.0,
            "data": {"sentinel": []},
        }
        dataset = current_data.CurrentMMSTDataset([record], train=True)
        with patch.object(dataset, "_load_images", return_value=torch.zeros(6, 1, 224, 224, 3)), \
             patch.object(dataset, "_load_coords", return_value=torch.zeros(1, 2)), \
             patch("current_data.load_short_term", return_value=torch.zeros(6, 1, 28, 9)), \
             patch("current_data.load_long_term", return_value=torch.zeros(5, 12, 9)):
            sample = dataset[0]
        self.assertEqual(len(sample), 6)
        self.assertEqual(sample[3].shape, (1,))
        self.assertAlmostEqual(float(sample[3].exp()), 200.0, places=4)
        self.assertEqual(sample[5].shape, (1, 2))

    def test_manifest_record_contains_production_and_yield(self):
        metadata = [{"Year": 2021, "State": " Iowa ", "FIPS": "19001", "County": "adair", "yield_per_acre": 200.0}]
        with tempfile.TemporaryDirectory() as directory, \
             patch("current_data.load_metadata", return_value=metadata), \
             patch("current_data.has_complete_inputs", return_value=True), \
             patch("current_data.load_corn_targets", return_value={"19001": (1000.0, 200.0)}):
            path = Path(directory) / "manifest.json"
            records = current_data.build_manifest([2021], path, require_long_term=True, require_targets=True)
        self.assertEqual(records[0]["production"], 1000.0)
        self.assertEqual(records[0]["yield_per_acre"], 200.0)
        self.assertEqual(records[0]["state"], "iowa")
        self.assertEqual(records[0]["crop"], "corn")

    def test_pretrain_manifest_can_write_crop_without_targets(self):
        metadata = [{"Year": 2020, "State": " Iowa ", "FIPS": "19001", "County": "adair"}]
        with tempfile.TemporaryDirectory() as directory, \
             patch("current_data.load_metadata", return_value=metadata), \
             patch("current_data.has_complete_inputs", return_value=True):
            path = Path(directory) / "manifest.json"
            records = current_data.build_manifest([2020], path, require_long_term=False, require_targets=False)
        self.assertEqual(records[0]["crop"], "corn")
        self.assertNotIn("production", records[0])
