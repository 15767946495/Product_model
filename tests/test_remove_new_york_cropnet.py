import contextlib
import io
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
from pandas.testing import assert_frame_equal

from train_dataset.remove_new_york_cropnet import (
    CleanupPlan,
    CleanupTarget,
    CleanupExecutionError,
    build_cleanup_plan,
    execute_cleanup,
    remove_new_york_rows,
)


class RemoveNewYorkCropNetTests(unittest.TestCase):
    def test_dry_run_changes_nothing_and_apply_removes_only_new_york(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ny_weather = root / "weather/2021/NY"
            ny_ag = root / "sentinel/source/AG/2021/NY"
            other_weather = root / "weather/2021/IA"
            ny_weather.mkdir(parents=True)
            ny_ag.mkdir(parents=True)
            other_weather.mkdir(parents=True)
            (ny_weather / "weather.csv").write_text("x")
            (ny_ag / "image.h5").write_text("x")
            (other_weather / "weather.csv").write_text("keep")
            csv_path = root / "label/hf/USDA_Corn_County_2021.csv"
            csv_path.parent.mkdir(parents=True)
            pd.DataFrame({
                "state_name": ["NEW YORK", "IOWA"],
                "state_ansi": ["36", "19"],
                "value": [1, 2],
            }).to_csv(csv_path, index=False)

            plan = build_cleanup_plan(root, [2021])
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                execute_cleanup(plan, apply=False)
            self.assertTrue(ny_weather.exists())
            self.assertTrue(ny_ag.exists())
            self.assertEqual(len(pd.read_csv(csv_path)), 2)
            self.assertIn("files=1", output.getvalue())
            self.assertIn("bytes=1", output.getvalue())

            execute_cleanup(plan, apply=True)
            self.assertFalse(ny_weather.exists())
            self.assertFalse(ny_ag.exists())
            self.assertTrue(other_weather.exists())
            self.assertEqual(pd.read_csv(csv_path)["state_name"].tolist(), ["IOWA"])

    def test_csv_matches_normalized_name_or_zero_padded_ansi(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "USDA_Corn_County_2022.csv"
            pd.DataFrame({
                "state_name": [" new york ", "iowa", "unknown"],
                "state_ansi": ["00", "19", "36"],
                "value": [1, 2, 3],
            }).to_csv(csv_path, index=False)

            self.assertEqual(remove_new_york_rows(csv_path), 2)
            self.assertEqual(pd.read_csv(csv_path)["value"].tolist(), [1, 2, 3])
            self.assertEqual(remove_new_york_rows(csv_path, apply=True), 2)
            self.assertEqual(pd.read_csv(csv_path)["value"].tolist(), [2])

    def test_csv_dry_run_only_reads_and_does_not_prepare_temp_file(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "USDA_Corn_County_2021.csv"
            pd.DataFrame({"state_name": ["NEW YORK", "IOWA"], "value": [1, 2]}).to_csv(csv_path, index=False)
            with mock.patch("train_dataset.remove_new_york_cropnet._prepare_csv") as prepare:
                self.assertEqual(remove_new_york_rows(csv_path, apply=False), 1)
            prepare.assert_not_called()
            self.assertEqual(list(csv_path.parent.glob(".*.tmp")), [])

    def test_plan_reports_all_target_counts_without_following_outside_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weather = root / "weather/2017/NY"
            weather.mkdir(parents=True)
            (weather / "a").write_bytes(b"12")
            (weather / "nested").mkdir()
            (weather / "nested/b").write_bytes(b"345")
            csv_path = root / "label/hf/USDA_Corn_County_2017.csv"
            csv_path.parent.mkdir(parents=True)
            pd.DataFrame({"state_name": ["IOWA"], "value": [1]}).to_csv(csv_path, index=False)

            plan = build_cleanup_plan(root, [2017])
            weather_target = next(target for target in plan.targets if target.kind == "weather")
            self.assertEqual(weather_target.file_count, 2)
            self.assertEqual(weather_target.byte_count, 5)
            self.assertEqual(plan.csv_paths, (csv_path.resolve(),))

    def test_rejects_invalid_years_and_missing_state_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                build_cleanup_plan(root, [2016])
            with self.assertRaises(ValueError):
                build_cleanup_plan(root, [2023])

            csv_path = root / "USDA_Corn_County_2021.csv"
            pd.DataFrame({"value": [1]}).to_csv(csv_path, index=False)
            with self.assertRaises(ValueError):
                remove_new_york_rows(csv_path)

    def test_rejects_targets_outside_data_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "label/hf/USDA_Corn_County_2021.csv"
            csv_path.parent.mkdir(parents=True)
            pd.DataFrame({"state_name": ["IOWA"], "value": [1]}).to_csv(csv_path, index=False)
            with self.assertRaises(ValueError):
                execute_cleanup(
                    build_cleanup_plan(root, [2021]),
                    apply=False,
                    data_root=root / "child",
                )

    def test_rejects_forged_plan_target_outside_data_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "label/hf/USDA_Corn_County_2021.csv"
            csv_path.parent.mkdir(parents=True)
            pd.DataFrame({"state_name": ["IOWA"], "value": [1]}).to_csv(csv_path, index=False)
            plan = build_cleanup_plan(root, [2021])
            forged = CleanupPlan(
                plan.data_root,
                plan.years,
                plan.targets + (CleanupTarget("weather", Path("/tmp/outside"), 0, 0),),
                plan.csv_paths,
            )
            with self.assertRaises(ValueError):
                execute_cleanup(forged)

    def test_rejects_forged_ia_target_even_when_it_is_inside_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "label/hf/USDA_Corn_County_2021.csv"
            csv_path.parent.mkdir(parents=True)
            pd.DataFrame({"state_name": ["IOWA"], "value": [1]}).to_csv(csv_path, index=False)
            plan = build_cleanup_plan(root, [2021])
            forged_target = CleanupTarget("weather", root / "weather/2021/IA", 0, 0)
            forged = CleanupPlan(plan.data_root, plan.years, (forged_target,) + plan.targets[1:], plan.csv_paths)
            with self.assertRaises(ValueError):
                execute_cleanup(forged)

    def test_rejects_forged_root_target(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "label/hf/USDA_Corn_County_2021.csv"
            csv_path.parent.mkdir(parents=True)
            pd.DataFrame({"state_name": ["IOWA"], "value": [1]}).to_csv(csv_path, index=False)
            plan = build_cleanup_plan(root, [2021])
            forged_target = CleanupTarget("weather", root, 0, 0)
            forged = CleanupPlan(plan.data_root, plan.years, (forged_target,) + plan.targets[1:], plan.csv_paths)
            with self.assertRaises(ValueError):
                execute_cleanup(forged)

    def test_rejects_symlinked_ny_directory_and_csv(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root = Path(directory)
            outside_root = Path(outside)
            weather_parent = root / "weather/2021"
            weather_parent.mkdir(parents=True)
            (outside_root / "NY").mkdir()
            (weather_parent / "NY").symlink_to(outside_root / "NY", target_is_directory=True)
            csv_path = root / "label/hf/USDA_Corn_County_2021.csv"
            csv_path.parent.mkdir(parents=True)
            real_csv = outside_root / "USDA_Corn_County_2021.csv"
            pd.DataFrame({"state_name": ["IOWA"], "value": [1]}).to_csv(real_csv, index=False)
            csv_path.symlink_to(real_csv)
            with self.assertRaises(ValueError):
                build_cleanup_plan(root, [2021])

    def test_rejects_symlinked_data_root(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as outside:
            root_link = Path(directory) / "DataSrc"
            root_link.symlink_to(Path(outside), target_is_directory=True)
            with self.assertRaises(ValueError):
                build_cleanup_plan(root_link, [2021])

    def test_execute_revalidates_plan_after_audit_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "label/hf/USDA_Corn_County_2021.csv"
            csv_path.parent.mkdir(parents=True)
            pd.DataFrame({"state_name": ["NEW YORK", "IOWA"], "value": [1, 2]}).to_csv(csv_path, index=False)
            plan = build_cleanup_plan(root, [2021])
            csv_path.write_text(csv_path.read_text() + "\n")
            with self.assertRaises(ValueError):
                execute_cleanup(plan, apply=True)
            self.assertEqual(len(pd.read_csv(csv_path)), 2)

    def test_directory_audit_drift_aborts_before_any_csv_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ny_weather = root / "weather/2021/NY"
            ny_weather.mkdir(parents=True)
            (ny_weather / "original").write_text("before")
            csv_path = root / "label/hf/USDA_Corn_County_2021.csv"
            csv_path.parent.mkdir(parents=True)
            pd.DataFrame({"state_name": ["NEW YORK", "IOWA"], "value": [1, 2]}).to_csv(csv_path, index=False)
            original_csv = csv_path.read_bytes()
            plan = build_cleanup_plan(root, [2021])
            (ny_weather / "drift").write_text("after")
            with self.assertRaises(ValueError):
                execute_cleanup(plan, apply=True)
            self.assertEqual(csv_path.read_bytes(), original_csv)
            self.assertTrue(ny_weather.exists())

    def test_all_csvs_prepare_before_any_replace_and_cleanup_on_prepare_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_paths = []
            for year in (2021, 2022):
                csv_path = root / f"label/hf/USDA_Corn_County_{year}.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"state_name": ["NEW YORK", "IOWA"], "value": [1, 2]}).to_csv(csv_path, index=False)
                csv_paths.append(csv_path)
            plan = build_cleanup_plan(root, [2021, 2022])
            originals = [path.read_bytes() for path in csv_paths]
            real_to_csv = pd.DataFrame.to_csv
            calls = 0

            def fail_on_second_prepare(frame, *args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("second prepare failed")
                return real_to_csv(frame, *args, **kwargs)

            with mock.patch.object(pd.DataFrame, "to_csv", autospec=True, side_effect=fail_on_second_prepare):
                with self.assertRaises(OSError):
                    execute_cleanup(plan, apply=True)
            self.assertEqual([path.read_bytes() for path in csv_paths], originals)
            self.assertEqual(list(csv_paths[0].parent.glob(".*.tmp")), [])

    def test_commit_failure_reports_completed_csvs_and_cleans_remaining_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_paths = []
            for year in (2021, 2022):
                csv_path = root / f"label/hf/USDA_Corn_County_{year}.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame({"state_name": ["NEW YORK", "IOWA"], "value": [1, 2]}).to_csv(csv_path, index=False)
                csv_paths.append(csv_path)
            plan = build_cleanup_plan(root, [2021, 2022])
            real_replace = os.replace
            calls = 0

            def fail_on_second_replace(source, destination):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("second replace failed")
                return real_replace(source, destination)

            with mock.patch("train_dataset.remove_new_york_cropnet.os.replace", side_effect=fail_on_second_replace):
                with self.assertRaises(CleanupExecutionError) as error:
                    execute_cleanup(plan, apply=True)
            self.assertEqual(error.exception.completed_csvs, (csv_paths[0],))
            self.assertEqual(list(csv_paths[0].parent.glob(".*.tmp")), [])
            self.assertEqual(pd.read_csv(csv_paths[0])["state_name"].tolist(), ["IOWA"])
            self.assertEqual(pd.read_csv(csv_paths[1])["state_name"].tolist(), ["NEW YORK", "IOWA"])

    def test_apply_preserves_columns_and_non_new_york_values_with_nan(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "USDA_Corn_County_2021.csv"
            source = pd.DataFrame({
                "value": [None, 2, 3],
                "state_name": ["NEW YORK", "IOWA", "OHIO"],
                "state_ansi": ["36", "19", "39"],
            })
            source.to_csv(csv_path, index=False)
            expected = source.loc[[1, 2]].copy()
            remove_new_york_rows(csv_path, apply=True)
            actual = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
            expected = pd.read_csv(io.StringIO(expected.to_csv(index=False)), dtype=str, keep_default_na=False)
            assert_frame_equal(actual, expected, check_dtype=True)

    def test_csv_apply_preserves_permissions_until_replace(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "USDA_Corn_County_2021.csv"
            pd.DataFrame({"state_name": ["NEW YORK", "IOWA"], "value": [1, 2]}).to_csv(csv_path, index=False)
            os.chmod(csv_path, 0o640)
            remove_new_york_rows(csv_path, apply=True)
            self.assertEqual(stat.S_IMODE(csv_path.stat().st_mode), 0o640)

    def test_csv_apply_keeps_original_when_preparation_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "USDA_Corn_County_2021.csv"
            pd.DataFrame({"state_name": ["NEW YORK", "IOWA"], "value": [1, 2]}).to_csv(csv_path, index=False)
            original = csv_path.read_bytes()
            with mock.patch.object(pd.DataFrame, "to_csv", side_effect=OSError("prepare failed")):
                with self.assertRaises(OSError):
                    remove_new_york_rows(csv_path, apply=True)
            self.assertEqual(csv_path.read_bytes(), original)

    def test_csv_apply_keeps_original_when_readback_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "USDA_Corn_County_2021.csv"
            pd.DataFrame({"state_name": ["NEW YORK", "IOWA"], "value": [1, 2]}).to_csv(csv_path, index=False)
            original = csv_path.read_bytes()
            real_read_csv = pd.read_csv

            def fail_for_temporary(path, *args, **kwargs):
                if Path(path).name.startswith(f".{csv_path.name}."):
                    raise OSError("readback failed")
                return real_read_csv(path, *args, **kwargs)

            with mock.patch.object(pd, "read_csv", side_effect=fail_for_temporary):
                with self.assertRaises(OSError):
                    remove_new_york_rows(csv_path, apply=True)
            self.assertEqual(csv_path.read_bytes(), original)

    def test_does_not_remove_approximate_state_values(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "USDA_Corn_County_2021.csv"
            pd.DataFrame({
                "state_name": ["NEW YORK STATE", "NEW YORKER", "NEW YORK"],
                "state_ansi": ["36.1", "136", "36"],
                "value": [1, 2, 3],
            }).to_csv(csv_path, index=False)
            self.assertEqual(remove_new_york_rows(csv_path), 1)

    def test_reports_completed_work_when_directory_removal_is_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "weather/2021/NY"
            second = root / "sentinel/source/AG/2021/NY"
            first.mkdir(parents=True)
            second.mkdir(parents=True)
            (first / "a").write_text("a")
            (second / "b").write_text("b")
            csv_path = root / "label/hf/USDA_Corn_County_2021.csv"
            csv_path.parent.mkdir(parents=True)
            pd.DataFrame({"state_name": ["NEW YORK", "IOWA"], "value": [1, 2]}).to_csv(csv_path, index=False)
            plan = build_cleanup_plan(root, [2021])
            real_rmtree = __import__("train_dataset.remove_new_york_cropnet", fromlist=["shutil"]).shutil.rmtree
            calls = 0

            def interrupt(path):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("directory removal failed")
                return real_rmtree(path)

            with mock.patch("train_dataset.remove_new_york_cropnet.shutil.rmtree", side_effect=interrupt):
                with self.assertRaises(CleanupExecutionError) as error:
                    execute_cleanup(plan, apply=True)
            self.assertEqual(error.exception.completed_csvs, (csv_path,))
            self.assertIn("completed_directories=", str(error.exception))
            self.assertIn(str(first), str(error.exception))
            self.assertTrue(second.exists())


if __name__ == "__main__":
    unittest.main()
