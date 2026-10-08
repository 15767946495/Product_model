import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import h5py

from train_dataset.download_official_tiny_cropnet import (
    DEFAULT_LOCAL_DIR,
    DEFAULT_REPO_ID,
    OFFICIAL_TOTAL_BYTES,
    download_snapshot,
    main,
    resolve_official_inventory,
    validate_snapshot,
    write_json,
)


YEARS = range(2017, 2023)
STATES = {"IA": "19", "IL": "17", "LA": "22", "MS": "28"}
QUARTERS = (("01-01", "03-31"), ("04-01", "06-30"), ("07-01", "09-30"), ("10-01", "12-31"))
CROPS = ("Corn", "Cotton", "Soybean", "WinterWheat")
WRF_HEADER = (
    "Year,Month,Day,Daily/Monthly,State,County,FIPS Code,Grid Index,"
    "Lat (llcrnr),Lon (llcrnr),Lat (urcrnr),Lon (urcrnr),"
    "Avg Temperature (K),Max Temperature (K),Min Temperature (K),"
    "Precipitation (kg m**-2),Relative Humidity (%),Wind Gust (m s**-1),"
    "Wind Speed (m s**-1),U Component of Wind (m s**-1),"
    "V Component of Wind (m s**-1),Downward Shortwave Radiation Flux (W m**-2),"
    "Vapor Pressure Deficit (kPa)\n"
)
USDA_COMMON = (
    "commodity_desc,reference_period_desc,year,state_ansi,state_name,county_ansi,"
    "county_name,asd_code,asd_desc,domain_desc,source_desc,agg_level_desc,"
)


def usda_header(crop: str) -> str:
    production = "PRODUCTION, MEASURED IN 480 LB BALES" if crop == "Cotton" else "PRODUCTION, MEASURED IN BU"
    crop_yield = "YIELD, MEASURED IN LB / ACRE" if crop == "Cotton" else "YIELD, MEASURED IN BU / ACRE"
    return USDA_COMMON + f'"{production}","{crop_yield}"\n'


def build_fixture(root: Path) -> dict:
    (root / ".gitattributes").write_text("*.h5 filter=lfs\n", encoding="utf-8")
    sentinel = root / "Sentinel-2 Imagery" / "data"
    weather = root / "WRF-HRRR Computed Dataset" / "data"
    usda = root / "USDA Crop Dataset" / "data"
    for year in YEARS:
        for state, ansi in STATES.items():
            for start, end in QUARTERS:
                path = sentinel / str(year) / state / f"Agriculture_{ansi}_{state}_{year}-{start}_{year}-{end}.h5"
                path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(path, "w") as handle:
                    county = handle.create_group(f"{ansi}001")
                    date = county.create_group(f"{year}-{start}")
                    date.create_dataset("data", data=[[[[1, 2, 3]]]])
                    date.create_dataset("coordinates", data=[[[1.0, 2.0]]])
            for month in range(1, 13):
                path = weather / str(year) / state / f"HRRR_{ansi}_{state}_{year}-{month:02d}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(WRF_HEADER, encoding="utf-8")
        for crop in CROPS:
            path = usda / crop / str(year) / f"USDA_{crop}_County_{year}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(usda_header(crop), encoding="utf-8")
    return inventory_from_fixture(root)


def inventory_from_fixture(root: Path, requested: str = "main", resolved: str = "a" * 40) -> dict:
    files = []
    for path in sorted(item for item in root.rglob("*") if item.is_file() and ".cache/huggingface" not in item.as_posix()):
        files.append({
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "blob_oid": hashlib.sha1(path.read_bytes()).hexdigest(),
        })
    directories = set()
    for item in files:
        parent = Path(item["path"]).parent
        while parent != Path("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return {
        "repo_id": DEFAULT_REPO_ID,
        "requested_revision": requested,
        "resolved_revision": resolved,
        "expected_file_count": len(files),
        "expected_total_bytes": sum(item["size"] for item in files),
        "expected_category_counts": {
            "hdf5": 96,
            "wrf_csv": 288,
            "usda_csv": 24,
            "root_metadata": 1,
        },
        "expected_directories": sorted(directories),
        "files": files,
    }


def validate_fixture(root: Path, inventory: dict) -> dict:
    with mock.patch(
        "train_dataset.download_official_tiny_cropnet.OFFICIAL_TOTAL_BYTES",
        inventory["expected_total_bytes"],
    ):
        return validate_snapshot(root, inventory)


def fingerprints(root: Path) -> dict:
    return {
        path.relative_to(root).as_posix(): (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
        for path in root.rglob("*") if path.is_file()
    }


def official_siblings() -> list:
    siblings = [
        SimpleNamespace(
            rfilename=".gitattributes",
            size=OFFICIAL_TOTAL_BYTES - 408,
            blob_id="blob-root",
        )
    ]
    siblings.extend(
        SimpleNamespace(
            rfilename=f"Sentinel-2 Imagery/data/{index}.h5",
            size=1,
            blob_id=f"hdf5-{index}",
        )
        for index in range(96)
    )
    siblings.extend(
        SimpleNamespace(
            rfilename=f"WRF-HRRR Computed Dataset/data/{index}.csv",
            size=1,
            blob_id=f"wrf-{index}",
        )
        for index in range(288)
    )
    siblings.extend(
        SimpleNamespace(
            rfilename=f"USDA Crop Dataset/data/{index}.csv",
            size=1,
            blob_id=f"usda-{index}",
        )
        for index in range(24)
    )
    return siblings


class OfficialTinyCropNetRestoreTests(unittest.TestCase):
    def test_complete_official_fixture_passes_with_exact_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = build_fixture(root)
            cache = root / ".cache" / "huggingface" / "download" / "partial.incomplete"
            cache.parent.mkdir(parents=True)
            cache.write_bytes(b"ignored cache")

            report = validate_fixture(root, inventory)

            self.assertTrue(report["valid"])
            self.assertEqual(report["requested_revision"], "main")
            self.assertEqual(report["resolved_revision"], "a" * 40)
            self.assertEqual(report["expected_file_count"], 409)
            self.assertEqual(report["actual_file_count"], 409)
            self.assertEqual(report["missing"], [])
            self.assertEqual(report["extra"], [])
            self.assertEqual(report["expected_category_counts"], {"hdf5": 96, "wrf_csv": 288, "usda_csv": 24, "root_metadata": 1})
            self.assertEqual(report["checked_files"], {"hdf5": 96, "wrf_csv": 288, "usda_csv": 24})

    def test_deleting_one_quarter_or_one_month_fails_exact_inventory(self):
        for pattern in ("Sentinel-2 Imagery/**/*.h5", "WRF-HRRR Computed Dataset/**/*.csv"):
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inventory = build_fixture(root)
                deleted = sorted(root.glob(pattern))[-1]
                relative = deleted.relative_to(root).as_posix()
                deleted.unlink()

                report = validate_fixture(root, inventory)

                self.assertFalse(report["valid"])
                self.assertEqual(report["missing"], [relative])
                category = "hdf5" if relative.endswith(".h5") else "wrf_csv"
                count_check = next(item for item in report["checks"] if item["name"] == "official_category_counts")
                self.assertFalse(count_check["categories"][category])

    def test_corrupt_non_first_csv_is_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = build_fixture(root)
            weather = sorted(root.glob("WRF-HRRR Computed Dataset/**/*.csv"))[100]
            usda = sorted(root.glob("USDA Crop Dataset/**/*.csv"))[10]
            weather.write_text("Year,FIPS Code\n", encoding="utf-8")
            usda.write_text("year,state_ansi\n", encoding="utf-8")
            inventory = inventory_from_fixture(root)

            report = validate_fixture(root, inventory)

            self.assertFalse(report["valid"])
            schema_errors = report["schema_errors"]
            self.assertIn(weather.relative_to(root).as_posix(), schema_errors)
            self.assertIn(usda.relative_to(root).as_posix(), schema_errors)

    def test_every_hdf5_requires_county_date_data_and_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = build_fixture(root)
            broken = sorted(root.glob("Sentinel-2 Imagery/**/*.h5"))[-1]
            with h5py.File(broken, "w") as handle:
                handle.create_group("19001").create_group("2017-01-01").create_dataset("data", data=[1])
            inventory = inventory_from_fixture(root)

            report = validate_fixture(root, inventory)

            self.assertFalse(report["valid"])
            self.assertIn(broken.relative_to(root).as_posix(), report["schema_errors"])

    def test_unknown_root_file_fails_but_gitattributes_is_allowed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = build_fixture(root)
            unknown = root / "notes.txt"
            unknown.write_text("unexpected", encoding="utf-8")

            report = validate_fixture(root, inventory)

            self.assertFalse(report["valid"])
            self.assertEqual(report["extra"], ["notes.txt"])

    def test_unknown_empty_top_level_directory_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = build_fixture(root)
            (root / "unexpected-empty").mkdir()

            report = validate_fixture(root, inventory)

            self.assertFalse(report["valid"])
            check = next(item for item in report["checks"] if item["name"] == "top_level_directories")
            self.assertEqual(check["extra"], ["unexpected-empty"])

    def test_only_root_huggingface_cache_is_excluded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = build_fixture(root)
            ignored = root / ".cache" / "huggingface" / "download" / "x"
            ignored.parent.mkdir(parents=True)
            ignored.write_text("cache", encoding="utf-8")
            nested = root / "Sentinel-2 Imagery" / ".cache" / "not-hf-cache"
            nested.parent.mkdir(parents=True)
            nested.write_text("extra", encoding="utf-8")

            report = validate_fixture(root, inventory)

            self.assertEqual(report["extra"], ["Sentinel-2 Imagery/.cache/not-hf-cache"])

    def test_validation_is_read_only_for_mtime_and_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = build_fixture(root)
            before = fingerprints(root)

            validate_fixture(root, inventory)

            self.assertEqual(fingerprints(root), before)

    def test_resolve_inventory_uses_resolved_commit_and_file_metadata(self):
        siblings = official_siblings()
        api = mock.Mock()
        api.dataset_info.return_value = SimpleNamespace(sha="b" * 40, siblings=siblings)

        inventory = resolve_official_inventory(DEFAULT_REPO_ID, "main", api=api)

        api.dataset_info.assert_called_once_with(DEFAULT_REPO_ID, revision="main", files_metadata=True)
        self.assertEqual(inventory["requested_revision"], "main")
        self.assertEqual(inventory["resolved_revision"], "b" * 40)
        sentinel = next(item for item in inventory["files"] if item["path"].endswith("0.h5"))
        self.assertEqual(sentinel["blob_oid"], "hdf5-0")
        self.assertEqual(inventory["expected_file_count"], 409)
        self.assertEqual(inventory["expected_total_bytes"], OFFICIAL_TOTAL_BYTES)

    def test_official_inventory_requires_96_288_and_24_categories(self):
        api = mock.Mock()
        api.dataset_info.return_value = SimpleNamespace(
            sha="b" * 40,
            siblings=[SimpleNamespace(rfilename=".gitattributes", size=1, blob_id="root")],
        )
        with self.assertRaises(ValueError):
            resolve_official_inventory(DEFAULT_REPO_ID, "main", api=api)

    def test_resolve_rejects_wrong_total_count_bytes_or_root_metadata(self):
        base = official_siblings()
        cases = {
            "extra file": base + [SimpleNamespace(rfilename="extra", size=1, blob_id="extra")],
            "wrong bytes": [
                SimpleNamespace(rfilename=item.rfilename, size=item.size, blob_id=item.blob_id)
                for item in base
            ],
            "wrong root metadata": [
                SimpleNamespace(
                    rfilename="README.md",
                    size=base[0].size,
                    blob_id="root",
                )
            ] + base[1:],
        }
        cases["wrong bytes"][0].size -= 1
        for name, siblings in cases.items():
            with self.subTest(name=name):
                api = mock.Mock()
                api.dataset_info.return_value = SimpleNamespace(sha="b" * 40, siblings=siblings)
                with self.assertRaises(ValueError):
                    resolve_official_inventory(DEFAULT_REPO_ID, "main", api=api)

    def test_validate_rejects_forged_inventory_count_size_root_metadata_and_extra_file(self):
        for mutation in ("count", "size", "root", "extra"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                inventory = build_fixture(root)
                if mutation == "count":
                    inventory["expected_file_count"] += 1
                elif mutation == "size":
                    inventory["expected_total_bytes"] += 1
                elif mutation == "root":
                    inventory["expected_category_counts"]["root_metadata"] = 2
                else:
                    inventory["files"].append({"path": "extra", "size": 0, "blob_oid": None})
                with mock.patch(
                    "train_dataset.download_official_tiny_cropnet.OFFICIAL_TOTAL_BYTES",
                    sum(item["size"] for item in inventory["files"]),
                ):
                    report = validate_snapshot(root, inventory)
                self.assertFalse(report["valid"])
                check = next(
                    item for item in report["checks"]
                    if item["name"] == "inventory_self_consistency"
                )
                self.assertFalse(check["ok"])

    def test_validate_reports_malformed_inventory_instead_of_crashing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = build_fixture(root)
            inventory["files"] = "not-a-list"

            report = validate_snapshot(root, inventory)

            self.assertFalse(report["valid"])
            check = next(
                item for item in report["checks"]
                if item["name"] == "inventory_self_consistency"
            )
            self.assertIn("files must be a list", check["errors"])

    def test_unrelated_root_cache_directory_is_not_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = build_fixture(root)
            (root / ".cache").mkdir()

            report = validate_fixture(root, inventory)

            self.assertEqual(report["extra_directories"], [".cache"])

    def test_nested_empty_directory_fails_exact_directory_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = build_fixture(root)
            nested = root / "Sentinel-2 Imagery" / "data" / "2017" / "IA" / "empty"
            nested.mkdir()

            report = validate_fixture(root, inventory)

            self.assertFalse(report["valid"])
            self.assertEqual(report["extra_directories"], [nested.relative_to(root).as_posix()])

    def test_inventory_rejects_files_without_size_instead_of_omitting_them(self):
        api = mock.Mock()
        api.dataset_info.return_value = SimpleNamespace(
            sha="b" * 40,
            siblings=[SimpleNamespace(rfilename="missing-size", size=None, blob_id=None)],
        )
        with self.assertRaisesRegex(ValueError, "缺少 size metadata"):
            resolve_official_inventory(
                DEFAULT_REPO_ID,
                "main",
                api=api,
            )

    @mock.patch("train_dataset.download_official_tiny_cropnet.snapshot_download")
    def test_snapshot_resume_uses_same_local_dir_and_resolved_sha(self, snapshot_download):
        snapshot_download.return_value = str(DEFAULT_LOCAL_DIR)
        resolved = "c" * 40

        first = download_snapshot(DEFAULT_REPO_ID, resolved, DEFAULT_LOCAL_DIR)
        second = download_snapshot(DEFAULT_REPO_ID, resolved, DEFAULT_LOCAL_DIR)

        self.assertEqual(first, DEFAULT_LOCAL_DIR)
        self.assertEqual(second, DEFAULT_LOCAL_DIR)
        expected = mock.call(
            repo_id=DEFAULT_REPO_ID,
            repo_type="dataset",
            revision=resolved,
            local_dir=str(DEFAULT_LOCAL_DIR),
        )
        self.assertEqual(snapshot_download.call_args_list, [expected, expected])

    def test_inventory_json_is_written_outside_data_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "data"
            root.mkdir()
            inventory = build_fixture(root)
            output = parent / "inventory.json"

            write_json(output, inventory)

            self.assertFalse(output.is_relative_to(root))
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["resolved_revision"], "a" * 40)

    @mock.patch("train_dataset.download_official_tiny_cropnet.validate_snapshot")
    @mock.patch("train_dataset.download_official_tiny_cropnet.download_snapshot")
    @mock.patch("train_dataset.download_official_tiny_cropnet.resolve_official_inventory")
    def test_download_cli_resolves_sha_writes_inventory_then_downloads(
        self, resolve_inventory, download, validate
    ):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            local_dir = parent / "snapshot"
            inventory_path = parent / "inventory.json"
            report_path = parent / "report.json"
            inventory = {
                "repo_id": DEFAULT_REPO_ID,
                "requested_revision": "main",
                "resolved_revision": "d" * 40,
                "files": [],
            }
            resolve_inventory.return_value = inventory
            download.return_value = local_dir
            validate.return_value = {"valid": True}

            exit_code = main([
                "download", "--local-dir", str(local_dir),
                "--inventory", str(inventory_path), "--report", str(report_path),
            ])

            self.assertEqual(exit_code, 0)
            resolve_inventory.assert_called_once_with(DEFAULT_REPO_ID, "main")
            download.assert_called_once_with(DEFAULT_REPO_ID, "d" * 40, local_dir)
            self.assertEqual(json.loads(inventory_path.read_text(encoding="utf-8")), inventory)
            self.assertFalse(inventory_path.is_relative_to(local_dir))

    @mock.patch("train_dataset.download_official_tiny_cropnet.resolve_official_inventory")
    def test_validate_cli_uses_supplied_inventory_without_remote_resolution(self, resolve_inventory):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "snapshot"
            root.mkdir()
            inventory = build_fixture(root)
            inventory_path = parent / "inventory.json"
            report_path = parent / "report.json"
            write_json(inventory_path, inventory)
            before = fingerprints(root)

            with mock.patch(
                "train_dataset.download_official_tiny_cropnet.OFFICIAL_TOTAL_BYTES",
                inventory["expected_total_bytes"],
            ):
                exit_code = main([
                    "validate", "--local-dir", str(root),
                    "--inventory", str(inventory_path), "--report", str(report_path),
                ])

            self.assertEqual(exit_code, 0)
            resolve_inventory.assert_not_called()
            self.assertEqual(fingerprints(root), before)


if __name__ == "__main__":
    unittest.main()
