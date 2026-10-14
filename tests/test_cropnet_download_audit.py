import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from DataSrc import download_cropnet as dl
from train_dataset.dataset import STATES, YEARS


def target_paths():
    paths = [f"USDA Crop Dataset/Corn/{y}/USDA_Corn_County_{y}.csv" for y in YEARS]
    for y in YEARS:
        for ansi, (_, s) in STATES.items():
            paths += [f"WRF-HRRR Computed Dataset/data/{y}/{s}/HRRR_{ansi}_{s}_{y}-{m:02d}.csv" for m in range(1, 13)]
            paths += [f"Sentinel-2 Imagery/data/AG/{y}/{s}/Agriculture_{ansi}_{s}_{y}-{q}.h5"
                      for q in (f"04-01_{y}-06-30", f"07-01_{y}-09-30")]
    return paths


def audit_row(fips, year, **overrides):
    row = {
        "fips": fips, "year": year, "state": "iowa", "county": "x", "status": "", "reason": "",
        "usda_status": "ok", "weather_days": 168, "weather_grids": 3, "remote_months": 6,
        "remote_grids": 3, "soil_status": "ok", "grid_match_status": "ok",
        "long_term_status": "ok" if year >= 2021 else "not_required", "grid_stability_status": "ok",
    }
    row.update(overrides)
    return row


ADJACENCY = (
    '"Adair County, IA"\t19001\t"Adair County, IA"\t19001\n'
    '\t\t"Adams County, IA"\t19003\n'
    '\t\t"Cass County, IA"\t19029\n'
    '"Adams County, IA"\t19003\t"Adams County, IA"\t19003\n'
    '\t\t"Adair County, IA"\t19001\n'
    '\t\t"Page County, IA"\t19145\n'
    '"Cass County, IA"\t19029\t"Cass County, IA"\t19029\n'
)


class DownloadAuditTests(unittest.TestCase):
    def test_inventory_filters_target_scope(self):
        noise = [
            "README.md", "USDA Crop Dataset/Corn/2016/USDA_Corn_County_2016.csv",
            "USDA Crop Dataset/Soybean/2021/USDA_Soybean_County_2021.csv",
            "WRF-HRRR Computed Dataset/data/2021/LA/HRRR_22_LA_2021-04.csv",
            "WRF-HRRR Computed Dataset/data/2018/MO/.DS_Store",
            "Sentinel-2 Imagery/data/AG/2021/IA/Agriculture_19_IA_2021-01-01_2021-03-31.h5",
            "Sentinel-2 Imagery/data/NDVI/2021/IA/Vegetation_19_IA_2021-04-01_2021-06-30.h5",
            "Sentinel-2 Imagery/data/AG/2023/IA/Agriculture_19_IA_2023-04-01_2023-06-30.h5",
        ]
        inventory = dl.parse_hf_inventory(target_paths() + noise)
        self.assertEqual(inventory.missing(), [])
        self.assertEqual(inventory.paths(), sorted(target_paths()))
        self.assertEqual(len(inventory.paths()), 6 + 6 * 9 * 14)

    def test_inventory_reports_missing(self):
        paths = [p for p in target_paths() if "2019/KY/HRRR_21_KY_2019-03" not in p and "AG/2022/MN" not in p]
        missing = dl.parse_hf_inventory(paths).missing()
        self.assertIn("hrrr 2019 KY 03", missing)
        self.assertIn("sentinel 2022 MN Q2", missing)
        self.assertIn("sentinel 2022 MN Q3", missing)

    def test_local_layout_mapping(self):
        self.assertEqual(
            dl.local_relpath("Sentinel-2 Imagery/data/AG/2021/IA/Agriculture_19_IA_2021-04-01_2021-06-30.h5"),
            "Sentinel-2 Imagery/data/2021/IA/Agriculture_19_IA_2021-04-01_2021-06-30.h5")
        self.assertEqual(dl.local_relpath("USDA Crop Dataset/Corn/2021/USDA_Corn_County_2021.csv"),
                         "USDA Crop Dataset/data/Corn/2021/USDA_Corn_County_2021.csv")
        with self.assertRaises(ValueError):
            dl.local_relpath("README.md")

    def test_default_root_is_raid(self):
        self.assertEqual(dl.DATA_ROOT, Path("/data/raid0/hqx/DataSrc"))

    def test_county_adjacency_is_symmetric_without_self_edges(self):
        graph = dl.parse_county_adjacency(ADJACENCY)
        self.assertEqual(graph["19001"], {"19003", "19029"})
        self.assertEqual(graph["19029"], {"19001"})
        for county, neighbors in graph.items():
            self.assertNotIn(county, neighbors)
            for n in neighbors:
                self.assertIn(county, graph[n])
        restricted = dl.parse_county_adjacency(ADJACENCY, {"19001", "19003", "19999"})
        self.assertEqual(restricted, {"19001": {"19003"}, "19003": {"19001"}, "19999": set()})

    def test_adjacency_rejects_malformed(self):
        with self.assertRaises(ValueError):
            dl.parse_county_adjacency('"A"\t1900\t"A"\t1900\n')
        with self.assertRaises(ValueError):
            dl.parse_county_adjacency('"A"\t19001\n')

    def test_complete_counties_requires_all_six_years_and_modalities(self):
        rows = [audit_row("19001", y) for y in YEARS]
        rows += [audit_row("19003", y) for y in YEARS if y != 2019]
        rows += [audit_row("19005", y, remote_months=5 if y == 2020 else 6) for y in YEARS]
        rows += [audit_row("19007", y, soil_status="incomplete") for y in YEARS]
        self.assertEqual(dl.complete_counties(rows, YEARS), {"19001"})
        with self.assertRaises(ValueError):
            dl.complete_counties(rows + [audit_row("19001", 2017)], YEARS)

    def test_audit_is_sorted_and_atomic(self):
        rows = dl.finalize_audit([audit_row("19003", 2018), audit_row("19001", 2020), audit_row("19001", 2017)],
                                 years=(2017, 2018, 2020))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            dl.write_audit(rows, path)
            lines = [json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual([(r["fips"], r["year"]) for r in lines],
                             [("19001", 2017), ("19001", 2020), ("19003", 2018)])
            self.assertEqual({r["status"] for r in lines if r["fips"] == "19003"}, {"excluded"})
            self.assertTrue(lines[2]["reason"])
            before = path.read_text()
            with self.assertRaises(ValueError):
                dl.write_audit([{"fips": "19001", "year": 2017}], path)
            self.assertEqual(path.read_text(), before)
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["audit.jsonl"])

    def test_audit_only_never_downloads(self):
        remote = {p: dl.RemoteFile(p, 10, None, "0" * 40) for p in target_paths()}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(dl, "list_remote", return_value=remote), \
                mock.patch.object(dl, "hf_download") as hf, \
                mock.patch.object(dl, "download_sources") as sources, \
                mock.patch.object(dl, "fetch_adjacency") as adjacency, \
                mock.patch.object(dl, "fetch_soil") as soil, redirect_stdout(io.StringIO()) as out:
            self.assertEqual(dl.main(["--audit-only", "--data-root", tmp]), 0)
            self.assertEqual(list(Path(tmp).iterdir()), [])
        for m in (hf, sources, adjacency, soil):
            m.assert_not_called()
        self.assertEqual(json.loads(out.getvalue())["pending"], len(target_paths()))

    def test_git_blob_digest_matches_git(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x"
            path.write_bytes(b"hello\n")
            # git hash-object 对 "hello\n" 的结果
            self.assertEqual(dl.file_digest(path, "git-sha1"), "ce013625030ba8dba906f756967f9e9ca394464a")

    def test_seed_is_reused_only_when_digest_matches(self):
        remote_path = "USDA Crop Dataset/Corn/2021/USDA_Corn_County_2021.csv"
        content = b"a,b\n1,2\n"
        remote = dl.RemoteFile(remote_path, len(content), hashlib.sha256(content).hexdigest(), "x")
        with tempfile.TemporaryDirectory() as tmp:
            root, seed = Path(tmp) / "root", Path(tmp) / "seed"
            (seed / dl.local_relpath(remote_path)).parent.mkdir(parents=True)
            (seed / dl.local_relpath(remote_path)).write_bytes(content)
            manifest = dl.Manifest(root)
            with mock.patch.object(dl, "hf_download") as hf:
                self.assertEqual(dl.ensure_file(remote, root, manifest, seed_root=seed), "seeded")
                self.assertEqual(dl.ensure_file(remote, root, manifest, seed_root=seed), "verified")
                hf.assert_not_called()
            self.assertEqual((root / dl.local_relpath(remote_path)).read_bytes(), content)

            (seed / dl.local_relpath(remote_path)).write_bytes(b"a,b\n9,9\n")
            (root / dl.local_relpath(remote_path)).unlink()

            def fake_download(path, staging, revision):
                out = staging / "fake.csv"
                out.write_bytes(content)
                return out

            with mock.patch.object(dl, "hf_download", side_effect=fake_download) as hf:
                self.assertEqual(dl.ensure_file(remote, root, dl.Manifest(root), seed_root=seed), "downloaded")
                hf.assert_called_once()
            self.assertFalse((root / ".staging").exists() and any((root / ".staging").iterdir()))

    def test_soil_aggregation_weights_and_rejects_missing(self):
        rows = [
            {"muacres": "100", "comppct_r": "50", "hzdept_r": "0", "hzdepb_r": "30", **{c: "10" for c in dl._SOIL_MAP.values()}},
            {"muacres": "100", "comppct_r": "50", "hzdept_r": "20", "hzdepb_r": "60", **{c: "20" for c in dl._SOIL_MAP.values()}},
        ]
        result = dl.aggregate_soil(rows)
        self.assertAlmostEqual(result["clay_pct"], (30 * 10 + 10 * 20) / 40)
        rows_missing = [dict(r, awc_r=None) for r in rows]
        with self.assertRaisesRegex(ValueError, "awc"):
            dl.aggregate_soil(rows_missing)


if __name__ == "__main__":
    unittest.main()
