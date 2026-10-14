import calendar
import csv
import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

from train_dataset import build_dataset as bd
from train_dataset.dataset import (
    ADJACENCY_RELPATH, SOIL_FEATURES, SOIL_RELPATH, WEATHER_FEATURES, YEARS,
    hrrr_relpath, sentinel_relpath, usda_relpath, validate_sample,
)

HRRR_HEADER = ["Year", "Month", "Day", "Daily/Monthly", "State", "County", "FIPS Code", "Grid Index",
               "Lat (llcrnr)", "Lon (llcrnr)", "Lat (urcrnr)", "Lon (urcrnr)", *bd.HRRR_COLUMNS]
USDA_HEADER = ["commodity_desc", "reference_period_desc", "year", "state_ansi", "state_name",
               "county_ansi", "county_name", "asd_code", "asd_desc", "domain_desc", "source_desc",
               "agg_level_desc", "PRODUCTION, MEASURED IN BU", bd.YIELD_COLUMN]

COUNTIES = {"19001": [(41.2, -94.6), (41.2, -94.5), (41.3, -94.6)], "19003": [(41.0, -94.7), (41.0, -94.6)]}


def weather_value(lat, month, day, k):
    return 280.0 + k + month + day / 100.0 + lat / 1000.0


def hrrr_rows(year, month, fips, coords, days=None, value=weather_value):
    days = days or range(1, calendar.monthrange(year, month)[1] + 1)
    for gi, (lat, lon) in enumerate(coords):
        for day in days:
            yield [year, month, float(day), "Daily", "IOWA", "X", fips, float(gi),
                   lat - 0.05, lon - 0.05, lat + 0.05, lon + 0.05,
                   *(value(lat, month, day, k) for k in range(11))]
    # 县级月度行（Grid Index 为 N/A）必须被忽略
    yield [year, month, "N/A", "Monthly", "IOWA", "X", fips, "N/A", "N/A", "N/A", "N/A", "N/A", *([1.0] * 11)]


def write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def usda_row(fips, year, value, commodity="CORN"):
    return [commodity, "YEAR", year, fips[:2], "IOWA", fips[2:], f"COUNTY {fips}", "10", "X",
            "TOTAL", "SURVEY", "COUNTY", "1000.0", value]


def write_h5_groups(path, groups, image_size=4):
    """groups: {(fips, date): coords list}；遥感坐标按逆序写入以验证重排。"""
    import h5py

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as handle:
        for (fips, date), coords in groups.items():
            coords = list(reversed(coords))
            group = handle.require_group(f"{fips}/{date}")
            data = np.zeros((len(coords), image_size, image_size, 3), dtype=np.uint8)
            for i in range(len(coords)):
                data[i] = 10 * (i + 1)
            group.create_dataset("data", data=data)
            group.create_dataset("coordinates", data=np.array(
                [[[lat - 0.05, lon - 0.05], [lat + 0.05, lon + 0.05]] for lat, lon in coords]))


def build_source(root, counties=COUNTIES, years=YEARS, skip=(), coords_override=None, soil_skip=()):
    """合成六年源数据树。skip: {(fips, year, kind)}，kind in usda/hrrr/image。"""
    root = Path(root)
    coords_override = coords_override or {}
    for year in years:
        write_csv(root / usda_relpath(year), USDA_HEADER,
                  [usda_row(f, year, 150.0 + year - 2017 + int(f[2:]) / 10)
                   for f in counties if (f, year, "usda") not in skip])
        for month in range(1, 13):
            rows = []
            for fips, coords in counties.items():
                if (fips, year, "hrrr") in skip:
                    continue
                rows.extend(hrrr_rows(year, month, fips, coords_override.get((fips, year), coords)))
            write_csv(root / hrrr_relpath(year, "IA", month), HRRR_HEADER, rows)
        for months in ((4, 5, 6), (7, 8, 9)):
            groups = {}
            for fips, coords in counties.items():
                for month in months:
                    if (fips, year, "image", month) in skip:
                        continue
                    groups[(fips, f"{year}-{month:02d}-01")] = coords
                    groups[(fips, f"{year}-{month:02d}-15")] = coords
            write_h5_groups(root / sentinel_relpath(year, "IA", months[0]), groups)
    write_csv(root / SOIL_RELPATH, ["FIPS", "areasymbol", *SOIL_FEATURES],
              [[f, "IA" + f[2:], *(1.0 + i for i in range(7))] for f in counties if f not in soil_skip])
    adjacency = root / ADJACENCY_RELPATH
    adjacency.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    fips_list = sorted(counties)
    for f in fips_list:
        lines.append(f'"C {f}"\t{f}\t"C {f}"\t{f}')
        for other in fips_list:
            if other != f:
                lines.append(f'\t\t"C {other}"\t{other}')
        lines.append('\t\t"Far County"\t19999')
    adjacency.write_text("\n".join(lines) + "\n", encoding="latin-1")
    return root


def growing_paths(root, year):
    return [root / hrrr_relpath(year, "IA", m) for m in range(4, 10)]


class BuilderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    # ---- Task 4: readers ---------------------------------------------------

    def test_usda_duplicate_conflict_is_rejected(self):
        path = self.root / "usda.csv"
        rows = [usda_row("19001", 2021, "198.5"), usda_row("19001", 2021, "198.5"), usda_row("19003", 2021, "1,201.0")]
        write_csv(path, USDA_HEADER, rows)
        result = bd.read_usda_yields(path, 2021)
        self.assertEqual(result["19001"]["yield_bu_ac"], 198.5)
        self.assertEqual(result["19003"]["yield_bu_ac"], 1201.0)
        write_csv(path, USDA_HEADER, rows + [usda_row("19001", 2021, "180.0")])
        with self.assertRaisesRegex(ValueError, "conflicting USDA rows for 19001_2021"):
            bd.read_usda_yields(path, 2021)

    def test_usda_filters_scope_and_rejects_bad_labels(self):
        path = self.root / "usda.csv"
        write_csv(path, USDA_HEADER, [
            usda_row("19001", 2021, "198.5"), usda_row("19005", 2021, "50", commodity="SOYBEANS"),
            usda_row("22001", 2021, "120"), usda_row("19005", 2020, "150"),
            ["CORN", "YEAR", 2021, "19", "IOWA", "", "OTHER COMBINED COUNTIES", "", "", "", "", "", "", "170"],
        ])
        self.assertEqual(sorted(bd.read_usda_yields(path, 2021)), ["19001"])
        for bad in ("(D)", "0", "nan"):
            write_csv(path, USDA_HEADER, [usda_row("19001", 2021, bad)])
            with self.assertRaises(ValueError):
                bd.read_usda_yields(path, 2021)

    def write_growing(self, year=2021, coords=COUNTIES["19001"], **kwargs):
        for m in range(1, 13):
            write_csv(self.root / hrrr_relpath(year, "IA", m), HRRR_HEADER,
                      list(hrrr_rows(year, m, "19001", coords, **kwargs)))
        return growing_paths(self.root, year)

    def test_daily_weather_cube_shape(self):
        coords = [(41.3, -94.6), (41.2, -94.6)]
        cube = bd.read_daily_weather(self.write_growing(coords=coords), "19001")
        self.assertEqual(len(cube.dates), 168)
        self.assertEqual(cube.coords.shape, (2, 2))
        self.assertEqual(cube.values.shape, (2, 168, 11))
        self.assertEqual(cube.coords[:, 0].tolist(), [41.2, 41.3])  # 稳定排序，不依赖 Grid Index
        self.assertAlmostEqual(cube.values[0, 28, 3], weather_value(41.2, 5, 1, 3))
        self.assertAlmostEqual(cube.values[1, 167, 10], weather_value(41.3, 9, 28, 10))

    def test_daily_weather_negative_cases(self):
        coords = [(41.2, -94.6), (41.3, -94.6)]
        paths = self.write_growing(coords=coords)
        # 缺整个文件
        with self.assertRaisesRegex(ValueError, "source file not found"):
            bd.read_daily_weather(paths + [self.root / "nope.csv"], "19001")
        # 重复日期：把 4 月文件写两次
        with self.assertRaisesRegex(ValueError, "duplicate HRRR date"):
            bd.read_daily_weather(paths + [paths[0]], "19001")
        # 缺一天
        write_csv(paths[1], HRRR_HEADER, list(hrrr_rows(2021, 5, "19001", coords, days=[d for d in range(1, 32) if d != 7])))
        with self.assertRaisesRegex(ValueError, "missing HRRR date 2021-05-07"):
            bd.read_daily_weather(paths, "19001")
        # 同网格坐标漂移（容差内 -> 重复网格；超出容差 -> 缺日期）
        for shift, message in ((0.0004, "duplicate grid"), (0.01, "missing HRRR date")):
            moved = [(41.2 + shift, -94.6), (41.3, -94.6)]
            write_csv(paths[1], HRRR_HEADER, list(hrrr_rows(2021, 5, "19001", moved)))
            with self.assertRaisesRegex(ValueError, message):
                bd.read_daily_weather(paths, "19001")
        # 非有限值
        bad = lambda lat, m, d, k: float("nan") if (d == 3 and k == 4) else 1.0  # noqa: E731
        write_csv(paths[1], HRRR_HEADER, list(hrrr_rows(2021, 5, "19001", coords, value=bad)))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            bd.read_daily_weather(paths, "19001")
        # 缺变量列
        bd._hrrr_table.cache_clear()
        write_csv(paths[1], HRRR_HEADER[:-1], [r[:-1] for r in hrrr_rows(2021, 5, "19001", coords)])
        with self.assertRaisesRegex(ValueError, "missing columns"):
            bd.read_daily_weather(paths, "19001")

    def test_daily_weather_rejects_non_integer_year_with_source_context(self):
        paths = self.write_growing()
        write_csv(paths[0], HRRR_HEADER, list(hrrr_rows(2021.5, 4, "19001", COUNTIES["19001"], days=range(1, 29))))
        bd._hrrr_table.cache_clear()

        with self.assertRaisesRegex(
            ValueError,
            rf"19001: {paths[0]}: HRRR Year must be a finite integer, got 2021\.5",
        ):
            bd.read_daily_weather(paths, "19001")

    def test_daily_weather_rejects_non_integer_month_with_source_context(self):
        paths = self.write_growing()
        write_csv(paths[0], HRRR_HEADER, list(hrrr_rows(2021, 4.5, "19001", COUNTIES["19001"], days=range(1, 29))))
        bd._hrrr_table.cache_clear()

        with self.assertRaisesRegex(
            ValueError,
            rf"19001: {paths[0]}: HRRR Month must be a finite integer, got 4\.5",
        ):
            bd.read_daily_weather(paths, "19001")

    def test_daily_weather_rejects_non_integer_day_with_source_context(self):
        paths = self.write_growing()
        write_csv(paths[0], HRRR_HEADER, list(hrrr_rows(2021, 4, "19001", COUNTIES["19001"], days=[1.5, *range(2, 29)])))
        bd._hrrr_table.cache_clear()

        with self.assertRaisesRegex(
            ValueError,
            rf"19001: {paths[0]}: HRRR Day must be a finite integer, got 1\.5",
        ):
            bd.read_daily_weather(paths, "19001")

    def test_monthly_weather_cube_shape(self):
        coords = [(41.3, -94.6), (41.2, -94.6)]
        self.write_growing(coords=coords)
        paths = [self.root / hrrr_relpath(2021, "IA", m) for m in range(1, 13)]
        cube = bd.read_monthly_weather(paths, "19001")
        self.assertEqual(cube.values.shape, (2, 12, 11))
        days = calendar.monthrange(2021, 2)[1]
        expected = np.mean([weather_value(41.2, 2, d, 0) for d in range(1, days + 1)])
        self.assertAlmostEqual(cube.values[0, 1, 0], expected)
        with self.assertRaisesRegex(ValueError, "incomplete HRRR month 12"):
            bd.read_monthly_weather(paths[:-1], "19001")
        with self.assertRaisesRegex(ValueError, "duplicate HRRR day"):
            bd.read_monthly_weather(paths + [paths[3]], "19001")
        write_csv(paths[6], HRRR_HEADER, list(hrrr_rows(2021, 7, "19001", [(41.31, -94.6), (41.2, -94.6)])))
        with self.assertRaises(ValueError):
            bd.read_monthly_weather(paths, "19001")

    def test_monthly_weather_rejects_out_of_range_month(self):
        self.write_growing()
        paths = [self.root / hrrr_relpath(2021, "IA", m) for m in range(1, 13)]
        write_csv(paths[0], HRRR_HEADER, list(hrrr_rows(2021, 13, "19001", COUNTIES["19001"], days=[1])))
        bd._hrrr_table.cache_clear()

        with self.assertRaisesRegex(ValueError, "19001_2021: invalid HRRR month"):
            bd.read_monthly_weather(paths, "19001")

    def test_image_index_selects_month_start_only(self):
        coords = COUNTIES["19001"]
        q2 = self.root / sentinel_relpath(2021, "IA", 4)
        q3 = self.root / sentinel_relpath(2021, "IA", 7)
        groups = lambda months: {("19001", f"2021-{m:02d}-{d}"): coords for m in months for d in ("01", "15")}  # noqa: E731
        write_h5_groups(q2, groups((4, 5, 6)))
        write_h5_groups(q3, groups((7, 8, 9)))
        index = bd.read_image_index([q2, q3], "19001", 2021, source_root=self.root)
        self.assertEqual(sorted(index), [4, 5, 6, 7, 8, 9])
        self.assertEqual([index[m].date_group for m in range(4, 10)],
                         [f"2021-{m:02d}-01" for m in range(4, 10)])
        self.assertEqual(index[4].hdf5_path, sentinel_relpath(2021, "IA", 4))
        self.assertEqual(index[7].fips_group, "19001")
        np.testing.assert_allclose(index[4].coords, np.array(list(reversed(coords))))

        q3.unlink()
        bd._h5_meta.cache_clear()
        write_h5_groups(q3, {("19001", "2021-07-15"): coords,
                             **{k: v for k, v in groups((8, 9)).items()}})
        with self.assertRaisesRegex(ValueError, "missing month-start image 2021-07-01"):
            bd.read_image_index([q2, q3], "19001", 2021, source_root=self.root)
        with self.assertRaisesRegex(ValueError, "not under source root"):
            bd.read_image_index([q2, q3], "19001", 2021, source_root=self.root / "elsewhere")

    def test_image_index_rejects_month_with_missing_grid(self):
        coords = COUNTIES["19001"]
        q2 = self.root / sentinel_relpath(2021, "IA", 4)
        q3 = self.root / sentinel_relpath(2021, "IA", 7)
        groups = {
            ("19001", f"2021-{month:02d}-01"): coords
            for month in range(4, 10)
        }
        groups[("19001", "2021-05-01")] = coords[:-1]
        write_h5_groups(q2, {key: value for key, value in groups.items() if int(key[1][5:7]) < 7})
        write_h5_groups(q3, {key: value for key, value in groups.items() if int(key[1][5:7]) >= 7})

        with self.assertRaisesRegex(
            ValueError,
            r"19001_2021: remote grid set differs in 2021-05 from reference 2021-04: grid count mismatch: 3 != 2",
        ):
            bd.read_image_index([q2, q3], "19001", 2021, source_root=self.root)

    def test_image_index_rejects_month_with_extra_grid(self):
        coords = COUNTIES["19001"]
        q2 = self.root / sentinel_relpath(2021, "IA", 4)
        q3 = self.root / sentinel_relpath(2021, "IA", 7)
        groups = {
            ("19001", f"2021-{month:02d}-01"): coords
            for month in range(4, 10)
        }
        groups[("19001", "2021-08-01")] = [*coords, (41.4, -94.6)]
        write_h5_groups(q2, {key: value for key, value in groups.items() if int(key[1][5:7]) < 7})
        write_h5_groups(q3, {key: value for key, value in groups.items() if int(key[1][5:7]) >= 7})

        with self.assertRaisesRegex(
            ValueError,
            r"19001_2021: remote grid set differs in 2021-08 from reference 2021-04: grid count mismatch: 3 != 4",
        ):
            bd.read_image_index([q2, q3], "19001", 2021, source_root=self.root)

    def test_image_index_rejects_month_with_shifted_grid(self):
        coords = COUNTIES["19001"]
        q2 = self.root / sentinel_relpath(2021, "IA", 4)
        q3 = self.root / sentinel_relpath(2021, "IA", 7)
        groups = {
            ("19001", f"2021-{month:02d}-01"): coords
            for month in range(4, 10)
        }
        groups[("19001", "2021-06-01")] = [(41.202, -94.6), *coords[1:]]
        write_h5_groups(q2, {key: value for key, value in groups.items() if int(key[1][5:7]) < 7})
        write_h5_groups(q3, {key: value for key, value in groups.items() if int(key[1][5:7]) >= 7})

        with self.assertRaisesRegex(
            ValueError,
            r"19001_2021: remote grid set differs in 2021-06 from reference 2021-04: .*41\.202.*no image grid within 0\.001 deg",
        ):
            bd.read_image_index([q2, q3], "19001", 2021, source_root=self.root)

    def test_image_index_accepts_month_grid_order_change(self):
        coords = COUNTIES["19001"]
        q2 = self.root / sentinel_relpath(2021, "IA", 4)
        q3 = self.root / sentinel_relpath(2021, "IA", 7)
        q2_groups = {
            ("19001", f"2021-{month:02d}-01"): coords
            for month in range(4, 7)
        }
        q3_groups = {
            ("19001", f"2021-{month:02d}-01"): list(reversed(coords)) if month == 8 else coords
            for month in range(7, 10)
        }
        write_h5_groups(q2, q2_groups)
        write_h5_groups(q3, q3_groups)

        index = bd.read_image_index([q2, q3], "19001", 2021, source_root=self.root)

        np.testing.assert_allclose(index[4].coords, np.array(list(reversed(coords))))
        np.testing.assert_allclose(index[8].coords, np.array(coords))

    def test_soil_requires_all_seven_features(self):
        path = self.root / "soil.csv"
        good = ["19001", "IA001", *range(7)]
        write_csv(path, ["FIPS", "areasymbol", *SOIL_FEATURES, "extra"], [good + ["x"], good + ["y"]])
        self.assertEqual(bd.read_soil(path)["19001"], {n: float(i) for i, n in enumerate(SOIL_FEATURES)})
        write_csv(path, ["FIPS", *SOIL_FEATURES[:-1]], [["19001", *range(6)]])
        with self.assertRaisesRegex(ValueError, "missing soil columns"):
            bd.read_soil(path)
        write_csv(path, ["FIPS", *SOIL_FEATURES], [["19001", *range(6), "nan"]])
        with self.assertRaisesRegex(ValueError, "not finite"):
            bd.read_soil(path)
        write_csv(path, ["FIPS", *SOIL_FEATURES], [["19001", *range(7)], ["19001", *range(1, 8)]])
        with self.assertRaisesRegex(ValueError, "conflicting soil rows"):
            bd.read_soil(path)

    # ---- Task 5: county-year samples -------------------------------------

    SOIL = {n: 1.0 + i for i, n in enumerate(SOIL_FEATURES)}

    def test_builds_valid_2020_sample(self):
        build_source(self.root)
        sample = bd.build_county_year_sample(self.root, "19001", 2020, self.SOIL, ["19003"])
        validate_sample(sample, check_files=True, data_root=self.root)
        self.assertEqual(sample["sample_id"], "19001_2020")
        coords = [(g["coordinates"]["lat"], g["coordinates"]["lon"]) for g in sample["grids"]]
        self.assertEqual(coords, sorted(coords))
        self.assertEqual(len(sample["grids"]), 3)
        for g in sample["grids"]:
            self.assertEqual(len(g["remote_sensing"]), 6)
            self.assertIsNone(g["long_term_weather"])
        # 遥感坐标逆序写入，网格匹配必须按坐标重排而非数组顺序
        self.assertEqual([g["remote_sensing"][0]["grid_index"] for g in sample["grids"]], [2, 1, 0])
        self.assertAlmostEqual(sample["grids"][0]["weather"]["values"][0][3], weather_value(41.2, 4, 1, 3))
        self.assertAlmostEqual(sample["yield_bu_ac"], 150.0 + 3 + 0.1)
        self.assertNotIn("/data/raid0", json.dumps(sample))

    def test_builds_2021_long_term_weather(self):
        build_source(self.root)
        sample = bd.build_county_year_sample(self.root, "19001", 2021, self.SOIL, [])
        lt = sample["grids"][0]["long_term_weather"]
        self.assertEqual(lt["years"], [2017, 2018, 2019, 2020])
        self.assertEqual(lt["months"], list(range(1, 13)))
        self.assertEqual(lt["features"], list(WEATHER_FEATURES))
        self.assertEqual(np.asarray(lt["values"]).shape, (4, 12, 11))
        days = calendar.monthrange(2017, 2)[1]
        expected = np.mean([weather_value(41.2, 2, d, 0) for d in range(1, days + 1)])
        self.assertAlmostEqual(lt["values"][0][1][0], expected, places=5)
        later = bd.build_county_year_sample(self.root, "19001", 2022, self.SOIL, [])
        self.assertEqual(later["grids"][0]["long_term_weather"]["years"], [2018, 2019, 2020, 2021])

    def test_grid_set_must_be_stable_across_history(self):
        fewer = {("19001", 2019): COUNTIES["19001"][:2]}
        build_source(self.root, coords_override=fewer)
        with self.assertRaisesRegex(ValueError, r"19001_2021 history year 2019.*\[\[41\.2"):
            bd.build_county_year_sample(self.root, "19001", 2021, self.SOIL, [])
        rows = {r["year"]: r for r in bd.audit_sources(self.root, {"19001"})}
        self.assertEqual(rows[2019]["grid_stability_status"], "failed")
        self.assertEqual(rows[2021]["long_term_status"], "failed")

    def test_grid_drift_beyond_tolerance_fails(self):
        drift = {("19001", 2019): [(41.2 + 0.01, -94.6), *COUNTIES["19001"][1:]]}
        with tempfile.TemporaryDirectory() as tmp:
            build_source(tmp, coords_override=drift)
            with self.assertRaisesRegex(ValueError, "19001_2022 history year 2019"):
                bd.build_county_year_sample(Path(tmp), "19001", 2022, self.SOIL, [])

    # ---- Task 6: full build ----------------------------------------------

    THREE = {**COUNTIES, "19005": [(40.8, -94.6)]}

    def read_lines(self, path):
        return [json.loads(x) for x in Path(path).read_text().splitlines()]

    def test_full_build_keeps_only_six_year_complete_counties(self):
        build_source(self.root, counties=self.THREE, skip={("19003", 2019, "usda")})
        out = self.root / "out" / "dataset.jsonl"
        summary = bd.build_dataset(self.root, out)
        rows = self.read_lines(out)
        self.assertEqual(summary.counties, 2)
        self.assertEqual(summary.samples, 12)
        self.assertEqual([(r["year"], r["fips"]) for r in rows],
                         [(y, f) for y in YEARS for f in ("19001", "19005")])
        self.assertEqual(summary.sha256, hashlib.sha256(out.read_bytes()).hexdigest())
        self.assertFalse(out.with_name("dataset.jsonl.tmp").exists())

    def test_ids_and_neighbors_are_stable(self):
        build_source(self.root, counties=self.THREE, skip={("19003", 2019, "usda")})
        out = self.root / "dataset.jsonl"
        bd.build_dataset(self.root, out)
        rows = self.read_lines(out)
        self.assertEqual(len({r["sample_id"] for r in rows}), len(rows))
        fixed = {r["fips"] for r in rows}
        for fips in fixed:
            lists = {tuple(r["neighbor_fips"]) for r in rows if r["fips"] == fips}
            self.assertEqual(len(lists), 1)
            self.assertTrue(set(next(iter(lists))) <= fixed)  # 19003、19999 不得出现
        self.assertEqual(rows[0]["neighbor_fips"], ["19005"])

    def test_audit_reports_missing_month_start(self):
        build_source(self.root, skip={("19003", 2018, "image", 7)})
        rows = {(r["fips"], r["year"]): r for r in bd.audit_sources(self.root)}
        bad = rows[("19003", 2018)]
        self.assertEqual(bad["remote_months"], 5)
        self.assertIn("remote: 19003_2018: missing month-start image 2018-07-01", bad["reason"])
        self.assertEqual(rows[("19001", 2018)]["reason"], "")

    def test_audit_reports_monthly_remote_grid_set_change(self):
        import h5py

        build_source(self.root)
        path = self.root / sentinel_relpath(2018, "IA", 4)
        with h5py.File(path, "r+") as handle:
            group = handle["19003/2018-05-01"]
            data = group["data"][()][:-1]
            coordinates = group["coordinates"][()][:-1]
            del group["data"]
            del group["coordinates"]
            group.create_dataset("data", data=data)
            group.create_dataset("coordinates", data=coordinates)
        bd._h5_meta.cache_clear()

        rows = {(r["fips"], r["year"]): r for r in bd.audit_sources(self.root, {"19003"})}

        bad = rows[("19003", 2018)]
        self.assertEqual(bad["remote_months"], 6)
        self.assertEqual(bad["remote_grids"], 0)
        self.assertIn(
            "remote: 19003_2018: remote grid set differs in 2018-05 from reference 2018-04: "
            "grid count mismatch: 2 != 1",
            bad["reason"],
        )

    def test_failed_build_does_not_replace_existing_dataset(self):
        build_source(self.root)
        out = self.root / "dataset.jsonl"
        out.write_text("OLD\n")
        real = bd.build_county_year_sample

        def flaky(root, fips, year, soil, neighbors):
            if year == 2022:
                raise ValueError("boom")
            return real(root, fips, year, soil, neighbors)

        with mock.patch.object(bd, "build_county_year_sample", side_effect=flaky):
            with self.assertRaisesRegex(ValueError, "boom"):
                bd.build_dataset(self.root, out)
        self.assertEqual(out.read_text(), "OLD\n")
        self.assertFalse(out.with_name("dataset.jsonl.tmp").exists())

    def test_only_fips_still_requires_six_complete_years(self):
        build_source(self.root, skip={("19003", 2020, "hrrr")})
        with self.assertRaisesRegex(ValueError, r"not six-year complete: \['19003'\]"):
            bd.build_dataset(self.root, self.root / "x.jsonl", only_fips={"19003"})
        summary = bd.build_dataset(self.root, self.root / "x.jsonl", only_fips={"19001"})
        self.assertEqual(summary.samples, 6)
        self.assertEqual(self.read_lines(self.root / "x.jsonl")[0]["neighbor_fips"], [])

    def test_validate_only_writes_nothing(self):
        build_source(self.root)
        out = self.root / "nested" / "dataset.jsonl"
        summary = bd.build_dataset(self.root, out, validate_only=True, link=self.root / "link.jsonl")
        self.assertEqual(summary.samples, 12)
        self.assertFalse(summary.written)
        self.assertFalse(out.parent.exists())
        self.assertFalse((self.root / "link.jsonl").exists())

    def test_symlink_is_created_without_overwriting_regular_file(self):
        target = self.root / "raid" / "dataset.jsonl"
        link = self.root / "proj" / "dataset.jsonl"
        bd.ensure_symlink(link, target)
        self.assertEqual(os.readlink(link), str(target))
        bd.ensure_symlink(link, target)  # 已正确，不改
        with self.assertRaisesRegex(FileExistsError, "points to"):
            bd.ensure_symlink(link, self.root / "other.jsonl")
        regular = self.root / "proj" / "regular.jsonl"
        regular.write_text("keep")
        with self.assertRaisesRegex(FileExistsError, "not a symlink"):
            bd.ensure_symlink(regular, target)
        self.assertEqual(regular.read_text(), "keep")

    def test_cli_forwards_arguments(self):
        fake = bd.BuildSummary(1, 6, 1, 1, 10, "x", None, False)
        with mock.patch.object(bd, "build_dataset", return_value=fake) as build, \
                redirect_stdout(io.StringIO()):
            bd.main(["--validate-only", "--only-fips", "19001", "1003",
                     "--output", str(self.root / "o.jsonl"), "--source-root", str(self.root)])
        build.assert_called_once_with(self.root, self.root / "o.jsonl", only_fips={"19001", "01003"},
                                      validate_only=True, link=None)
        with mock.patch.object(bd, "build_dataset", return_value=fake) as build, \
                redirect_stdout(io.StringIO()):
            bd.main([])
        self.assertEqual(build.call_args.kwargs["link"], bd.PROJECT_LINK)
        self.assertEqual(build.call_args.args, (bd.DATA_ROOT, bd.DATASET_PATH))


if __name__ == "__main__":
    unittest.main()
