import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from train_dataset.dataset import (
    SOIL_FEATURES, WEATHER_FEATURES, long_term_years, remote_months,
    validate_sample, weather_dates,
)


def make_grid(fips, year, index, lat, lon):
    history = long_term_years(year)
    return {
        "grid_id": f"{fips}_grid_{index:04d}",
        "grid_index": index,
        "coordinates": {"lat": lat, "lon": lon},
        "weather": {"values": [[float(d + f) for f in range(11)] for d in range(168)]},
        "remote_sensing": [
            {
                "month": month,
                "hdf5_path": f"Sentinel-2 Imagery/data/{year}/IA/Agriculture_19_IA_{year}-"
                             + ("04-01_%d-06-30.h5" % year if int(month[-2:]) <= 6 else "07-01_%d-09-30.h5" % year),
                "fips_group": fips,
                "date_group": f"{month}-01",
                "grid_index": index,
            }
            for month in remote_months(year)
        ],
        "long_term_weather": None if history is None else {
            "years": history,
            "months": list(range(1, 13)),
            "features": list(WEATHER_FEATURES),
            "values": [[[1.0] * 11 for _ in range(12)] for _ in range(4)],
        },
    }


def make_sample(fips="19001", year=2021, coords=((41.0, -94.5), (41.1, -94.4)),
                neighbors=("19003", "19009"), label=198.5):
    return {
        "schema_version": 1,
        "sample_id": f"{fips}_{year}",
        "fips": fips,
        "year": year,
        "state": "iowa",
        "state_abbr": "IA",
        "county": "adair",
        "yield_bu_ac": label,
        "neighbor_fips": list(neighbors),
        "weather_features": list(WEATHER_FEATURES),
        "weather_dates": weather_dates(year),
        "remote_sensing_months": remote_months(year),
        "soil": {name: 1.0 + i for i, name in enumerate(SOIL_FEATURES)},
        "grids": [make_grid(fips, year, i, lat, lon) for i, (lat, lon) in enumerate(coords)],
    }


def write_h5(root, sample, *, image_size=8, skip_dates=(), fips_override=None):
    """为样本的全部遥感引用写一个最小 HDF5 夹具。"""
    import h5py

    files = {}
    for grid in sample["grids"]:
        for ref in grid["remote_sensing"]:
            files.setdefault(ref["hdf5_path"], set()).add(ref["date_group"])
    corners = np.array([
        [[g["coordinates"]["lat"] - 0.05, g["coordinates"]["lon"] - 0.05],
         [g["coordinates"]["lat"] + 0.05, g["coordinates"]["lon"] + 0.05]]
        for g in sample["grids"]
    ])
    count = len(sample["grids"])
    for rel, dates in files.items():
        path = Path(root) / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "a") as handle:
            for date in sorted(dates):
                if date in skip_dates:
                    continue
                group = handle.require_group(f"{fips_override or sample['fips']}/{date}")
                data = np.full((count, image_size, image_size, 3), 100, dtype=np.uint8)
                for i in range(count):
                    data[i] += i
                group.create_dataset("data", data=data)
                group.create_dataset("coordinates", data=corners)


class CropNetSchemaTests(unittest.TestCase):
    def assertInvalid(self, sample, fragment):
        with self.assertRaises(ValueError) as ctx:
            validate_sample(sample)
        self.assertIn(fragment, str(ctx.exception))

    def test_weather_dates_are_exactly_six_months_of_28_days(self):
        dates = weather_dates(2021)
        self.assertEqual(len(dates), 168)
        self.assertEqual(dates[0], "2021-04-01")
        self.assertEqual(dates[27], "2021-04-28")
        self.assertEqual(dates[28], "2021-05-01")
        self.assertEqual(dates[-1], "2021-09-28")

    def test_long_term_years_protocol(self):
        self.assertIsNone(long_term_years(2020))
        self.assertEqual(long_term_years(2021), [2017, 2018, 2019, 2020])
        self.assertEqual(long_term_years(2022), [2018, 2019, 2020, 2021])

    def test_valid_samples_pass(self):
        validate_sample(make_sample(year=2021))
        validate_sample(make_sample(year=2022))
        validate_sample(make_sample(year=2017))
        validate_sample(make_sample(neighbors=()))

    def test_duplicate_neighbors_rejected(self):
        self.assertInvalid(make_sample(neighbors=("19003", "19003")), "neighbor_fips")

    def test_unsorted_neighbors_rejected(self):
        self.assertInvalid(make_sample(neighbors=("19009", "19003")), "neighbor_fips: must be sorted")

    def test_self_neighbor_rejected(self):
        self.assertInvalid(make_sample(neighbors=("19001", "19003")), "neighbor_fips")

    def test_wrong_weather_shape_rejected(self):
        sample = make_sample()
        sample["grids"][0]["weather"]["values"].pop()
        self.assertInvalid(sample, "19001_2021.grids[0].weather.values: expected shape 168x11")

    def test_non_finite_weather_rejected(self):
        sample = make_sample()
        sample["grids"][1]["weather"]["values"][3][2] = float("nan")
        self.assertInvalid(sample, "grids[1].weather.values: contains non-finite")

    def test_missing_remote_month_rejected(self):
        sample = make_sample()
        sample["grids"][0]["remote_sensing"].pop()
        self.assertInvalid(sample, "grids[0].remote_sensing: expected exactly 6")

    def test_mid_month_image_rejected(self):
        sample = make_sample()
        sample["grids"][0]["remote_sensing"][2]["date_group"] = "2021-06-15"
        self.assertInvalid(sample, "date_group")

    def test_2017_long_term_must_be_null(self):
        sample = make_sample(year=2017)
        sample["grids"][0]["long_term_weather"] = make_sample(year=2021)["grids"][0]["long_term_weather"]
        self.assertInvalid(sample, "grids[0].long_term_weather: must be null for 2017")

    def test_2022_wrong_history_years_rejected(self):
        sample = make_sample(year=2022)
        sample["grids"][0]["long_term_weather"]["years"] = [2017, 2018, 2019, 2020]
        self.assertInvalid(sample, "long_term_weather.years")

    def test_long_term_shape_rejected(self):
        sample = make_sample(year=2021)
        sample["grids"][0]["long_term_weather"]["values"].pop()
        self.assertInvalid(sample, "long_term_weather.values: expected shape 4x12x11")

    def test_unsorted_or_duplicate_grids_rejected(self):
        self.assertInvalid(make_sample(coords=((41.1, -94.4), (41.0, -94.5))), "sorted")
        self.assertInvalid(make_sample(coords=((41.0, -94.5), (41.0, -94.5))), "duplicate grid")

    def test_missing_and_extra_fields_rejected(self):
        sample = make_sample()
        del sample["soil"]["awc"]
        self.assertInvalid(sample, "soil.awc: missing field")
        sample = make_sample()
        sample["grid_cache"] = 1
        self.assertInvalid(sample, "unexpected field")

    def test_sample_id_and_label_rejected(self):
        sample = make_sample()
        sample["sample_id"] = "19001_2020"
        self.assertInvalid(sample, "sample_id")
        self.assertInvalid(make_sample(label=-1.0), "yield_bu_ac")

    def test_absolute_hdf5_path_rejected(self):
        sample = make_sample()
        sample["grids"][0]["remote_sensing"][0]["hdf5_path"] = "/data/raid0/hqx/x.h5"
        self.assertInvalid(sample, "hdf5_path")

    def test_hdf5_reference_is_checked(self):
        sample = make_sample()
        with tempfile.TemporaryDirectory() as tmp:
            write_h5(tmp, sample)
            validate_sample(sample, check_files=True, data_root=Path(tmp))

            bad = copy.deepcopy(sample)
            bad["grids"][1]["remote_sensing"][0]["grid_index"] = 5
            with self.assertRaisesRegex(ValueError, "out of range"):
                validate_sample(bad, check_files=True, data_root=Path(tmp))

            swapped = copy.deepcopy(sample)
            swapped["grids"][0]["remote_sensing"][0]["grid_index"] = 1
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_sample(swapped, check_files=True, data_root=Path(tmp))

        with tempfile.TemporaryDirectory() as tmp:
            write_h5(tmp, sample, fips_override="19003")
            with self.assertRaisesRegex(ValueError, "group 19001/2021-04-01 not found"):
                validate_sample(sample, check_files=True, data_root=Path(tmp))

        with tempfile.TemporaryDirectory() as tmp:
            write_h5(tmp, sample, skip_dates=("2021-08-01",))
            with self.assertRaisesRegex(ValueError, "2021-08-01 not found"):
                validate_sample(sample, check_files=True, data_root=Path(tmp))

        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "file not found"):
                validate_sample(sample, check_files=True, data_root=Path(tmp))


if __name__ == "__main__":
    unittest.main()
