import unittest

import numpy as np

from train_dataset.dataset import match_grid_coordinates, stable_grid_order


class GridAlignmentTests(unittest.TestCase):
    def test_same_coordinates_different_order(self):
        weather = np.array([[41.0, -94.0], [42.0, -95.0]])
        images = np.array([[42.0, -95.0], [41.0, -94.0]])
        self.assertEqual(match_grid_coordinates(weather, images), [1, 0])

    def test_within_tolerance_matches(self):
        weather = np.array([[41.0, -94.0], [42.0, -95.0]])
        images = np.array([[41.0005, -94.0005], [42.0, -95.0]])
        self.assertEqual(match_grid_coordinates(weather, images), [0, 1])

    def test_beyond_tolerance_rejected(self):
        weather = np.array([[41.0, -94.0], [42.0, -95.0]])
        images = np.array([[41.002, -94.0], [42.0, -95.0]])
        with self.assertRaisesRegex(ValueError, "no image grid"):
            match_grid_coordinates(weather, images)

    def test_two_weather_grids_matching_same_image_rejected(self):
        weather = np.array([[41.0, -94.0], [41.0006, -94.0]])
        images = np.array([[41.0003, -94.0], [45.0, -90.0]])
        with self.assertRaisesRegex(ValueError, "both match image grid 0"):
            match_grid_coordinates(weather, images)

    def test_weather_grid_with_two_candidates_rejected(self):
        weather = np.array([[41.0, -94.0], [45.0, -90.0]])
        images = np.array([[41.0004, -94.0], [40.9996, -94.0]])
        with self.assertRaisesRegex(ValueError, "multiple image grids"):
            match_grid_coordinates(weather, images)

    def test_count_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "grid count mismatch"):
            match_grid_coordinates(np.array([[41.0, -94.0]]), np.array([[41.0, -94.0], [42.0, -95.0]]))

    def test_non_finite_and_bad_shape_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-finite"):
            match_grid_coordinates(np.array([[np.nan, -94.0]]), np.array([[41.0, -94.0]]))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            match_grid_coordinates(np.array([[41.0, -94.0]]), np.array([[41.0, np.inf]]))
        with self.assertRaisesRegex(ValueError, r"\(G,2\)"):
            match_grid_coordinates(np.array([41.0, -94.0]), np.array([[41.0, -94.0]]))

    def test_stable_order_is_input_order_independent(self):
        coords = np.array([[42.0, -95.0], [41.0, -94.0], [41.0, -95.0], [42.0, -94.0]])
        expected = coords[stable_grid_order(coords)].tolist()
        self.assertEqual(expected, [[41.0, -95.0], [41.0, -94.0], [42.0, -95.0], [42.0, -94.0]])
        rng = np.random.default_rng(0)
        for _ in range(5):
            shuffled = coords[rng.permutation(len(coords))]
            self.assertEqual(shuffled[stable_grid_order(shuffled)].tolist(), expected)

    def test_stable_order_rejects_duplicates(self):
        with self.assertRaisesRegex(ValueError, "duplicate grid"):
            stable_grid_order(np.array([[41.0, -94.0], [41.0002, -94.0]]))


if __name__ == "__main__":
    unittest.main()
