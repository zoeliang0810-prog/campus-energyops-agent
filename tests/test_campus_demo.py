from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from energyops.campus_demo import build_demo_inputs, build_demo_request
from energyops.contracts import OperatingMode


class CampusDemoTests(unittest.TestCase):
    def test_demo_inputs_are_explicitly_simulation_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pv.csv"
            timestamps = pd.date_range(
                "2026-02-19T00:00:00+08:00", periods=96, freq="15min"
            )
            pd.DataFrame(
                {"timestamp": timestamps, "campus_pv_power_kw": [0.0] * 96}
            ).to_csv(path, index=False)
            inputs = build_demo_inputs(path, dataset_version="test_v1")
            request = build_demo_request(inputs)
            self.assertEqual(len(inputs.points), 96)
            self.assertEqual(request.operating_mode, OperatingMode.SIMULATION_ONLY)
            self.assertTrue(any("synthetic" in item for item in request.assumptions))
            self.assertEqual(request.battery.capacity_kwh, 100.0)

    def test_measured_load_replaces_synthetic_curve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pv_path = root / "pv.csv"
            load_path = root / "load.csv"
            timestamps = pd.date_range(
                "2026-02-19T00:00:00+08:00", periods=96, freq="15min"
            )
            pd.DataFrame(
                {"timestamp": timestamps, "campus_pv_power_kw": [0.0] * 96}
            ).to_csv(pv_path, index=False)
            pd.DataFrame(
                {"timestamp": timestamps, "load_power_kw": [123.0] * 96}
            ).to_csv(load_path, index=False)

            inputs = build_demo_inputs(
                pv_path,
                dataset_version="combined_v1",
                load_csv=load_path,
            )
            request = build_demo_request(inputs, measured_load=True)

            self.assertTrue(all(point.load_power_kw == 123.0 for point in inputs.points))
            self.assertFalse(any("synthetic" in item for item in request.assumptions))

    def test_measured_load_requires_exact_timestamp_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pv_path = root / "pv.csv"
            load_path = root / "load.csv"
            timestamps = pd.date_range(
                "2026-02-19T00:00:00+08:00", periods=4, freq="15min"
            )
            pd.DataFrame(
                {"timestamp": timestamps, "campus_pv_power_kw": [0.0] * 4}
            ).to_csv(pv_path, index=False)
            pd.DataFrame(
                {"timestamp": timestamps[:-1], "load_power_kw": [1.0] * 3}
            ).to_csv(load_path, index=False)

            with self.assertRaisesRegex(ValueError, "timestamps must match"):
                build_demo_inputs(
                    pv_path,
                    dataset_version="combined_v1",
                    load_csv=load_path,
                )


if __name__ == "__main__":
    unittest.main()
