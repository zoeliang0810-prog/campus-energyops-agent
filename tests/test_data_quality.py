from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from energyops.contracts import QualityStatus
from energyops.data_quality import (
    canonicalize_campus_pv_window,
    canonicalize_pv_window,
    load_campus_pv_csv,
    load_pv_csv,
    profile_pv_data,
)


class DataQualityTests(unittest.TestCase):
    def test_contiguous_source_resamples_to_15_minutes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pv.csv"
            pd.DataFrame(
                {
                    "time": pd.date_range("2026-02-19", periods=12, freq="5min"),
                    "实时发电功率": range(12),
                }
            ).to_csv(path, index=False)
            frame = load_pv_csv(path)
            quality = profile_pv_data(frame, dataset_version="test")
            self.assertEqual(quality.status, QualityStatus.PASS)
            canonical = canonicalize_pv_window(
                frame,
                start="2026-02-19T00:00:00+08:00",
                end="2026-02-19T01:00:00+08:00",
            )
            self.assertEqual(len(canonical), 4)
            self.assertAlmostEqual(canonical.iloc[0]["pv_power_kw"], 1.0)

    def test_gap_is_reported_as_warning(self):
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2026-02-19 00:00", "2026-02-19 00:05", "2026-02-19 00:15"]
                ),
                "pv_power_kw": [0.0, 0.0, 0.0],
            }
        )
        quality = profile_pv_data(frame, dataset_version="test")
        self.assertEqual(quality.status, QualityStatus.WARNING)
        self.assertEqual(quality.missing_timestamp_count, 1)

    def test_five_devices_are_aggregated_and_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campus.csv"
            data = {"time": pd.date_range("2026-02-19", periods=12, freq="5min")}
            for index in range(1, 6):
                data[f"pv_device_{index}"] = [float(index)] * 12
            pd.DataFrame(data).to_csv(path, index=False)

            source = load_campus_pv_csv(path)
            canonical = canonicalize_campus_pv_window(
                source,
                start="2026-02-19T00:00:00+08:00",
                end="2026-02-19T01:00:00+08:00",
            )

            self.assertEqual(len(canonical), 4)
            self.assertEqual(canonical.iloc[0]["campus_pv_power_kw"], 15.0)
            self.assertEqual(
                canonical.iloc[0][[f"pv_device_{i}" for i in range(1, 6)]].sum(),
                canonical.iloc[0]["campus_pv_power_kw"],
            )

    def test_missing_device_column_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campus.csv"
            pd.DataFrame(
                {
                    "time": ["2026-02-19 00:00"],
                    "pv_device_1": [1.0],
                }
            ).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing required columns"):
                load_campus_pv_csv(path)


if __name__ == "__main__":
    unittest.main()
