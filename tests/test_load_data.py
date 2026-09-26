from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from energyops.load_data import (
    canonicalize_campus_load_window,
    classify_end_use,
    classify_milp_role,
    extract_zone_code,
    infer_site_group,
    infer_site_name,
    parse_meter_insert_line,
    scan_meter_dump,
    usersn_prefix3,
)


COLUMNS = (
    "(`ssuid`, `usersn`, `name`, `address`, `remark`, `refreshdt`, `type`, "
    "`errssulist`, `comstate`, `warn`, `remark`, `chno`, `scale`, `value`)"
)


def insert_line(
    timestamp: str,
    value: str,
    *,
    ssuid: str = "meter-1",
    scale: str = "1.00",
    state: str = "通信正常",
    warning: str = "''",
    secondary_remark: str = "''",
) -> bytes:
    return (
        f"INSERT INTO `<table_name>` {COLUMNS} VALUES "
        f"('{ssuid}', 'B1-总表', '教学楼总表', '配电房', '', "
        f"'{timestamp}', '电', '[{{errmsg=, ssuid=}}]', '{state}', "
        f"{warning}, {secondary_remark}, '1', '{scale}', '{value}');\n"
    ).encode()


class LoadDataTests(unittest.TestCase):
    def test_parser_preserves_duplicate_remark_positions_and_nulls(self):
        reading = parse_meter_insert_line(
            insert_line(
                "2026-02-19 01:00:00",
                "10.0",
                warning="NULL",
                secondary_remark="NULL",
            )
        )
        self.assertIsNotNone(reading)
        assert reading is not None
        self.assertEqual(reading.ssuid, "meter-1")
        self.assertIsNone(reading.warning)
        self.assertIsNone(reading.remark_secondary)

    def test_inventory_recovers_meter_metadata_and_categories(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "load.sql"
            path.write_bytes(
                insert_line("2026-02-19 00:00:00", "10.0")
                + insert_line("2026-02-19 01:00:00", "11.0")
            )
            result = scan_meter_dump(path)
            row = result.inventory.iloc[0]
            self.assertEqual(result.total_rows, 2)
            self.assertEqual(result.parse_error_rows, 0)
            self.assertEqual(row["end_use_category"], "general_power")
            self.assertTrue(row["boundary_candidate"])
            self.assertEqual(row["usersn_prefix3"], "B1-")
            prefix_row = result.usersn_prefix3_summary.iloc[0]
            self.assertEqual(prefix_row["usersn_prefix3"], "B1-")
            self.assertEqual(prefix_row["meter_count"], 1)
            self.assertEqual(prefix_row["record_count"], 2)
            self.assertEqual(prefix_row["representative_explanation"], "配电房")
            self.assertEqual(prefix_row["explanation_source"], "address")

    def test_bounded_scan_keeps_lookahead_for_final_interval(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "load.sql"
            path.write_bytes(
                insert_line("2026-02-19 22:00:00", "10.0")
                + insert_line("2026-02-20 01:00:00", "13.0")
                + insert_line("2026-02-20 04:00:00", "16.0")
            )
            result = scan_meter_dump(
                path,
                selected_meter_ids={"meter-1"},
                collect_start=datetime(2026, 2, 19),
                collect_end=datetime(2026, 2, 20),
                lookahead_hours=3,
            )
            timestamps = {reading.timestamp for reading in result.selected_readings}
            self.assertIn(datetime(2026, 2, 20, 1), timestamps)
            self.assertNotIn(datetime(2026, 2, 20, 4), timestamps)

    def test_usersn_prefix3_trims_and_handles_missing_values(self):
        self.assertEqual(usersn_prefix3(" 1A-ZAP "), "1A-")
        self.assertEqual(usersn_prefix3("AB"), "AB")
        self.assertEqual(usersn_prefix3(None), "UNKNOWN")

    def test_cumulative_register_becomes_scaled_fifteen_minute_power(self):
        readings = [
            parse_meter_insert_line(
                insert_line("2026-02-19 00:00:00", "10.0", scale="120.00")
            ),
            parse_meter_insert_line(
                insert_line("2026-02-19 01:00:00", "11.0", scale="120.00")
            ),
        ]
        result = canonicalize_campus_load_window(
            [item for item in readings if item is not None],
            selected_meter_ids={"meter-1"},
            start="2026-02-19T00:00:00+08:00",
            end="2026-02-19T01:00:00+08:00",
        )
        self.assertEqual(len(result.campus), 4)
        self.assertEqual(result.quality["status"], "pass")
        self.assertTrue((result.campus["load_power_kw"] == 120.0).all())

    def test_missing_boundary_meter_fails_quality(self):
        readings = [
            parse_meter_insert_line(insert_line("2026-02-19 00:00:00", "10.0")),
            parse_meter_insert_line(insert_line("2026-02-19 01:00:00", "11.0")),
        ]
        result = canonicalize_campus_load_window(
            [item for item in readings if item is not None],
            selected_meter_ids={"meter-1", "missing-meter"},
            start="2026-02-19T00:00:00+08:00",
            end="2026-02-19T01:00:00+08:00",
        )
        self.assertEqual(result.quality["status"], "fail")
        self.assertEqual(result.quality["incomplete_dispatch_bin_count"], 4)

    def test_implausible_terminal_meter_power_is_rejected(self):
        readings = [
            parse_meter_insert_line(insert_line("2026-02-19 00:00:00", "10.0")),
            parse_meter_insert_line(insert_line("2026-02-19 01:00:00", "20010.0")),
        ]
        result = canonicalize_campus_load_window(
            [item for item in readings if item is not None],
            selected_meter_ids={"meter-1"},
            start="2026-02-19T00:00:00+08:00",
            end="2026-02-19T01:00:00+08:00",
            max_meter_power_kw=10_000.0,
        )
        self.assertEqual(
            result.quality["rejected_interval_counts"]["implausible_meter_power"], 1
        )
        self.assertEqual(result.quality["status"], "fail")

    def test_load_classifier_finds_end_uses(self):
        self.assertEqual(classify_end_use("BLD1 景观照明"), "lighting")
        self.assertEqual(classify_end_use("宿舍空调"), "hvac")

    def test_public_building_mapping_stays_generic(self):
        self.assertEqual(extract_zone_code("BLD12-METER-1"), "BLD12")
        self.assertEqual(
            infer_site_group("BLD12"),
            ("unknown", "insufficient_semantic_metadata"),
        )
        self.assertEqual(
            infer_site_name("BLD12"),
            ("", "insufficient_semantic_metadata"),
        )

    def test_milp_roles_are_conservative_candidates(self):
        self.assertEqual(
            classify_milp_role("fire_protection", "消防风机")[0],
            "critical_locked",
        )
        self.assertEqual(
            classify_milp_role("hvac", "教学楼空调")[0],
            "flexible_candidate",
        )
        self.assertEqual(
            classify_milp_role("lighting", "普通照明")[0],
            "fixed_base",
        )


if __name__ == "__main__":
    unittest.main()
