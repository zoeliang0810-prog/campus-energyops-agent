from __future__ import annotations

import json
from pathlib import Path

from scripts.build_stage_f_dashboard_snapshot import build_snapshot, sync_dashboard_data


def test_stage_f_snapshot_preserves_missing_actuals_and_safety_boundary() -> None:
    snapshot = build_snapshot()
    summary = snapshot["queries"]["dispatch_summary"]["rows"][0]
    schedule = snapshot["queries"]["dispatch_schedule"]["rows"]

    assert summary["verificationPassed"] is True
    assert summary["simulationOnly"] is True
    assert summary["executable"] is False
    assert len(schedule) == 24
    assert all(row["actualLoadKw"] is None for row in schedule)
    assert all(row["actualPvKw"] is None for row in schedule)
    assert all(row["actualGridImportKw"] is None for row in schedule)
    assert all(row["actualBatteryEnergyKwh"] is None for row in schedule)


def test_dashboard_sync_preserves_stable_identity_and_build_status(tmp_path: Path) -> None:
    dashboard_path = tmp_path / "data.json"
    dashboard_path.write_text(
        json.dumps(
            {
                "id": "dashboard:stable-id",
                "surface": "dashboard",
                "title": "old",
                "generatedAt": "old",
                "status": "old",
                "buildStatus": "complete",
                "queries": {},
            }
        ),
        encoding="utf-8",
    )

    sync_dashboard_data(build_snapshot(), dashboard_path)
    refreshed = json.loads(dashboard_path.read_text(encoding="utf-8"))

    assert refreshed["id"] == "dashboard:stable-id"
    assert refreshed["buildStatus"] == "complete"
    assert len(refreshed["queries"]["dispatch_schedule"]["rows"]) == 24
