from __future__ import annotations

import pandas as pd
import pytest

from energyops.reviewed_metadata import apply_reviewed_meter_labels


REVIEW_COLUMNS = {
    "ssuid": ["meter-1"],
    "chno": ["1"],
    "canonical_usersn": ["BUILDING-B-L3-M1"],
    "canonical_building_or_zone": ["BUILDING-B"],
    "canonical_room_group": ["LEVEL-3"],
    "canonical_meter_label": ["BUILDING-B-L3-M1"],
    "canonical_remark": ["PUBLIC-SAMPLE"],
    "aggregation_group": ["BUILDING-B-L3-M1"],
    "aggregation_action": ["keep_separate"],
    "review_note": ["reviewed"],
    "evidence_basis": ["user_confirmed"],
    "review_status": ["approved"],
}


def inventory() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ssuid": ["meter-1", "meter-2"],
            "chno": ["1", "1"],
            "usersn": ["BUILDING-A-L3-M1", "BUILDING-A-L3-M4"],
            "zone_code": ["BUILDING-A", "BUILDING-A"],
        }
    )


def test_review_overlay_preserves_raw_and_adds_canonical_fields() -> None:
    result = apply_reviewed_meter_labels(inventory(), pd.DataFrame(REVIEW_COLUMNS))
    first = result.set_index("ssuid").loc["meter-1"]
    second = result.set_index("ssuid").loc["meter-2"]
    assert first["usersn"] == "BUILDING-A-L3-M1"
    assert first["canonical_usersn"] == "BUILDING-B-L3-M1"
    assert first["canonical_remark"] == "PUBLIC-SAMPLE"
    assert first["review_status"] == "approved"
    assert second["canonical_usersn"] == "BUILDING-A-L3-M4"
    assert second["review_status"] == "unreviewed"


def test_review_overlay_rejects_unknown_meter() -> None:
    review = pd.DataFrame(REVIEW_COLUMNS).assign(ssuid="missing")
    with pytest.raises(ValueError, match="not found in inventory"):
        apply_reviewed_meter_labels(inventory(), review)


def test_review_overlay_rejects_duplicate_key() -> None:
    review = pd.concat([pd.DataFrame(REVIEW_COLUMNS)] * 2, ignore_index=True)
    with pytest.raises(ValueError, match="duplicate reviewed"):
        apply_reviewed_meter_labels(inventory(), review)
