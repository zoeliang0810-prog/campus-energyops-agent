"""Apply user-reviewed meter labels without overwriting raw source metadata."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


KEY_COLUMNS = ("ssuid", "chno")
REVIEW_COLUMNS = (
    "canonical_usersn",
    "canonical_building_or_zone",
    "canonical_room_group",
    "canonical_meter_label",
    "canonical_remark",
    "aggregation_group",
    "aggregation_action",
    "review_note",
    "evidence_basis",
    "review_status",
)


def apply_reviewed_meter_labels(
    inventory: pd.DataFrame,
    reviewed_labels: pd.DataFrame | str | Path,
) -> pd.DataFrame:
    """Return inventory plus canonical fields from a reviewed ssuid+chno overlay."""
    source = inventory.copy()
    review = (
        pd.read_csv(reviewed_labels, dtype=str).fillna("")
        if isinstance(reviewed_labels, (str, Path))
        else reviewed_labels.copy().fillna("")
    )
    required = set(KEY_COLUMNS) | set(REVIEW_COLUMNS)
    missing = required - set(review.columns)
    if missing:
        raise ValueError(f"reviewed labels missing columns: {sorted(missing)}")

    for frame in (source, review):
        for field in KEY_COLUMNS:
            frame[field] = frame[field].astype(str).str.strip()
    duplicate_review = review.duplicated(list(KEY_COLUMNS), keep=False)
    if duplicate_review.any():
        duplicate_keys = review.loc[duplicate_review, list(KEY_COLUMNS)].drop_duplicates()
        raise ValueError(
            "duplicate reviewed ssuid + chno rows: "
            + ", ".join(
                f"{row.ssuid}:{row.chno}" for row in duplicate_keys.itertuples(index=False)
            )
        )

    inventory_keys = set(map(tuple, source[list(KEY_COLUMNS)].itertuples(index=False, name=None)))
    review_keys = set(map(tuple, review[list(KEY_COLUMNS)].itertuples(index=False, name=None)))
    unknown = sorted(review_keys - inventory_keys)
    if unknown:
        raise ValueError(
            "reviewed meters not found in inventory: "
            + ", ".join(f"{ssuid}:{chno}" for ssuid, chno in unknown[:10])
        )

    source = source.drop(columns=list(REVIEW_COLUMNS), errors="ignore")
    source["canonical_usersn"] = (
        source["usersn"].fillna("").astype(str) if "usersn" in source else ""
    )
    source["canonical_building_or_zone"] = (
        source["zone_code"].fillna("").astype(str) if "zone_code" in source else ""
    )
    source["canonical_room_group"] = ""
    source["canonical_meter_label"] = source["canonical_usersn"]
    source["canonical_remark"] = (
        source["remark_primary"].fillna("").astype(str)
        if "remark_primary" in source
        else ""
    )
    source["aggregation_group"] = ""
    source["aggregation_action"] = "unreviewed"
    source["review_note"] = ""
    source["evidence_basis"] = ""
    source["review_status"] = "unreviewed"

    source_index = pd.MultiIndex.from_frame(source[list(KEY_COLUMNS)])
    review_indexed = review.set_index(list(KEY_COLUMNS))
    for field in REVIEW_COLUMNS:
        mapped = pd.Series(source_index.map(review_indexed[field]), index=source.index)
        has_review = mapped.notna()
        source.loc[has_review, field] = mapped.loc[has_review].astype(str)
    return source
