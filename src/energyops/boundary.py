"""Human-reviewed campus metering-boundary contract and validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "building_or_zone",
    "selected_ssuid",
    "channel",
    "parent_ssuid",
    "inclusion_reason",
    "reviewer",
    "effective_date",
    "include",
}


@dataclass(frozen=True, slots=True)
class BoundaryValidation:
    passed: bool
    selected_meter_keys: frozenset[tuple[str, str]]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


def validate_boundary_selection(
    boundary_csv: str | Path,
    inventory_csv: str | Path,
) -> BoundaryValidation:
    """Reject unknown, duplicate, unreviewed, or parent/child-overlapping meters."""
    boundary = pd.read_csv(boundary_csv, dtype=str).fillna("")
    inventory = pd.read_csv(inventory_csv, dtype=str).fillna("")
    missing = REQUIRED_COLUMNS - set(boundary.columns)
    if missing:
        return BoundaryValidation(
            passed=False,
            selected_meter_keys=frozenset(),
            errors=(f"missing columns: {sorted(missing)}",),
            warnings=(),
        )

    include = boundary["include"].str.strip().str.lower().isin(
        {"1", "true", "yes", "y", "是"}
    )
    selected = boundary.loc[include].copy()
    errors: list[str] = []
    warnings: list[str] = []
    if selected.empty:
        errors.append("no reviewed boundary meters are selected")

    for field in (
        "building_or_zone",
        "selected_ssuid",
        "channel",
        "inclusion_reason",
        "reviewer",
        "effective_date",
    ):
        if selected[field].str.strip().eq("").any():
            errors.append(f"selected rows require non-empty {field}")
    invalid_dates = []
    for raw in selected["effective_date"]:
        try:
            date.fromisoformat(raw.strip())
        except ValueError:
            invalid_dates.append(raw)
    if invalid_dates:
        errors.append("effective_date must use YYYY-MM-DD")

    keys = list(zip(selected["selected_ssuid"], selected["channel"], strict=True))
    if len(keys) != len(set(keys)):
        errors.append("duplicate selected ssuid + channel rows")

    inventory_keys = set(zip(inventory["ssuid"], inventory["chno"], strict=True))
    unknown = sorted(set(keys) - inventory_keys)
    if unknown:
        errors.append(
            "selected meters not found in inventory: "
            + ", ".join(f"{ssuid}:{channel}" for ssuid, channel in unknown[:10])
        )

    selected_ids = {ssuid for ssuid, _ in keys}
    overlap_rows = selected[
        selected["parent_ssuid"].str.strip().isin(selected_ids)
    ]
    if not overlap_rows.empty:
        errors.append("a selected parent_ssuid and its selected child cannot both be included")

    blank_parent = selected["parent_ssuid"].str.strip().eq("")
    terminal_confirmed = (
        selected["boundary_basis"].str.strip().eq("user_confirmed_terminal_submeter")
        if "boundary_basis" in selected
        else pd.Series(False, index=selected.index)
    )
    if (blank_parent & ~terminal_confirmed).any():
        warnings.append(
            "blank parent_ssuid means the electrical parent relationship still needs reviewer confirmation"
        )
    provisional = inventory[
        inventory.set_index(["ssuid", "chno"]).index.isin(set(keys))
        & inventory["site_name_basis"].str.startswith("provisional")
    ]
    if not provisional.empty:
        warnings.append("selection includes provisionally mapped buildings")

    return BoundaryValidation(
        passed=not errors,
        selected_meter_keys=frozenset(keys),
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
