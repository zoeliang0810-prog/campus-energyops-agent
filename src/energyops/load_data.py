"""Streaming ingestion for cumulative electricity-meter SQL exports.

The source dump stores cumulative register readings rather than interval power.
This module keeps inventory discovery separate from campus aggregation because
adding parent meters and their submeters would double-count campus load.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


INSERT_MARKER = b") VALUES ("
NORMAL_COMMUNICATION_STATE = "通信正常"


@dataclass(frozen=True, slots=True)
class MeterReading:
    ssuid: str
    usersn: str | None
    name: str | None
    address: str | None
    remark_primary: str | None
    timestamp: datetime
    energy_type: str
    error_list: str | None
    communication_state: str | None
    warning: str | None
    remark_secondary: str | None
    channel: str
    scale: float
    value: float


@dataclass(slots=True)
class _MeterSummary:
    ssuid: bytes
    channel: bytes
    usersn: bytes | None = None
    name: bytes | None = None
    address: bytes | None = None
    remark_primary: bytes | None = None
    remark_secondary: bytes | None = None
    metadata_timestamp: bytes | None = None
    first_timestamp: bytes | None = None
    last_timestamp: bytes | None = None
    record_count: int = 0
    valid_reading_count: int = 0
    negative_reading_count: int = 0
    communication_states: Counter[bytes] = field(default_factory=Counter)
    scales: set[bytes] = field(default_factory=set)

    def observe(self, values: Sequence[bytes | None]) -> None:
        timestamp = values[5]
        assert timestamp is not None
        self.record_count += 1
        if self.first_timestamp is None or timestamp < self.first_timestamp:
            self.first_timestamp = timestamp
        if self.last_timestamp is None or timestamp > self.last_timestamp:
            self.last_timestamp = timestamp

        state = values[8]
        self.communication_states[state or b"NULL"] += 1
        value = _ascii_float(values[13])
        if value is not None and value < 0:
            self.negative_reading_count += 1
        if state == NORMAL_COMMUNICATION_STATE.encode() and value is not None and value >= 0:
            self.valid_reading_count += 1

        if values[12] is not None:
            self.scales.add(values[12])

        metadata = values[1], values[2], values[3], values[4], values[10]
        has_metadata = any(item not in (None, b"") for item in metadata)
        if has_metadata and (
            self.metadata_timestamp is None or timestamp >= self.metadata_timestamp
        ):
            self.metadata_timestamp = timestamp
            if values[1] not in (None, b""):
                self.usersn = values[1]
            if values[2] not in (None, b""):
                self.name = values[2]
            if values[3] not in (None, b""):
                self.address = values[3]
            if values[4] not in (None, b""):
                self.remark_primary = values[4]
            if values[10] not in (None, b""):
                self.remark_secondary = values[10]


@dataclass(frozen=True, slots=True)
class DumpScanResult:
    inventory: pd.DataFrame
    usersn_prefix3_summary: pd.DataFrame
    site_mapping: pd.DataFrame
    category_summary: pd.DataFrame
    milp_summary: pd.DataFrame
    selected_readings: tuple[MeterReading, ...]
    source_sha256: str
    total_rows: int
    parse_error_rows: int
    earliest_timestamp: datetime | None
    latest_timestamp: datetime | None


@dataclass(frozen=True, slots=True)
class CanonicalLoadResult:
    campus: pd.DataFrame
    meter_intervals: pd.DataFrame
    quality: dict[str, object]


END_USE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("charging", ("充电桩", "充电站", "充电")),
    ("hvac", ("空调", "风机盘管", "冷机", "制冷", "热泵", "新风", "暖通", "ahu", "vrv")),
    ("lighting", ("照明", "景观灯", "路灯", "灯光")),
    ("socket", ("插座",)),
    ("elevator", ("电梯", "扶梯")),
    ("pump", ("水泵", "泵房", "给水", "排水")),
    ("kitchen", ("厨房", "餐厅", "食堂", "后厨")),
    ("fire_protection", ("消防",)),
    ("data_room", ("数据中心", "服务器", "机房")),
    ("general_power", ("动力", "公共区域", "公共用电", "总表", "总配", "进线", "母线")),
)

FACILITY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("laboratory", ("实验室", "实验")),
    ("dormitory", ("宿舍", "公寓")),
    ("commercial", ("商业", "商铺", "餐厅", "食堂")),
    ("teaching", ("教学", "教室")),
    ("office", ("办公",)),
    ("campus_common", ("公共区域", "公共用电", "景观", "道路")),
)

BOUNDARY_KEYWORDS = ("总表", "总配", "进线", "母线", "变压器", "高压", "低压总")

USER_CONFIRMED_SITE_GROUPS: dict[str, str] = {}
USER_CONFIRMED_SITE_NAMES: dict[str, str] = {}
USER_CONFIRMED_SITE_BASIS: dict[str, str] = {}
SITE_MAPPING_NOTES: dict[str, str] = {}

# Site-specific building mappings stay in private reviewed data. The public
# implementation keeps the inference API without shipping campus asset labels.
DORMITORY_MIXED_CODES: set[str] = set()
RESIDENTIAL_CANDIDATE_CODES: set[str] = set()
COMMERCIAL_SERVICE_CODES: set[str] = set()
OFFICE_TEACHING_MIXED_CODES: set[str] = set()
BUILDING_SERVICE_CODES: set[str] = set()

CRITICAL_KEYWORDS = ("消防", "应急", "ups", "数据中心", "服务器")
SHIFTABLE_KEYWORDS = ("充电", "洗衣")


def _decode(value: bytes | None) -> str | None:
    if value is None:
        return None
    return value.decode("utf-8", errors="replace").replace("''", "'")


def _ascii_float(value: bytes | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def usersn_prefix3(usersn: str | None) -> str:
    """Return the first three trimmed characters used for user-requested grouping."""
    normalized = (usersn or "").strip()
    return normalized[:3] if normalized else "UNKNOWN"


def _representative_text(values: pd.Series) -> str:
    cleaned = values.fillna("").astype(str).str.strip()
    cleaned = cleaned[cleaned.ne("")]
    if cleaned.empty:
        return ""
    counts = cleaned.value_counts()
    highest_count = counts.max()
    return sorted(counts[counts.eq(highest_count)].index)[0]


def _split_insert_values(line: bytes) -> list[bytes | None] | None:
    position = line.find(INSERT_MARKER)
    if position < 0:
        return None
    payload = line[position + len(INSERT_MARKER) :].rstrip(b"\r\n")
    if not payload.endswith(b");"):
        return None
    payload = payload[:-2]
    if not payload.startswith(b"'") or not payload.endswith(b"'"):
        return None

    # Fast path for this export format. NULL is unquoted, while all other
    # values are quoted and separated by exactly comma + space.
    normalized = payload.replace(b"NULL", b"'NULL'")[1:-1]
    raw_values = normalized.split(b"', '")
    if len(raw_values) != 14:
        return None
    return [None if value == b"NULL" else value for value in raw_values]


def parse_meter_insert_line(line: bytes) -> MeterReading | None:
    """Parse one exported INSERT line, preserving the two remark columns."""
    values = _split_insert_values(line)
    if values is None or values[0] is None or values[5] is None:
        return None
    scale = _ascii_float(values[12])
    reading = _ascii_float(values[13])
    if scale is None or reading is None:
        return None
    return MeterReading(
        ssuid=_decode(values[0]) or "",
        usersn=_decode(values[1]),
        name=_decode(values[2]),
        address=_decode(values[3]),
        remark_primary=_decode(values[4]),
        timestamp=datetime.fromisoformat(values[5].decode("ascii")),
        energy_type=_decode(values[6]) or "",
        error_list=_decode(values[7]),
        communication_state=_decode(values[8]),
        warning=_decode(values[9]),
        remark_secondary=_decode(values[10]),
        channel=_decode(values[11]) or "",
        scale=scale,
        value=reading,
    )


def _classify(text: str, rules: Sequence[tuple[str, Sequence[str]]]) -> str:
    normalized = text.lower()
    for category, keywords in rules:
        if any(keyword in normalized for keyword in keywords):
            return category
    return "unknown"


def classify_end_use(*parts: str | None) -> str:
    return _classify(" ".join(part or "" for part in parts), END_USE_RULES)


def classify_facility(*parts: str | None) -> str:
    return _classify(" ".join(part or "" for part in parts), FACILITY_RULES)


def is_boundary_candidate(*parts: str | None) -> bool:
    text = " ".join(part or "" for part in parts)
    return any(keyword in text for keyword in BOUNDARY_KEYWORDS)


def extract_zone_code(*parts: str | None) -> str:
    """Extract a stable building/distribution prefix from meter metadata."""
    for part in parts:
        normalized = (part or "").strip().upper()
        match = re.match(
            r"^([A-Z]{1,4}\d+|\d+[A-Z]?)",
            normalized,
        )
        if match:
            return match.group(1)
    return "OTHER"


def infer_site_group(zone_code: str) -> tuple[str, str]:
    """Return site group and evidence basis without pretending inferred data is a ledger."""
    if zone_code in USER_CONFIRMED_SITE_GROUPS:
        return USER_CONFIRMED_SITE_GROUPS[zone_code], USER_CONFIRMED_SITE_BASIS[
            zone_code
        ]
    if zone_code in DORMITORY_MIXED_CODES:
        return "dormitory_mixed_use", "inferred_from_room_and_load_labels"
    if zone_code in RESIDENTIAL_CANDIDATE_CODES:
        return "residential_candidate", "low_confidence_code_pattern"
    if zone_code in COMMERCIAL_SERVICE_CODES:
        return "commercial_service", "inferred_from_tenant_and_catering_labels"
    if zone_code in OFFICE_TEACHING_MIXED_CODES:
        return "office_teaching_mixed", "inferred_from_room_labels"
    if zone_code in BUILDING_SERVICE_CODES or zone_code.startswith("SE"):
        return "building_service_unknown", "insufficient_semantic_metadata"
    return "unknown", "insufficient_semantic_metadata"


def infer_site_name(zone_code: str) -> tuple[str, str]:
    if zone_code in USER_CONFIRMED_SITE_NAMES:
        return USER_CONFIRMED_SITE_NAMES[zone_code], USER_CONFIRMED_SITE_BASIS[zone_code]
    return "", "insufficient_semantic_metadata"


def classify_milp_role(end_use: str, *parts: str | None) -> tuple[str, str]:
    """Classify optimization role while keeping all inferred assets uncontrollable."""
    text = " ".join(part or "" for part in parts).lower()
    if end_use in {"fire_protection", "data_room"} or any(
        keyword in text for keyword in CRITICAL_KEYWORDS
    ):
        return "critical_locked", "safety_or_continuity_keyword"
    if end_use == "charging" or any(keyword in text for keyword in SHIFTABLE_KEYWORDS):
        return "shiftable_candidate", "schedule_candidate_without_control_authority"
    if end_use in {"hvac", "pump"}:
        return "flexible_candidate", "power_flexibility_candidate_without_constraints"
    return "fixed_base", "conservative_default"


def scan_meter_dump(
    path: str | Path,
    *,
    selected_meter_ids: set[str] | None = None,
    collect_start: datetime | None = None,
    collect_end: datetime | None = None,
    lookback_hours: int = 48,
    lookahead_hours: int = 3,
) -> DumpScanResult:
    """Stream the dump and retain bounded readings plus edge context.

    Readings after ``collect_end`` are needed to calculate the final dispatch
    interval when a meter reports after midnight rather than exactly at it.
    Canonicalization clips the resulting interval back to the requested window.
    """
    source_path = Path(path)
    selected_bytes = (
        {value.encode("utf-8") for value in selected_meter_ids}
        if selected_meter_ids
        else set()
    )
    collect_lower = (
        (collect_start - timedelta(hours=lookback_hours)).isoformat(sep=" ").encode()
        if collect_start
        else None
    )
    collect_upper = (
        (collect_end + timedelta(hours=lookahead_hours)).isoformat(sep=" ").encode()
        if collect_end
        else None
    )

    digest = hashlib.sha256()
    summaries: dict[tuple[bytes, bytes], _MeterSummary] = {}
    selected_readings: list[MeterReading] = []
    total_rows = 0
    parse_errors = 0
    earliest: bytes | None = None
    latest: bytes | None = None

    with source_path.open("rb") as source:
        for line in source:
            digest.update(line)
            values = _split_insert_values(line)
            if values is None or values[0] is None or values[5] is None or values[11] is None:
                parse_errors += 1
                continue
            total_rows += 1
            timestamp = values[5]
            earliest = timestamp if earliest is None or timestamp < earliest else earliest
            latest = timestamp if latest is None or timestamp > latest else latest

            key = values[0], values[11]
            summary = summaries.get(key)
            if summary is None:
                summary = _MeterSummary(ssuid=values[0], channel=values[11])
                summaries[key] = summary
            summary.observe(values)

            if (
                selected_bytes
                and values[0] in selected_bytes
                and collect_lower is not None
                and collect_upper is not None
                and collect_lower <= timestamp <= collect_upper
            ):
                parsed = parse_meter_insert_line(line)
                if parsed is not None:
                    selected_readings.append(parsed)

    inventory_rows: list[dict[str, object]] = []
    for summary in summaries.values():
        usersn = _decode(summary.usersn)
        name = _decode(summary.name)
        address = _decode(summary.address)
        remark_primary = _decode(summary.remark_primary)
        remark_secondary = _decode(summary.remark_secondary)
        text_parts = usersn, name, address, remark_primary, remark_secondary
        zone_code = extract_zone_code(*text_parts)
        site_group, site_group_basis = infer_site_group(zone_code)
        site_name, site_name_basis = infer_site_name(zone_code)
        end_use_category = classify_end_use(*text_parts)
        milp_role, milp_role_basis = classify_milp_role(
            end_use_category, *text_parts
        )
        state_counts = {
            _decode(state) or "NULL": count
            for state, count in sorted(summary.communication_states.items())
        }
        inventory_rows.append(
            {
                "ssuid": _decode(summary.ssuid),
                "chno": _decode(summary.channel),
                "usersn": usersn,
                "usersn_prefix3": usersn_prefix3(usersn),
                "name": name,
                "address": address,
                "remark_primary": remark_primary,
                "remark_secondary": remark_secondary,
                "zone_code": zone_code,
                "site_group": site_group,
                "site_group_basis": site_group_basis,
                "site_name": site_name,
                "site_name_basis": site_name_basis,
                "end_use_category": end_use_category,
                "facility_category": classify_facility(*text_parts),
                "milp_role": milp_role,
                "milp_role_basis": milp_role_basis,
                "dispatch_controllable": False,
                "boundary_candidate": is_boundary_candidate(*text_parts),
                "scales": ",".join(
                    value.decode("ascii") for value in sorted(summary.scales)
                ),
                "first_timestamp": _decode(summary.first_timestamp),
                "last_timestamp": _decode(summary.last_timestamp),
                "record_count": summary.record_count,
                "valid_reading_count": summary.valid_reading_count,
                "negative_reading_count": summary.negative_reading_count,
                "communication_states": ";".join(
                    f"{state}:{count}" for state, count in state_counts.items()
                ),
            }
        )

    inventory = pd.DataFrame(inventory_rows).sort_values(
        ["boundary_candidate", "end_use_category", "usersn", "ssuid"],
        ascending=[False, True, True, True],
        na_position="last",
    )
    category_summary = (
        inventory.groupby(["end_use_category", "facility_category"], dropna=False)
        .agg(
            meter_count=("ssuid", "count"),
            record_count=("record_count", "sum"),
            valid_reading_count=("valid_reading_count", "sum"),
            boundary_candidate_count=("boundary_candidate", "sum"),
        )
        .reset_index()
        .sort_values(["meter_count", "record_count"], ascending=False)
    )
    prefix_totals = (
        inventory.groupby("usersn_prefix3", dropna=False)
        .agg(
            meter_count=("ssuid", "count"),
            record_count=("record_count", "sum"),
            valid_reading_count=("valid_reading_count", "sum"),
            negative_reading_count=("negative_reading_count", "sum"),
            first_timestamp=("first_timestamp", "min"),
            last_timestamp=("last_timestamp", "max"),
        )
        .reset_index()
    )
    prefix_metadata = (
        inventory.groupby("usersn_prefix3", dropna=False)
        .agg(
            representative_address=("address", _representative_text),
            representative_name=("name", _representative_text),
            representative_remark=("remark_primary", _representative_text),
        )
        .reset_index()
    )
    prefix_metadata["representative_explanation"] = prefix_metadata[
        "representative_address"
    ]
    prefix_metadata["explanation_source"] = "address"
    missing_address = prefix_metadata["representative_explanation"].eq("")
    prefix_metadata.loc[missing_address, "representative_explanation"] = (
        prefix_metadata.loc[missing_address, "representative_name"]
    )
    prefix_metadata.loc[missing_address, "explanation_source"] = "name"
    missing_name = prefix_metadata["representative_explanation"].eq("")
    prefix_metadata.loc[missing_name, "representative_explanation"] = (
        prefix_metadata.loc[missing_name, "representative_remark"]
    )
    prefix_metadata.loc[missing_name, "explanation_source"] = "remark_primary"
    missing_remark = prefix_metadata["representative_explanation"].eq("")
    prefix_metadata.loc[missing_remark, "representative_explanation"] = (
        "仅有编码：" + prefix_metadata.loc[missing_remark, "usersn_prefix3"]
    )
    prefix_metadata.loc[missing_remark, "explanation_source"] = "usersn_prefix3"
    role_counts = (
        inventory.groupby(["usersn_prefix3", "milp_role"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reindex(
            columns=[
                "critical_locked",
                "fixed_base",
                "flexible_candidate",
                "shiftable_candidate",
            ],
            fill_value=0,
        )
        .rename(columns=lambda role: f"{role}_meter_count")
        .reset_index()
    )
    usersn_prefix3_summary = (
        prefix_totals.merge(
            prefix_metadata,
            on="usersn_prefix3",
            how="left",
            validate="one_to_one",
        ).merge(role_counts, on="usersn_prefix3", how="left", validate="one_to_one")
    )
    usersn_prefix3_summary["valid_reading_rate"] = (
        usersn_prefix3_summary["valid_reading_count"]
        / usersn_prefix3_summary["record_count"]
    )
    usersn_prefix3_summary = usersn_prefix3_summary.sort_values(
        ["meter_count", "record_count", "usersn_prefix3"],
        ascending=[False, False, True],
    )
    mapping_counts = (
        inventory.groupby("zone_code", dropna=False)
        .agg(
            current_meter_count=("ssuid", "count"),
            current_record_count=("record_count", "sum"),
        )
        .reset_index()
    )
    site_mapping = pd.DataFrame(
        [
            {
                "zone_code": code,
                "site_name": USER_CONFIRMED_SITE_NAMES[code],
                "site_group": group,
                "mapping_basis": USER_CONFIRMED_SITE_BASIS[code],
                "mapping_note": SITE_MAPPING_NOTES.get(code, ""),
            }
            for code, group in USER_CONFIRMED_SITE_GROUPS.items()
        ],
        columns=[
            "zone_code",
            "site_name",
            "site_group",
            "mapping_basis",
            "mapping_note",
        ],
    ).merge(mapping_counts, on="zone_code", how="left", validate="one_to_one")
    site_mapping[["current_meter_count", "current_record_count"]] = site_mapping[
        ["current_meter_count", "current_record_count"]
    ].fillna(0).astype("int64")
    site_mapping["present_in_dump"] = site_mapping["current_meter_count"].gt(0)
    site_mapping = site_mapping.sort_values(["site_group", "zone_code"])
    milp_summary = (
        inventory.groupby(
            ["site_group", "site_group_basis", "milp_role"], dropna=False
        )
        .agg(
            meter_count=("ssuid", "count"),
            record_count=("record_count", "sum"),
            valid_reading_count=("valid_reading_count", "sum"),
        )
        .reset_index()
        .sort_values(["site_group", "milp_role"])
    )
    return DumpScanResult(
        inventory=inventory.reset_index(drop=True),
        usersn_prefix3_summary=usersn_prefix3_summary.reset_index(drop=True),
        site_mapping=site_mapping.reset_index(drop=True),
        category_summary=category_summary.reset_index(drop=True),
        milp_summary=milp_summary.reset_index(drop=True),
        selected_readings=tuple(selected_readings),
        source_sha256=digest.hexdigest(),
        total_rows=total_rows,
        parse_error_rows=parse_errors,
        earliest_timestamp=(datetime.fromisoformat(earliest.decode()) if earliest else None),
        latest_timestamp=(datetime.fromisoformat(latest.decode()) if latest else None),
    )


def _reading_rank(reading: MeterReading) -> tuple[int, int]:
    return (
        int(reading.communication_state == NORMAL_COMMUNICATION_STATE),
        int(reading.value >= 0),
    )


def _deduplicate_readings(readings: Iterable[MeterReading]) -> tuple[list[MeterReading], int]:
    selected: dict[tuple[str, str, datetime], MeterReading] = {}
    duplicate_count = 0
    for reading in readings:
        key = reading.ssuid, reading.channel, reading.timestamp
        previous = selected.get(key)
        if previous is not None:
            duplicate_count += 1
        if previous is None or _reading_rank(reading) >= _reading_rank(previous):
            selected[key] = reading
    return list(selected.values()), duplicate_count


def canonicalize_campus_load_window(
    readings: Iterable[MeterReading],
    *,
    selected_meter_ids: set[str] | None = None,
    selected_meter_keys: set[tuple[str, str]] | None = None,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    source_timezone: str = "Asia/Shanghai",
    dispatch_resolution_minutes: int = 15,
    max_source_interval_hours: float = 3.0,
    max_meter_power_kw: float | None = None,
    apply_scale: bool = True,
) -> CanonicalLoadResult:
    """Convert selected boundary-meter registers into a canonical campus load.

    Energy between two cumulative readings is distributed uniformly over the
    elapsed interval. This explicit assumption is necessary when source data is
    hourly but the dispatch contract requires 15-minute power.
    """
    selected_meter_ids = selected_meter_ids or set()
    selected_meter_keys = selected_meter_keys or set()
    if not selected_meter_ids and not selected_meter_keys:
        raise ValueError(
            "selected_meter_ids or selected_meter_keys must contain explicit campus boundary meters"
        )
    if 60 % dispatch_resolution_minutes != 0:
        raise ValueError("dispatch_resolution_minutes must divide one hour")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(source_timezone)
    else:
        start_ts = start_ts.tz_convert(source_timezone)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize(source_timezone)
    else:
        end_ts = end_ts.tz_convert(source_timezone)
    if end_ts <= start_ts:
        raise ValueError("end must be later than start")

    deduplicated, duplicate_count = _deduplicate_readings(
        reading
        for reading in readings
        if (
            (reading.ssuid, reading.channel) in selected_meter_keys
            if selected_meter_keys
            else reading.ssuid in selected_meter_ids
        )
    )
    by_meter: dict[tuple[str, str], list[MeterReading]] = defaultdict(list)
    for reading in deduplicated:
        by_meter[(reading.ssuid, reading.channel)].append(reading)

    interval_rows: list[dict[str, object]] = []
    rejected = Counter()
    for (ssuid, channel), meter_readings in by_meter.items():
        meter_readings.sort(key=lambda item: item.timestamp)
        for previous, current in zip(meter_readings, meter_readings[1:]):
            previous_ts = pd.Timestamp(previous.timestamp, tz=source_timezone)
            current_ts = pd.Timestamp(current.timestamp, tz=source_timezone)
            hours = (current_ts - previous_ts).total_seconds() / 3600
            if hours <= 0:
                rejected["non_positive_interval"] += 1
                continue
            if hours > max_source_interval_hours:
                rejected["source_gap"] += 1
                continue
            if (
                previous.communication_state != NORMAL_COMMUNICATION_STATE
                or current.communication_state != NORMAL_COMMUNICATION_STATE
            ):
                rejected["communication_state"] += 1
                continue
            if previous.value < 0 or current.value < 0:
                rejected["negative_register"] += 1
                continue
            if previous.scale != current.scale:
                rejected["scale_change"] += 1
                continue
            delta = current.value - previous.value
            if delta < 0:
                rejected["register_reset"] += 1
                continue
            energy_kwh = delta * (current.scale if apply_scale else 1.0)
            average_power_kw = energy_kwh / hours
            if max_meter_power_kw is not None and average_power_kw > max_meter_power_kw:
                rejected["implausible_meter_power"] += 1
                continue
            interval_rows.append(
                {
                    "ssuid": ssuid,
                    "chno": channel,
                    "interval_start": previous_ts,
                    "interval_end": current_ts,
                    "source_interval_hours": hours,
                    "energy_kwh": energy_kwh,
                    "average_power_kw": average_power_kw,
                }
            )

    grid = pd.date_range(
        start_ts,
        end_ts,
        inclusive="left",
        freq=f"{dispatch_resolution_minutes}min",
    )
    energy_by_bin: dict[pd.Timestamp, float] = defaultdict(float)
    meters_by_bin: dict[pd.Timestamp, set[tuple[str, str]]] = defaultdict(set)
    step = pd.to_timedelta(dispatch_resolution_minutes, unit="min")

    for row in interval_rows:
        interval_start = max(row["interval_start"], start_ts)
        interval_end = min(row["interval_end"], end_ts)
        if interval_end <= interval_start:
            continue
        cursor = interval_start.floor(f"{dispatch_resolution_minutes}min")
        while cursor < interval_end:
            bin_end = cursor + step
            overlap_start = max(cursor, interval_start)
            overlap_end = min(bin_end, interval_end)
            overlap_hours = max(0.0, (overlap_end - overlap_start).total_seconds() / 3600)
            if overlap_hours and cursor in grid:
                energy_by_bin[cursor] += row["average_power_kw"] * overlap_hours
                meters_by_bin[cursor].add((row["ssuid"], row["chno"]))
            cursor = bin_end

    expected_keys = (
        selected_meter_keys
        if selected_meter_keys
        else {(meter_id, channel) for meter_id, channel in by_meter}
    )
    selected_count = len(expected_keys) if selected_meter_keys else len(selected_meter_ids)
    campus_rows: list[dict[str, object]] = []
    incomplete_bins = 0
    for timestamp in grid:
        active_count = (
            len(meters_by_bin[timestamp])
            if selected_meter_keys
            else len({key[0] for key in meters_by_bin[timestamp]})
        )
        complete = active_count == selected_count
        if not complete:
            incomplete_bins += 1
        energy_kwh = energy_by_bin.get(timestamp, 0.0)
        campus_rows.append(
            {
                "timestamp": timestamp,
                "load_power_kw": (
                    energy_kwh / (dispatch_resolution_minutes / 60) if complete else float("nan")
                ),
                "interval_energy_kwh": energy_kwh,
                "active_meter_count": active_count,
                "selected_meter_count": selected_count,
                "coverage_ratio": active_count / selected_count,
            }
        )

    campus = pd.DataFrame(campus_rows)
    meter_intervals = pd.DataFrame(interval_rows)
    if selected_meter_keys:
        missing_selected = [
            f"{ssuid}:{channel}" for ssuid, channel in sorted(selected_meter_keys - set(by_meter))
        ]
    else:
        missing_selected = sorted(selected_meter_ids - {key[0] for key in by_meter})
    quality = {
        "status": "pass" if not incomplete_bins and not missing_selected else "fail",
        "selected_meter_count": selected_count,
        "observed_meter_count": len(by_meter),
        "missing_selected_meter_ids": missing_selected,
        "duplicate_reading_count": duplicate_count,
        "accepted_interval_count": len(interval_rows),
        "rejected_interval_counts": dict(sorted(rejected.items())),
        "incomplete_dispatch_bin_count": incomplete_bins,
        "dispatch_bin_count": len(campus),
        "apply_scale": apply_scale,
        "max_meter_power_kw": max_meter_power_kw,
        "source_to_dispatch_assumption": (
            "相邻累计读数的电量差在对应时间区间内均匀分配到 "
            f"{dispatch_resolution_minutes} 分钟时间片。"
        ),
    }
    return CanonicalLoadResult(campus=campus, meter_intervals=meter_intervals, quality=quality)
