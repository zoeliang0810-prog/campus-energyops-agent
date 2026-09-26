"""Data-quality and canonicalization tools for PV time series."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import pandas as pd

from .contracts import DataQualityReport, DatasetManifest, QualityStatus


DEFAULT_CAMPUS_PV_COLUMNS = tuple(f"pv_device_{index}" for index in range(1, 6))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pv_csv(
    path: str | Path,
    *,
    timestamp_column: str = "time",
    power_column: str = "实时发电功率",
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing_columns = {timestamp_column, power_column} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"missing required columns: {sorted(missing_columns)}")
    output = frame[[timestamp_column, power_column]].rename(
        columns={timestamp_column: "timestamp", power_column: "pv_power_kw"}
    )
    output["timestamp"] = pd.to_datetime(output["timestamp"], errors="coerce")
    output["pv_power_kw"] = pd.to_numeric(output["pv_power_kw"], errors="coerce")
    return output


def load_campus_pv_csv(
    path: str | Path,
    *,
    timestamp_column: str = "time",
    device_columns: Sequence[str] = DEFAULT_CAMPUS_PV_COLUMNS,
) -> pd.DataFrame:
    """Load device-level PV readings and derive a campus total.

    Device columns remain in the canonical frame so the aggregate used by the
    optimizer can always be traced back to an individual inverter.
    """
    frame = pd.read_csv(path)
    required = {timestamp_column, *device_columns}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"missing required columns: {sorted(missing_columns)}")

    output = frame[[timestamp_column, *device_columns]].rename(
        columns={timestamp_column: "timestamp"}
    )
    output["timestamp"] = pd.to_datetime(output["timestamp"], errors="coerce")
    for column in device_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["pv_power_kw"] = output[list(device_columns)].sum(axis=1, min_count=len(device_columns))
    return output


def profile_pv_data(
    frame: pd.DataFrame,
    *,
    dataset_version: str,
    expected_resolution_minutes: int = 5,
) -> DataQualityReport:
    timestamp = frame["timestamp"]
    power = frame["pv_power_kw"]
    parsed = timestamp.dropna().sort_values()
    duplicate_count = int(parsed.duplicated().sum())

    deltas = parsed.drop_duplicates().diff().dropna()
    delta_minutes = deltas.dt.total_seconds().div(60)
    positive_delta_minutes = delta_minutes[delta_minutes > 0]
    inferred_resolution = None
    if not positive_delta_minutes.empty:
        inferred_resolution = int(positive_delta_minutes.mode().iloc[0])

    missing_timestamp_count = 0
    gap_count = 0
    maximum_gap_minutes = 0.0
    if not parsed.empty and inferred_resolution:
        expected = pd.date_range(
            parsed.min(), parsed.max(), freq=f"{expected_resolution_minutes}min"
        )
        missing_timestamp_count = len(expected.difference(pd.DatetimeIndex(parsed)))
        gap_count = int((delta_minutes > expected_resolution_minutes).sum())
        maximum_gap_minutes = float(
            positive_delta_minutes.max() if not positive_delta_minutes.empty else 0
        )

    null_power_count = int(power.isna().sum())
    negative_power_count = int((power < 0).sum())
    findings: list[str] = []
    status = QualityStatus.PASS

    if int(timestamp.isna().sum()):
        findings.append("存在无法解析的时间戳。")
        status = QualityStatus.FAIL
    if duplicate_count:
        findings.append("时间戳在目标时间粒度上不唯一。")
        status = QualityStatus.FAIL
    if null_power_count:
        findings.append("光伏功率存在空值。")
        status = QualityStatus.FAIL
    if negative_power_count:
        findings.append("光伏功率存在负值。")
        status = QualityStatus.FAIL
    if inferred_resolution and inferred_resolution != expected_resolution_minutes:
        findings.append(
            f"主要时间分辨率为 {inferred_resolution} 分钟，预期为 {expected_resolution_minutes} 分钟。"
        )
        status = QualityStatus.FAIL
    if missing_timestamp_count and status != QualityStatus.FAIL:
        findings.append(
            f"完整时间范围内缺少 {missing_timestamp_count} 个时间片；调度前必须选择连续窗口或显式修复。"
        )
        status = QualityStatus.WARNING
    if not findings:
        findings.append("时间戳、功率范围和连续性检查均通过。")

    evidence_suffix = hashlib.sha256(
        f"{dataset_version}:{len(frame)}:{missing_timestamp_count}".encode()
    ).hexdigest()[:12]
    return DataQualityReport(
        dataset_version=dataset_version,
        status=status,
        row_count=len(frame),
        parsed_timestamp_count=len(parsed),
        duplicate_timestamp_count=duplicate_count,
        missing_timestamp_count=missing_timestamp_count,
        null_power_count=null_power_count,
        negative_power_count=negative_power_count,
        gap_count=gap_count,
        maximum_gap_minutes=maximum_gap_minutes,
        inferred_resolution_minutes=inferred_resolution,
        findings=findings,
        evidence_id=f"dq_{evidence_suffix}",
    )


def canonicalize_pv_window(
    frame: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    source_timezone: str = "Asia/Shanghai",
    dispatch_resolution_minutes: int = 15,
) -> pd.DataFrame:
    data = frame.copy()
    if data["timestamp"].dt.tz is None:
        data["timestamp"] = data["timestamp"].dt.tz_localize(source_timezone)
    else:
        data["timestamp"] = data["timestamp"].dt.tz_convert(source_timezone)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(source_timezone)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize(source_timezone)

    window = data[(data["timestamp"] >= start_ts) & (data["timestamp"] < end_ts)]
    if window.empty:
        raise ValueError("requested PV window contains no rows")
    if window["timestamp"].duplicated().any():
        raise ValueError("requested PV window contains duplicate timestamps")

    indexed = window.set_index("timestamp").sort_index()
    expected_source = pd.date_range(
        start_ts, end_ts, inclusive="left", freq="5min"
    )
    missing_source = expected_source.difference(indexed.index)
    if len(missing_source):
        raise ValueError(
            f"requested PV window is not continuous: {len(missing_source)} source points missing"
        )

    canonical = indexed[["pv_power_kw"]].resample(
        f"{dispatch_resolution_minutes}min", label="left", closed="left"
    ).mean()
    expected_dispatch = pd.date_range(
        start_ts, end_ts, inclusive="left", freq=f"{dispatch_resolution_minutes}min"
    )
    canonical = canonical.reindex(expected_dispatch)
    if canonical["pv_power_kw"].isna().any():
        raise ValueError("resampling produced missing dispatch points")
    canonical.index.name = "timestamp"
    return canonical.reset_index()


def canonicalize_campus_pv_window(
    frame: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    device_columns: Sequence[str] = DEFAULT_CAMPUS_PV_COLUMNS,
    source_timezone: str = "Asia/Shanghai",
    source_resolution_minutes: int = 5,
    dispatch_resolution_minutes: int = 15,
) -> pd.DataFrame:
    """Build a traceable campus series while retaining each device contribution."""
    required = {"timestamp", "pv_power_kw", *device_columns}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"missing required columns: {sorted(missing_columns)}")

    data = frame.copy()
    if data["timestamp"].dt.tz is None:
        data["timestamp"] = data["timestamp"].dt.tz_localize(source_timezone)
    else:
        data["timestamp"] = data["timestamp"].dt.tz_convert(source_timezone)

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize(source_timezone)
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize(source_timezone)

    window = data[(data["timestamp"] >= start_ts) & (data["timestamp"] < end_ts)]
    if window.empty:
        raise ValueError("requested campus PV window contains no rows")
    if window["timestamp"].duplicated().any():
        raise ValueError("requested campus PV window contains duplicate timestamps")

    indexed = window.set_index("timestamp").sort_index()
    expected_source = pd.date_range(
        start_ts,
        end_ts,
        inclusive="left",
        freq=f"{source_resolution_minutes}min",
    )
    missing_source = expected_source.difference(indexed.index)
    if len(missing_source):
        raise ValueError(
            f"requested campus PV window is not continuous: {len(missing_source)} source points missing"
        )
    if indexed[[*device_columns, "pv_power_kw"]].isna().any().any():
        raise ValueError("requested campus PV window contains missing device power")

    canonical = indexed[[*device_columns, "pv_power_kw"]].resample(
        f"{dispatch_resolution_minutes}min", label="left", closed="left"
    ).mean()
    expected_dispatch = pd.date_range(
        start_ts, end_ts, inclusive="left", freq=f"{dispatch_resolution_minutes}min"
    )
    canonical = canonical.reindex(expected_dispatch)
    if canonical.isna().any().any():
        raise ValueError("resampling produced missing dispatch points")

    recomputed_total = canonical[list(device_columns)].sum(axis=1)
    if not (recomputed_total - canonical["pv_power_kw"]).abs().le(1e-9).all():
        raise ValueError("campus PV total does not equal the sum of the five devices")
    canonical = canonical.rename(columns={"pv_power_kw": "campus_pv_power_kw"})
    canonical.index.name = "timestamp"
    return canonical.reset_index()


def build_manifest(
    *,
    source_path: str | Path,
    canonical: pd.DataFrame,
    site_id: str,
    source_timezone: str,
    source_resolution_minutes: int,
    dispatch_resolution_minutes: int,
    dataset_id: str | None = None,
    variable: str = "pv_power_kw",
    transformations: list[str] | None = None,
    assumptions: list[str] | None = None,
) -> DatasetManifest:
    source_hash = sha256_file(source_path)
    version = f"sha256:{source_hash[:16]}"
    return DatasetManifest(
        dataset_id=dataset_id or f"{site_id}_pv",
        dataset_version=version,
        site_id=site_id,
        source_sha256=source_hash,
        source_name=Path(source_path).name,
        source_timezone=source_timezone,
        variable=variable,
        source_resolution_minutes=source_resolution_minutes,
        dispatch_resolution_minutes=dispatch_resolution_minutes,
        rows=len(canonical),
        start_time=canonical["timestamp"].min().to_pydatetime(),
        end_time=canonical["timestamp"].max().to_pydatetime(),
        transformations=transformations
        or [
            "保留原始 5 分钟功率数据，不进行静默删行或插值",
            "按 15 分钟窗口计算平均功率，能量由平均功率乘以 0.25 小时得到",
        ],
        assumptions=assumptions or [],
    )
