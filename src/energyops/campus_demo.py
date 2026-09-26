"""Deterministic temporary inputs for the campus PV-storage demo."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from .contracts import (
    BatteryConfig,
    DispatchInputSeries,
    DispatchRequest,
    DispatchSeriesPoint,
    OperatingMode,
)


def synthetic_campus_load_kw(timestamp: pd.Timestamp) -> float:
    """Return a transparent placeholder load shape, never a measured load."""
    hour = timestamp.hour + timestamp.minute / 60
    if hour < 6:
        return 35.0
    if hour < 8:
        return 50.0
    if hour < 17:
        return 65.0 + 8.0 * math.sin((hour - 8) / 9 * math.pi)
    if hour < 18:
        return 80.0
    if hour < 22:
        return 95.0
    return 55.0


def demo_buy_price(timestamp: pd.Timestamp) -> float:
    hour = timestamp.hour
    if hour < 6:
        return 0.35
    if 10 <= hour < 15 or 18 <= hour < 22:
        return 1.00
    return 0.65


def build_demo_inputs(
    pv_csv: str | Path,
    *,
    dataset_version: str,
    load_csv: str | Path | None = None,
) -> DispatchInputSeries:
    frame = pd.read_csv(pv_csv)
    required = {"timestamp", "campus_pv_power_kw"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required campus columns: {sorted(missing)}")
    timestamps = pd.to_datetime(frame["timestamp"], errors="raise")
    measured_load: pd.Series | None = None
    if load_csv is not None:
        load_frame = pd.read_csv(load_csv)
        load_required = {"timestamp", "load_power_kw"}
        load_missing = load_required - set(load_frame.columns)
        if load_missing:
            raise ValueError(f"missing required load columns: {sorted(load_missing)}")
        load_timestamps = pd.to_datetime(load_frame["timestamp"], errors="raise")
        load_power = pd.to_numeric(load_frame["load_power_kw"], errors="raise")
        if load_timestamps.duplicated().any():
            raise ValueError("load data contains duplicate timestamps")
        if load_power.isna().any() or (load_power < 0).any():
            raise ValueError("load power must be complete and non-negative")
        measured_load = pd.Series(load_power.to_numpy(), index=load_timestamps)
        missing_load = pd.DatetimeIndex(timestamps).difference(measured_load.index)
        extra_load = measured_load.index.difference(pd.DatetimeIndex(timestamps))
        if len(missing_load) or len(extra_load):
            raise ValueError(
                "load timestamps must match the campus PV timestamps exactly"
            )
    points = [
        DispatchSeriesPoint(
            timestamp=timestamp.to_pydatetime(),
            pv_power_kw=float(row.campus_pv_power_kw),
            load_power_kw=(
                float(measured_load.loc[timestamp])
                if measured_load is not None
                else synthetic_campus_load_kw(timestamp)
            ),
            buy_price_cny_per_kwh=demo_buy_price(timestamp),
            sell_price_cny_per_kwh=0.30,
        )
        for timestamp, row in zip(timestamps, frame.itertuples(index=False), strict=True)
    ]
    return DispatchInputSeries(dataset_version=dataset_version, points=points)


def build_demo_request(
    inputs: DispatchInputSeries,
    *,
    measured_load: bool = False,
) -> DispatchRequest:
    if not inputs.points:
        raise ValueError("campus demo requires at least one input point")
    return DispatchRequest(
        request_id="jiujiang_campus_demo_2026-02-19",
        site_id="jiujiang_campus",
        dataset_version=inputs.dataset_version,
        operating_mode=OperatingMode.SIMULATION_ONLY,
        horizon_start=inputs.points[0].timestamp,
        horizon_steps=len(inputs.points),
        step_minutes=15,
        grid_import_limit_kw=200.0,
        grid_export_limit_kw=0.0,
        allow_grid_charging=False,
        allow_grid_export=False,
        tariff_region="unverified_demo",
        tariff_version="synthetic_tou_v1",
        battery=BatteryConfig(
            asset_id="campus_battery_scenario_50kw_100kwh",
            capacity_kwh=100.0,
            initial_energy_kwh=50.0,
            min_energy_kwh=10.0,
            max_energy_kwh=90.0,
            max_charge_kw=50.0,
            max_discharge_kw=50.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            degradation_cost_cny_per_kwh_throughput=0.03,
        ),
        required_terminal_energy_kwh=50.0,
        assumptions=([
            "校园负荷为确定性 synthetic 占位曲线，不是实测负荷",
        ] if not measured_load else []) + [
            "分时电价为未核验演示价格，不代表江西实际结算",
            "50 kW / 100 kWh 储能为敏感性分析场景，不是设备选型",
            "光伏使用历史实测回放，等价于完美预测上界",
        ],
    )
