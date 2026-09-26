"""Build the 24-hour HKUST(GZ) load + Jiujiang PV simulation scenario."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from .contracts import (
    BatteryConfig,
    DispatchInputSeries,
    DispatchRequest,
    DispatchSeriesPoint,
    ParsedCampusRequest,
)

if TYPE_CHECKING:
    from .scenario_config import CampusScenarioConfig


def jiujiang_reference_buy_price_cny_per_kwh(hour: int) -> float:
    """Legacy Jiujiang notebook tariff used only as a simulation parameter.

    This is not a verified HKUST(GZ) settlement tariff and must never be used to
    claim actual savings.
    """
    if 0 <= hour < 8:
        return 0.1721
    if hour in {10, 14, 16, 17, 18}:
        return 0.7701
    if hour in {11, 15}:
        return 0.9626
    return 0.4530


def _hourly_profile(
    path: str | Path,
    *,
    value_column: str,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = {timestamp_column, value_column} - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    timestamps = pd.to_datetime(frame[timestamp_column], errors="raise")
    values = pd.to_numeric(frame[value_column], errors="coerce")
    if timestamps.duplicated().any():
        raise ValueError(f"{path} contains duplicate timestamps")
    if values.isna().any() or (values < 0).any():
        raise ValueError(f"{value_column} must be complete and non-negative")
    normalized = pd.DataFrame({value_column: values.to_numpy()}, index=timestamps)
    hourly = normalized.resample("1h").mean()
    if len(hourly) != 24 or hourly[value_column].isna().any():
        raise ValueError(f"{path} must resolve to one complete 24-hour profile")
    return hourly


def synthetic_hkust_hourly_load_kw(hour: int) -> float:
    """Transparent placeholder used only before boundary-meter approval."""
    if hour < 6:
        return 450.0
    if hour < 8:
        return 650.0
    if hour < 17:
        return 900.0 + 100.0 * math.sin((hour - 8) / 9 * math.pi)
    if hour < 22:
        return 820.0
    return 560.0


def build_hourly_inputs(
    pv_csv: str | Path,
    *,
    dataset_version: str,
    load_csv: str | Path | None = None,
    buy_price_cny_per_kwh_by_hour: list[float] | None = None,
    sell_price_cny_per_kwh: float = 0.0,
) -> DispatchInputSeries:
    """Align one complete campus load and PV day by hour-of-day.

    A missing load file intentionally produces an explicitly synthetic profile;
    callers must keep the resulting workflow in simulation-only mode.
    """
    pv = _hourly_profile(pv_csv, value_column="campus_pv_power_kw")
    prices = buy_price_cny_per_kwh_by_hour or [
        jiujiang_reference_buy_price_cny_per_kwh(hour) for hour in range(24)
    ]
    if len(prices) != 24 or any(price < 0 for price in prices):
        raise ValueError("buy price profile must contain 24 non-negative values")
    if sell_price_cny_per_kwh < 0:
        raise ValueError("sell price must be non-negative")
    load_values: list[float]
    if load_csv is None:
        load_values = [synthetic_hkust_hourly_load_kw(index) for index in range(24)]
    else:
        load = _hourly_profile(load_csv, value_column="load_power_kw")
        load_values = [float(value) for value in load["load_power_kw"]]

    points = [
        DispatchSeriesPoint(
            timestamp=timestamp.to_pydatetime(),
            pv_power_kw=float(pv_power),
            load_power_kw=load_values[index],
            buy_price_cny_per_kwh=prices[timestamp.hour],
            sell_price_cny_per_kwh=sell_price_cny_per_kwh,
        )
        for index, (timestamp, pv_power) in enumerate(
            pv["campus_pv_power_kw"].items()
        )
    ]
    return DispatchInputSeries(dataset_version=dataset_version, points=points)


def build_dispatch_request(
    parsed: ParsedCampusRequest,
    inputs: DispatchInputSeries,
    *,
    measured_load: bool,
    scenario_config: "CampusScenarioConfig | None" = None,
) -> DispatchRequest:
    if len(inputs.points) != parsed.horizon_steps:
        raise ValueError("input horizon does not match parsed request")
    if scenario_config is None:
        assumptions = [
            "九江历史光伏按小时序与港科广负荷组合，不代表广州同期实测光伏",
            "50 kW / 100 kWh 储能是场景假设，不是已核验铭牌",
            "购电价格沿用九江旧方案中的广东2025参考分时价格，仅用于复现调度动作，不代表广州实际结算",
            "校园分表聚合负荷是固定、不可转移的功率平衡输入，不是优化决策变量",
            "允许电网在参考低价时段给储能充电，但禁止向电网上网",
        ]
        request_id = "hkust_gz_jiujiang_pv_storage_v1"
        scenario_name = parsed.scenario_name
        grid_import_headroom_kw = 100.0
        grid_export_limit_kw = 0.0
        tariff_region = "jiujiang_notebook_guangdong_reference_unverified"
        tariff_version = "jiujiang_notebook_guangdong_2025_reference_v1"
        tariff_verified = False
        battery = BatteryConfig(
            asset_id=parsed.battery_config_id,
            capacity_kwh=100.0,
            initial_energy_kwh=50.0,
            min_energy_kwh=10.0,
            max_energy_kwh=90.0,
            max_charge_kw=50.0,
            max_discharge_kw=50.0,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            degradation_cost_cny_per_kwh_throughput=0.128,
        )
        terminal_energy = 50.0
        allow_grid_charging = parsed.allow_grid_charging
        allow_grid_export = parsed.allow_grid_export
        objective_mode = parsed.objective
    else:
        if parsed.objective != scenario_config.objective_mode:
            raise ValueError("parsed objective does not match scenario configuration")
        if parsed.allow_grid_charging != scenario_config.grid.allow_grid_charging:
            raise ValueError("parsed grid-charging policy does not match configuration")
        if parsed.allow_grid_export != scenario_config.grid.allow_grid_export:
            raise ValueError("parsed grid-export policy does not match configuration")
        assumptions = list(scenario_config.assumptions)
        request_id = scenario_config.scenario_id
        scenario_name = scenario_config.scenario_name
        grid_import_headroom_kw = scenario_config.grid.import_headroom_kw
        grid_export_limit_kw = scenario_config.grid.export_limit_kw
        tariff_region = scenario_config.tariff.region
        tariff_version = scenario_config.tariff.version
        tariff_verified = scenario_config.tariff.verified
        battery = scenario_config.battery
        terminal_energy = scenario_config.required_terminal_energy_kwh
        allow_grid_charging = scenario_config.grid.allow_grid_charging
        allow_grid_export = scenario_config.grid.allow_grid_export
        objective_mode = scenario_config.objective_mode
    if not measured_load:
        assumptions.append("校园负荷是流程联调用 synthetic 曲线，不是港科广实测负荷")
    return DispatchRequest(
        request_id=request_id,
        scenario_name=scenario_name,
        site_id=parsed.site_id,
        dataset_version=inputs.dataset_version,
        operating_mode=parsed.operating_mode,
        horizon_start=inputs.points[0].timestamp,
        horizon_steps=parsed.horizon_steps,
        step_minutes=parsed.step_minutes,
        grid_import_limit_kw=max(point.load_power_kw for point in inputs.points)
        + grid_import_headroom_kw,
        grid_export_limit_kw=grid_export_limit_kw,
        allow_grid_charging=allow_grid_charging,
        allow_grid_export=allow_grid_export,
        tariff_region=tariff_region,
        tariff_version=tariff_version,
        tariff_verified=tariff_verified,
        objective_mode=objective_mode,
        carbon_factor_version=inputs.carbon_factor_version,
        peak_penalty=1.0,
        curtailment_penalty=0.1,
        throughput_penalty=0.001,
        battery=battery,
        required_terminal_energy_kwh=terminal_energy,
        assumptions=assumptions,
        critical_load_locked=parsed.critical_load_locked,
        simulation_only=True,
        executable=False,
    )
