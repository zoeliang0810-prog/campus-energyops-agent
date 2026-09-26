"""Versioned synthetic forecast and carbon-factor fixtures for stage-one tests."""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

from .contracts import (
    CarbonFactorDataset,
    CarbonFactorPoint,
    DispatchInputSeries,
    DispatchSeriesPoint,
    ForecastDataset,
    ForecastSeriesPoint,
    SyntheticDatasetMetadata,
)


CARBON_PATTERNS: dict[str, tuple[float, ...]] = {
    "flat": tuple([0.55] * 24),
    "day_low_night_high": tuple(
        0.68 if hour < 7 else 0.28 if 10 <= hour < 16 else 0.76 if 18 <= hour < 22 else 0.50
        for hour in range(24)
    ),
    "volatile": (
        0.72, 0.69, 0.61, 0.58, 0.66, 0.74, 0.63, 0.49,
        0.57, 0.31, 0.22, 0.44, 0.26, 0.19, 0.38, 0.47,
        0.71, 0.83, 0.77, 0.62, 0.86, 0.73, 0.59, 0.68,
    ),
}


def _metadata(
    *,
    dataset_id: str,
    version: str,
    generated_at,
    unit: str,
    assumptions: list[str],
) -> SyntheticDatasetMetadata:
    return SyntheticDatasetMetadata(
        dataset_id=dataset_id,
        version=version,
        generated_at=generated_at,
        unit=unit,
        assumptions=assumptions,
    )


def build_forecast_scenarios(
    base_inputs: DispatchInputSeries,
) -> dict[str, ForecastDataset]:
    """Create deterministic baseline, low-PV, and high-evening-load forecasts."""
    if len(base_inputs.points) != 24:
        raise ValueError("synthetic forecast scenarios require 24 hourly points")
    generated_at = base_inputs.points[0].timestamp - timedelta(hours=6)
    scenarios: dict[str, ForecastDataset] = {}
    for name in ("baseline", "low_pv", "high_evening_load"):
        points: list[ForecastSeriesPoint] = []
        for point in base_inputs.points:
            pv = point.pv_power_kw * (0.8 if name == "low_pv" else 1.0)
            evening_multiplier = (
                1.15
                if name == "high_evening_load" and 18 <= point.timestamp.hour < 22
                else 1.0
            )
            points.append(
                ForecastSeriesPoint(
                    timestamp=point.timestamp,
                    load_kw=point.load_power_kw * evening_multiplier,
                    pv_kw=pv,
                    load_confidence=0.90,
                    pv_confidence=0.85,
                )
            )
        scenarios[name] = ForecastDataset(
            metadata=_metadata(
                dataset_id=f"hkust_gz_synthetic_forecast_{name}",
                version=f"synthetic-forecast-{name}-v1",
                generated_at=generated_at,
                unit="kW",
                assumptions=[
                    "用于接口、优化和Agent业务流程验证",
                    "不代表校园真实预测",
                    "不用于真实设备控制",
                ],
            ),
            model_version="synthetic-perfect-forecast-v1",
            points=points,
        )
    return scenarios


def build_carbon_scenarios(
    base_inputs: DispatchInputSeries,
) -> dict[str, CarbonFactorDataset]:
    """Create three deterministic hourly carbon-factor patterns."""
    if len(base_inputs.points) != 24:
        raise ValueError("synthetic carbon scenarios require 24 hourly points")
    generated_at = base_inputs.points[0].timestamp - timedelta(hours=6)
    return {
        name: CarbonFactorDataset(
            metadata=_metadata(
                dataset_id=f"hkust_gz_synthetic_carbon_{name}",
                version=f"synthetic-carbon-{name}-v1",
                generated_at=generated_at,
                unit="kgCO2e/kWh",
                assumptions=[
                    "逐时碳因子为确定性模拟曲线",
                    "不代表电网真实或预测碳因子",
                    "只用于低碳调度响应测试",
                ],
            ),
            points=[
                CarbonFactorPoint(
                    timestamp=source.timestamp,
                    carbon_factor_kgco2e_per_kwh=value,
                )
                for source, value in zip(base_inputs.points, values, strict=True)
            ],
        )
        for name, values in CARBON_PATTERNS.items()
    }


def _validate_hourly_timestamps(timestamps, *, label: str) -> None:
    if len(timestamps) != 24:
        raise ValueError(f"{label} must contain exactly 24 hourly points")
    if len(set(timestamps)) != len(timestamps):
        raise ValueError(f"{label} contains duplicate timestamps")
    for previous, current in zip(timestamps, timestamps[1:]):
        if current - previous != timedelta(hours=1):
            raise ValueError(f"{label} timestamps are not hourly and contiguous")


def validate_forecast_dataset(
    dataset: ForecastDataset,
    *,
    horizon_start,
    max_age_hours: float = 24,
) -> None:
    if dataset.metadata.unit != "kW":
        raise ValueError("FORECAST_UNIT_INVALID")
    _validate_hourly_timestamps(
        [point.timestamp for point in dataset.points], label="forecast"
    )
    if dataset.points[0].timestamp != horizon_start:
        raise ValueError("FORECAST_HORIZON_MISMATCH")
    age_hours = (horizon_start - dataset.metadata.generated_at).total_seconds() / 3600
    if age_hours < 0 or age_hours > max_age_hours:
        raise ValueError("FORECAST_STALE")


def validate_carbon_dataset(
    dataset: CarbonFactorDataset,
    *,
    horizon_start,
    max_age_hours: float = 24,
) -> None:
    if dataset.metadata.unit != "kgCO2e/kWh":
        raise ValueError("CARBON_FACTOR_UNIT_INVALID")
    _validate_hourly_timestamps(
        [point.timestamp for point in dataset.points], label="carbon factor"
    )
    if dataset.points[0].timestamp != horizon_start:
        raise ValueError("CARBON_FACTOR_HORIZON_MISMATCH")
    age_hours = (horizon_start - dataset.metadata.generated_at).total_seconds() / 3600
    if age_hours < 0 or age_hours > max_age_hours:
        raise ValueError("CARBON_FACTOR_STALE")


def combine_decision_inputs(
    forecast: ForecastDataset,
    carbon: CarbonFactorDataset,
    *,
    buy_prices: list[float],
    sell_price: float = 0.0,
) -> DispatchInputSeries:
    """Validate and combine external-module contracts into optimizer inputs."""
    horizon_start = forecast.points[0].timestamp
    validate_forecast_dataset(forecast, horizon_start=horizon_start)
    validate_carbon_dataset(carbon, horizon_start=horizon_start)
    if len(buy_prices) != 24 or any(price < 0 for price in buy_prices):
        raise ValueError("TARIFF_PROFILE_INVALID")
    carbon_by_time = {point.timestamp: point for point in carbon.points}
    points = []
    for index, point in enumerate(forecast.points):
        factor = carbon_by_time.get(point.timestamp)
        if factor is None:
            raise ValueError("CARBON_FACTOR_MISSING")
        points.append(
            DispatchSeriesPoint(
                timestamp=point.timestamp,
                load_power_kw=point.load_kw,
                pv_power_kw=point.pv_kw,
                buy_price_cny_per_kwh=buy_prices[index],
                sell_price_cny_per_kwh=sell_price,
                carbon_factor_kgco2e_per_kwh=(
                    factor.carbon_factor_kgco2e_per_kwh
                ),
            )
        )
    return DispatchInputSeries(
        dataset_version=(
            f"forecast={forecast.metadata.version};carbon={carbon.metadata.version}"
        ),
        points=points,
        source="synthetic_fixture",
        generated_at=min(
            forecast.metadata.generated_at, carbon.metadata.generated_at
        ),
        timezone=forecast.metadata.timezone,
        power_unit=forecast.metadata.unit,
        carbon_factor_unit=carbon.metadata.unit,
        forecast_model_version=forecast.model_version,
        forecast_confidence=min(
            min(point.load_confidence for point in forecast.points),
            min(point.pv_confidence for point in forecast.points),
        ),
        carbon_factor_version=carbon.metadata.version,
        verified=False,
        simulation_only=True,
        assumptions=[*forecast.metadata.assumptions, *carbon.metadata.assumptions],
    )


def write_synthetic_fixture_set(
    output_dir: str | Path,
    base_inputs: DispatchInputSeries,
) -> tuple[dict[str, ForecastDataset], dict[str, CarbonFactorDataset]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    forecasts = build_forecast_scenarios(base_inputs)
    carbon = build_carbon_scenarios(base_inputs)
    for name, dataset in {**forecasts, **carbon}.items():
        prefix = "forecast" if name in forecasts else "carbon"
        path = output / f"{prefix}_{name}.json"
        path.write_text(
            json.dumps(dataset.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return forecasts, carbon
