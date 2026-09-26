"""Validated, file-backed EnergyOps scenario configuration."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from pydantic import Field, model_validator

from .contracts import BatteryConfig, ObjectiveMode, StrictModel


class LocationConfig(StrictModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    address: str
    coordinate_basis: str


class DatasetPathConfig(StrictModel):
    dataset_id: str
    csv: str
    manifest: str
    quality: str


class PVScenarioConfig(DatasetPathConfig):
    capacity_kwp: float = Field(gt=0)
    capacity_status: str = "scenario_assumption"
    profile_basis: str


class TariffConfig(StrictModel):
    region: str
    version: str
    verified: bool = False
    buy_price_cny_per_kwh_by_hour: list[float]
    sell_price_cny_per_kwh: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_hourly_prices(self) -> "TariffConfig":
        if len(self.buy_price_cny_per_kwh_by_hour) != 24:
            raise ValueError("tariff must contain exactly 24 hourly buy prices")
        if any(price < 0 for price in self.buy_price_cny_per_kwh_by_hour):
            raise ValueError("tariff prices must be non-negative")
        return self


class GridConfig(StrictModel):
    import_headroom_kw: float = Field(gt=0)
    export_limit_kw: float = Field(default=0, ge=0)
    allow_grid_charging: bool = True
    allow_grid_export: bool = False


class CampusScenarioConfig(StrictModel):
    scenario_id: str
    scenario_name: str
    site_id: str
    study_day: date
    step_minutes: int = 60
    horizon_steps: int = 24
    objective_mode: ObjectiveMode = ObjectiveMode.COST
    location: LocationConfig
    load: DatasetPathConfig
    pv: PVScenarioConfig
    battery: BatteryConfig
    required_terminal_energy_kwh: float = Field(ge=0)
    grid: GridConfig
    tariff: TariffConfig
    assumptions: list[str]
    simulation_only: bool = True
    executable: bool = False

    @model_validator(mode="after")
    def validate_policy(self) -> "CampusScenarioConfig":
        if self.step_minutes != 60 or self.horizon_steps != 24:
            raise ValueError("the baseline campus scenario requires 24 hourly points")
        if self.objective_mode != ObjectiveMode.COST:
            raise ValueError("the baseline campus scenario uses the cost objective")
        if not self.grid.allow_grid_charging or self.grid.allow_grid_export:
            raise ValueError("baseline policy requires grid charging and forbids export")
        if self.tariff.verified:
            raise ValueError("the current reference tariff must remain unverified")
        if self.required_terminal_energy_kwh > self.battery.max_energy_kwh:
            raise ValueError("terminal energy cannot exceed battery maximum")
        if not self.simulation_only or self.executable:
            raise ValueError("unverified equipment scenarios must be simulation-only")
        return self


def load_scenario_config(path: str | Path) -> CampusScenarioConfig:
    config_path = Path(path)
    return CampusScenarioConfig.model_validate_json(
        config_path.read_text(encoding="utf-8")
    )


def write_scenario_snapshot(path: str | Path, config: CampusScenarioConfig) -> None:
    Path(path).write_text(
        json.dumps(config.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
