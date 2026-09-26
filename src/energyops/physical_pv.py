"""Auditable physical PV model for the HKUST(GZ) planning scenarios."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from .contracts import StrictModel


MODEL_VERSION = "hkust-gz-physical-pv-v1"
SOLAR_CONSTANT_W_M2 = 1367.0


class PhysicalPVLocation(StrictModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    timezone: str
    elevation_m: float = 0.0
    coordinate_basis: str

    @model_validator(mode="after")
    def validate_timezone(self) -> "PhysicalPVLocation":
        ZoneInfo(self.timezone)
        return self


class PVModuleSpec(StrictModel):
    rated_power_wp: float = Field(gt=0)
    width_mm: float = Field(gt=0)
    height_mm: float = Field(gt=0)
    temperature_coefficient_per_c: float = Field(lt=0)
    noct_c: float = Field(gt=20)
    technology: str
    parameter_status: str

    @property
    def area_m2(self) -> float:
        return self.width_mm / 1000.0 * self.height_mm / 1000.0

    @property
    def nominal_efficiency(self) -> float:
        return self.rated_power_wp / (1000.0 * self.area_m2)


class PVInverterSpec(StrictModel):
    peak_efficiency: float = Field(gt=0, le=1)
    dc_ac_ratio: float = Field(gt=0)
    parameter_status: str


class PVOrientationSpec(StrictModel):
    orientation_id: str
    module_count: int = Field(gt=0)
    tilt_degrees: float = Field(ge=0, le=180)
    azimuth_degrees: float = Field(ge=0, lt=360)


class PVArraySpec(StrictModel):
    array_id: str
    name: str
    module_count: int = Field(gt=0)
    orientations: list[PVOrientationSpec]
    parameter_status: str

    @model_validator(mode="after")
    def validate_orientation_counts(self) -> "PVArraySpec":
        if not self.orientations:
            raise ValueError("PV array requires at least one orientation")
        if sum(item.module_count for item in self.orientations) != self.module_count:
            raise ValueError(
                f"orientation module counts do not match array {self.array_id}"
            )
        identifiers = [item.orientation_id for item in self.orientations]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"duplicate orientation id in array {self.array_id}")
        return self


class PVPlanningScenario(StrictModel):
    scenario_id: str
    name: str
    array_ids: list[str]
    expected_capacity_kwp: float = Field(gt=0)


class FacadeOrientationCase(StrictModel):
    case_id: str
    description: str
    module_counts: dict[str, int]

    @model_validator(mode="after")
    def validate_module_counts(self) -> "FacadeOrientationCase":
        expected = {"east", "south", "west", "north"}
        if set(self.module_counts) != expected:
            raise ValueError(
                f"facade case {self.case_id} must define east/south/west/north"
            )
        if any(value <= 0 for value in self.module_counts.values()):
            raise ValueError("facade sensitivity module counts must be positive")
        return self


class PVSensitivityConfig(StrictModel):
    facade_orientation_cases: list[FacadeOrientationCase]
    system_loss_fractions: list[float]

    @model_validator(mode="after")
    def validate_sensitivity(self) -> "PVSensitivityConfig":
        identifiers = [item.case_id for item in self.facade_orientation_cases]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("facade sensitivity case ids must be unique")
        if not self.system_loss_fractions:
            raise ValueError("at least one system loss sensitivity is required")
        if any(value < 0 or value >= 1 for value in self.system_loss_fractions):
            raise ValueError("system loss sensitivities must be in [0, 1)")
        return self


class PhysicalPVConfig(StrictModel):
    config_id: str
    site_id: str
    study_day: date
    step_minutes: int = 60
    location: PhysicalPVLocation
    module: PVModuleSpec
    inverter: PVInverterSpec
    system_loss_fraction: float = Field(ge=0, lt=1)
    albedo: float = Field(ge=0, le=1)
    bifacial_gain_fraction: float = Field(ge=0)
    weather_csv: str
    output_prefix: str
    arrays: list[PVArraySpec]
    scenarios: list[PVPlanningScenario]
    sensitivity: PVSensitivityConfig
    assumptions: list[str]
    simulation_only: bool = True
    executable: bool = False

    @model_validator(mode="after")
    def validate_configuration(self) -> "PhysicalPVConfig":
        if self.step_minutes != 60:
            raise ValueError("stage A currently requires hourly weather")
        if not self.simulation_only or self.executable:
            raise ValueError("planning PV scenarios must remain simulation-only")
        array_ids = [item.array_id for item in self.arrays]
        if len(array_ids) != len(set(array_ids)):
            raise ValueError("array ids must be unique")
        array_by_id = {item.array_id: item for item in self.arrays}
        scenario_ids = [item.scenario_id for item in self.scenarios]
        if len(scenario_ids) != len(set(scenario_ids)):
            raise ValueError("scenario ids must be unique")
        for scenario in self.scenarios:
            unknown = set(scenario.array_ids) - set(array_by_id)
            if unknown:
                raise ValueError(
                    f"scenario {scenario.scenario_id} references unknown arrays: {unknown}"
                )
            capacity = sum(
                array_by_id[array_id].module_count * self.module.rated_power_wp
                / 1000.0
                for array_id in scenario.array_ids
            )
            if not math.isclose(
                capacity, scenario.expected_capacity_kwp, rel_tol=0, abs_tol=1e-6
            ):
                raise ValueError(
                    f"scenario {scenario.scenario_id} capacity is {capacity}, "
                    f"not {scenario.expected_capacity_kwp}"
                )
        facade = array_by_id.get("facade")
        if facade is None:
            raise ValueError("stage A sensitivity requires the facade array")
        for case in self.sensitivity.facade_orientation_cases:
            if sum(case.module_counts.values()) != facade.module_count:
                raise ValueError(
                    f"facade case {case.case_id} module counts do not total "
                    f"{facade.module_count}"
                )
        return self

    def array_capacity_kwp(self, array_id: str) -> float:
        array = next(item for item in self.arrays if item.array_id == array_id)
        return array.module_count * self.module.rated_power_wp / 1000.0


@dataclass(frozen=True)
class PhysicalPVRun:
    components: pd.DataFrame
    scenarios: dict[str, pd.DataFrame]
    manifests: dict[str, dict[str, Any]]
    quality_reports: dict[str, dict[str, Any]]


def load_physical_pv_config(path: str | Path) -> PhysicalPVConfig:
    return PhysicalPVConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _solar_position(
    timestamp: datetime, *, latitude: float, longitude: float
) -> tuple[float, float, float]:
    """Return solar elevation, azimuth and cosine of zenith using NOAA equations."""
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("solar position timestamps must be timezone-aware")
    day = timestamp.timetuple().tm_yday
    local_hour = (
        timestamp.hour
        + timestamp.minute / 60.0
        + timestamp.second / 3600.0
    )
    fractional_year = 2.0 * math.pi / 365.0 * (
        day - 1 + (local_hour - 12.0) / 24.0
    )
    equation_of_time = 229.18 * (
        0.000075
        + 0.001868 * math.cos(fractional_year)
        - 0.032077 * math.sin(fractional_year)
        - 0.014615 * math.cos(2.0 * fractional_year)
        - 0.040849 * math.sin(2.0 * fractional_year)
    )
    declination = (
        0.006918
        - 0.399912 * math.cos(fractional_year)
        + 0.070257 * math.sin(fractional_year)
        - 0.006758 * math.cos(2.0 * fractional_year)
        + 0.000907 * math.sin(2.0 * fractional_year)
        - 0.002697 * math.cos(3.0 * fractional_year)
        + 0.00148 * math.sin(3.0 * fractional_year)
    )
    offset_hours = timestamp.utcoffset().total_seconds() / 3600.0
    time_offset_minutes = equation_of_time + 4.0 * longitude - 60.0 * offset_hours
    true_solar_minutes = (
        timestamp.hour * 60.0
        + timestamp.minute
        + timestamp.second / 60.0
        + time_offset_minutes
    ) % 1440.0
    hour_angle_degrees = true_solar_minutes / 4.0 - 180.0
    hour_angle = math.radians(hour_angle_degrees)
    latitude_rad = math.radians(latitude)
    cosine_zenith = (
        math.sin(latitude_rad) * math.sin(declination)
        + math.cos(latitude_rad) * math.cos(declination) * math.cos(hour_angle)
    )
    cosine_zenith = float(np.clip(cosine_zenith, -1.0, 1.0))
    zenith = math.acos(cosine_zenith)
    elevation_degrees = 90.0 - math.degrees(zenith)
    azimuth = math.atan2(
        math.sin(hour_angle),
        math.cos(hour_angle) * math.sin(latitude_rad)
        - math.tan(declination) * math.cos(latitude_rad),
    )
    azimuth_degrees = (math.degrees(azimuth) + 180.0) % 360.0
    return elevation_degrees, azimuth_degrees, max(0.0, cosine_zenith)


def _erbs_decomposition(
    ghi_w_m2: float, cosine_zenith: float, day_of_year: int
) -> tuple[float, float]:
    """Derive DNI and DHI from GHI using the hourly Erbs correlation."""
    if ghi_w_m2 <= 0 or cosine_zenith <= 0:
        return 0.0, 0.0
    dni_extra = SOLAR_CONSTANT_W_M2 * (
        1.0 + 0.033 * math.cos(2.0 * math.pi * day_of_year / 365.0)
    )
    extraterrestrial_horizontal = dni_extra * cosine_zenith
    if extraterrestrial_horizontal <= 0:
        return 0.0, 0.0
    clearness_index = float(
        np.clip(ghi_w_m2 / extraterrestrial_horizontal, 0.0, 1.2)
    )
    if clearness_index <= 0.22:
        diffuse_fraction = 1.0 - 0.09 * clearness_index
    elif clearness_index <= 0.8:
        diffuse_fraction = (
            0.9511
            - 0.1604 * clearness_index
            + 4.388 * clearness_index**2
            - 16.638 * clearness_index**3
            + 12.336 * clearness_index**4
        )
    else:
        diffuse_fraction = 0.165
    diffuse_fraction = float(np.clip(diffuse_fraction, 0.0, 1.0))
    dhi = ghi_w_m2 * diffuse_fraction
    dni = max(0.0, (ghi_w_m2 - dhi) / cosine_zenith)
    return min(dni, dni_extra), dhi


def _poa_irradiance(
    *,
    ghi_w_m2: float,
    dni_w_m2: float,
    dhi_w_m2: float,
    cosine_zenith: float,
    solar_azimuth_degrees: float,
    tilt_degrees: float,
    surface_azimuth_degrees: float,
    albedo: float,
) -> float:
    if ghi_w_m2 <= 0 or cosine_zenith <= 0:
        return 0.0
    tilt = math.radians(tilt_degrees)
    zenith = math.acos(float(np.clip(cosine_zenith, 0.0, 1.0)))
    azimuth_delta = math.radians(
        solar_azimuth_degrees - surface_azimuth_degrees
    )
    cosine_incidence = (
        math.cos(zenith) * math.cos(tilt)
        + math.sin(zenith) * math.sin(tilt) * math.cos(azimuth_delta)
    )
    direct = dni_w_m2 * max(0.0, cosine_incidence)
    sky_diffuse = dhi_w_m2 * (1.0 + math.cos(tilt)) / 2.0
    ground_reflected = ghi_w_m2 * albedo * (1.0 - math.cos(tilt)) / 2.0
    return max(0.0, direct + sky_diffuse + ground_reflected)


def _orientation_power(
    *,
    capacity_kwp: float,
    poa_w_m2: float,
    air_temperature_c: float,
    wind_speed_m_s: float,
    config: PhysicalPVConfig,
) -> tuple[float, float, float]:
    if poa_w_m2 <= 0:
        return air_temperature_c, 0.0, 0.0
    wind_correction = 1.0 + 0.05 * max(0.0, wind_speed_m_s)
    cell_temperature = air_temperature_c + (
        (config.module.noct_c - 20.0) / 800.0 * poa_w_m2 / wind_correction
    )
    temperature_factor = max(
        0.0,
        1.0
        + config.module.temperature_coefficient_per_c
        * (cell_temperature - 25.0),
    )
    dc_power = capacity_kwp * poa_w_m2 / 1000.0 * temperature_factor
    dc_power *= 1.0 + config.bifacial_gain_fraction
    ac_limit = capacity_kwp / config.inverter.dc_ac_ratio
    ac_before_system_losses = min(
        ac_limit, dc_power * config.inverter.peak_efficiency
    )
    ac_power = ac_before_system_losses * (1.0 - config.system_loss_fraction)
    return cell_temperature, max(0.0, dc_power), max(0.0, ac_power)


def _prepare_weather(
    frame: pd.DataFrame, config: PhysicalPVConfig
) -> tuple[pd.DataFrame, str]:
    required = {"timestamp", "ghi_w_m2", "air_temperature_c", "wind_speed_10m_m_s"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"weather data missing columns: {sorted(missing)}")
    weather = frame.copy()
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], errors="raise")
    if weather["timestamp"].dt.tz is None:
        raise ValueError("weather timestamps must include a timezone")
    weather = weather.sort_values("timestamp").reset_index(drop=True)
    if weather["timestamp"].duplicated().any():
        raise ValueError("weather timestamps must be unique")
    if len(weather) != 24:
        raise ValueError("stage A weather must contain exactly 24 hourly rows")
    deltas = weather["timestamp"].diff().dropna()
    expected_step = timedelta(minutes=config.step_minutes)
    if not deltas.map(lambda value: value.to_pytimedelta() == expected_step).all():
        raise ValueError("weather timestamps must be continuous hourly values")
    timezone = ZoneInfo(config.location.timezone)
    local_dates = weather["timestamp"].map(
        lambda value: value.to_pydatetime().astimezone(timezone).date()
    )
    if not (local_dates == config.study_day).all():
        raise ValueError("weather timestamps do not match configured study day")
    numeric_columns = [
        "ghi_w_m2",
        "air_temperature_c",
        "wind_speed_10m_m_s",
    ]
    for column in numeric_columns:
        weather[column] = pd.to_numeric(weather[column], errors="raise")
    if weather[numeric_columns].isna().any().any():
        raise ValueError("weather data contains missing values")
    if (weather["ghi_w_m2"] < 0).any() or (
        weather["wind_speed_10m_m_s"] < 0
    ).any():
        raise ValueError("irradiance and wind speed must be non-negative")
    has_dni = "dni_w_m2" in weather.columns
    has_dhi = "dhi_w_m2" in weather.columns
    if has_dni != has_dhi:
        raise ValueError("weather must provide both DNI and DHI or neither")
    if has_dni:
        weather["dni_w_m2"] = pd.to_numeric(weather["dni_w_m2"], errors="raise")
        weather["dhi_w_m2"] = pd.to_numeric(weather["dhi_w_m2"], errors="raise")
        if (weather[["dni_w_m2", "dhi_w_m2"]] < 0).any().any():
            raise ValueError("DNI and DHI must be non-negative")
        radiation_basis = "provided_ghi_dni_dhi"
    else:
        radiation_basis = "erbs_dni_dhi_derived_from_ghi"
    return weather, radiation_basis


def calculate_physical_pv(
    weather_frame: pd.DataFrame,
    config: PhysicalPVConfig,
    *,
    weather_sha256: str = "in-memory",
    config_sha256: str = "in-memory",
) -> PhysicalPVRun:
    weather, radiation_basis = _prepare_weather(weather_frame, config)
    timezone = ZoneInfo(config.location.timezone)
    rows: list[dict[str, Any]] = []

    for source in weather.to_dict(orient="records"):
        timestamp = pd.Timestamp(source["timestamp"])
        solar_timestamp = (
            timestamp.to_pydatetime().astimezone(timezone)
            + timedelta(minutes=config.step_minutes / 2)
        )
        elevation, solar_azimuth, cosine_zenith = _solar_position(
            solar_timestamp,
            latitude=config.location.latitude,
            longitude=config.location.longitude,
        )
        ghi = float(source["ghi_w_m2"])
        if elevation <= 0:
            ghi_for_model = 0.0
        else:
            ghi_for_model = ghi
        if radiation_basis == "provided_ghi_dni_dhi":
            dni = float(source["dni_w_m2"]) if elevation > 0 else 0.0
            dhi = float(source["dhi_w_m2"]) if elevation > 0 else 0.0
        else:
            dni, dhi = _erbs_decomposition(
                ghi_for_model, cosine_zenith, solar_timestamp.timetuple().tm_yday
            )
        row: dict[str, Any] = {
            "timestamp": timestamp.isoformat(),
            "solar_elevation_degrees": elevation,
            "solar_azimuth_degrees": solar_azimuth,
            "ghi_w_m2": ghi,
            "dni_w_m2": dni,
            "dhi_w_m2": dhi,
            "air_temperature_c": float(source["air_temperature_c"]),
            "wind_speed_10m_m_s": float(source["wind_speed_10m_m_s"]),
        }
        for array in config.arrays:
            array_dc = 0.0
            array_ac = 0.0
            weighted_poa = 0.0
            weighted_temperature = 0.0
            for orientation in array.orientations:
                capacity_kwp = (
                    orientation.module_count * config.module.rated_power_wp / 1000.0
                )
                poa = _poa_irradiance(
                    ghi_w_m2=ghi_for_model,
                    dni_w_m2=dni,
                    dhi_w_m2=dhi,
                    cosine_zenith=cosine_zenith,
                    solar_azimuth_degrees=solar_azimuth,
                    tilt_degrees=orientation.tilt_degrees,
                    surface_azimuth_degrees=orientation.azimuth_degrees,
                    albedo=config.albedo,
                )
                cell_temperature, dc_power, ac_power = _orientation_power(
                    capacity_kwp=capacity_kwp,
                    poa_w_m2=poa,
                    air_temperature_c=float(source["air_temperature_c"]),
                    wind_speed_m_s=float(source["wind_speed_10m_m_s"]),
                    config=config,
                )
                prefix = f"{array.array_id}_{orientation.orientation_id}"
                row[f"{prefix}_poa_w_m2"] = poa
                row[f"{prefix}_dc_power_kw"] = dc_power
                row[f"{prefix}_ac_power_kw"] = ac_power
                array_dc += dc_power
                array_ac += ac_power
                weighted_poa += poa * capacity_kwp
                weighted_temperature += cell_temperature * capacity_kwp
            array_capacity = config.array_capacity_kwp(array.array_id)
            row[f"{array.array_id}_poa_w_m2"] = weighted_poa / array_capacity
            row[f"{array.array_id}_cell_temperature_c"] = (
                weighted_temperature / array_capacity
            )
            row[f"{array.array_id}_dc_power_kw"] = array_dc
            row[f"{array.array_id}_ac_power_kw"] = array_ac
        rows.append(row)

    components = pd.DataFrame(rows)
    array_by_id = {item.array_id: item for item in config.arrays}
    scenario_frames: dict[str, pd.DataFrame] = {}
    manifests: dict[str, dict[str, Any]] = {}
    quality_reports: dict[str, dict[str, Any]] = {}

    for scenario in config.scenarios:
        base_columns = [
            "timestamp",
            "solar_elevation_degrees",
            "solar_azimuth_degrees",
            "ghi_w_m2",
            "dni_w_m2",
            "dhi_w_m2",
            "air_temperature_c",
            "wind_speed_10m_m_s",
        ]
        frame = components[base_columns].copy()
        for array_id in scenario.array_ids:
            for suffix in (
                "poa_w_m2",
                "cell_temperature_c",
                "dc_power_kw",
                "ac_power_kw",
            ):
                column = f"{array_id}_{suffix}"
                frame[column] = components[column]
        ac_columns = [f"{array_id}_ac_power_kw" for array_id in scenario.array_ids]
        frame["campus_pv_power_kw"] = frame[ac_columns].sum(axis=1)
        scenario_frames[scenario.scenario_id] = frame

        max_expected_ac = (
            scenario.expected_capacity_kwp
            / config.inverter.dc_ac_ratio
            * (1.0 - config.system_loss_fraction)
        )
        night_mask = components["solar_elevation_degrees"] <= 0
        night_max = float(frame.loc[night_mask, "campus_pv_power_kw"].max())
        if math.isnan(night_max):
            night_max = 0.0
        nonnegative = bool((frame["campus_pv_power_kw"] >= -1e-9).all())
        capacity_limit_ok = bool(
            (frame["campus_pv_power_kw"] <= max_expected_ac + 1e-6).all()
        )
        night_zero = night_max <= 1e-6
        hard_gate_passed = nonnegative and capacity_limit_ok and night_zero
        if not hard_gate_passed:
            raise ValueError(f"physical PV quality gate failed: {scenario.scenario_id}")

        array_capacities = {
            array_id: config.array_capacity_kwp(array_id)
            for array_id in scenario.array_ids
        }
        manifests[scenario.scenario_id] = {
            "dataset_id": f"hkust_gz_physical_pv_{scenario.scenario_id}",
            "dataset_version": (
                f"{MODEL_VERSION}:config-{config_sha256[:16]}:"
                f"weather-{weather_sha256[:16]}"
            ),
            "model_version": MODEL_VERSION,
            "config_id": config.config_id,
            "config_sha256": config_sha256,
            "weather_sha256": weather_sha256,
            "weather_radiation_basis": radiation_basis,
            "site_id": config.site_id,
            "study_day": config.study_day.isoformat(),
            "timezone": config.location.timezone,
            "latitude": config.location.latitude,
            "longitude": config.location.longitude,
            "scenario_id": scenario.scenario_id,
            "scenario_name": scenario.name,
            "capacity_kwp": scenario.expected_capacity_kwp,
            "array_capacities_kwp": array_capacities,
            "module": config.module.model_dump(mode="json"),
            "module_area_m2": config.module.area_m2,
            "module_nominal_efficiency": config.module.nominal_efficiency,
            "inverter": config.inverter.model_dump(mode="json"),
            "system_loss_fraction": config.system_loss_fraction,
            "albedo": config.albedo,
            "bifacial_gain_fraction": config.bifacial_gain_fraction,
            "arrays": [
                array_by_id[array_id].model_dump(mode="json")
                for array_id in scenario.array_ids
            ],
            "resolution_minutes": config.step_minutes,
            "rows": len(frame),
            "simulation_only": config.simulation_only,
            "executable": config.executable,
            "assumptions": config.assumptions,
        }
        quality_reports[scenario.scenario_id] = {
            "status": "warning",
            "hard_gate_passed": hard_gate_passed,
            "scenario_id": scenario.scenario_id,
            "row_count": len(frame),
            "missing_value_count": int(frame.isna().sum().sum()),
            "negative_power_count": int((frame["campus_pv_power_kw"] < 0).sum()),
            "night_max_power_kw": night_max,
            "capacity_limit_ok": capacity_limit_ok,
            "capacity_kwp": scenario.expected_capacity_kwp,
            "daily_pv_energy_kwh": float(frame["campus_pv_power_kw"].sum()),
            "peak_pv_power_kw": float(frame["campus_pv_power_kw"].max()),
            "capacity_factor_for_study_day": float(
                frame["campus_pv_power_kw"].sum()
                / (scenario.expected_capacity_kwp * 24.0)
            ),
            "array_energy_kwh": {
                array_id: float(frame[f"{array_id}_ac_power_kw"].sum())
                for array_id in scenario.array_ids
            },
            "checks": {
                "timestamps_complete_hourly": True,
                "nonnegative_power": nonnegative,
                "capacity_limit": capacity_limit_ok,
                "night_power_zero": night_zero,
            },
            "warnings": [
                "组件和阵列方向仍含规划假设，输出不是实测发电量。",
                "DNI/DHI若未提供则使用Erbs方法由GHI推导。",
                "未计入遮挡和双面增益，逆变器与损失参数待铭牌核验。",
                "结果仅用于simulation_only容量情景分析。",
            ],
        }

    return PhysicalPVRun(
        components=components,
        scenarios=scenario_frames,
        manifests=manifests,
        quality_reports=quality_reports,
    )


def run_physical_pv_from_files(
    config_path: str | Path, *, project_root: str | Path
) -> PhysicalPVRun:
    root = Path(project_root)
    config_bytes = Path(config_path).read_bytes()
    config = PhysicalPVConfig.model_validate_json(config_bytes)
    weather_path = root / config.weather_csv
    weather_bytes = weather_path.read_bytes()
    weather = pd.read_csv(weather_path)
    return calculate_physical_pv(
        weather,
        config,
        weather_sha256=_sha256_bytes(weather_bytes),
        config_sha256=_sha256_bytes(config_bytes),
    )


def build_stage_a_sensitivity(
    weather_frame: pd.DataFrame, config: PhysicalPVConfig
) -> pd.DataFrame:
    """Compare facade orientation distributions and system-loss assumptions."""
    rows: list[dict[str, Any]] = []
    for facade_case in config.sensitivity.facade_orientation_cases:
        for system_loss in config.sensitivity.system_loss_fractions:
            payload = config.model_dump(mode="json")
            payload["config_id"] = (
                f"{config.config_id}:{facade_case.case_id}:loss-{system_loss:g}"
            )
            payload["system_loss_fraction"] = system_loss
            for array in payload["arrays"]:
                if array["array_id"] != "facade":
                    continue
                for orientation in array["orientations"]:
                    orientation["module_count"] = facade_case.module_counts[
                        orientation["orientation_id"]
                    ]
            variant = PhysicalPVConfig.model_validate(payload)
            run = calculate_physical_pv(weather_frame, variant)
            full = run.quality_reports["full_build_20077kwp"]
            array_energy = full["array_energy_kwh"]
            rows.append(
                {
                    "facade_orientation_case": facade_case.case_id,
                    "facade_orientation_description": facade_case.description,
                    "system_loss_fraction": system_loss,
                    "capacity_kwp": full["capacity_kwp"],
                    "daily_pv_energy_kwh": full["daily_pv_energy_kwh"],
                    "peak_pv_power_kw": full["peak_pv_power_kw"],
                    "capacity_factor_for_study_day": full[
                        "capacity_factor_for_study_day"
                    ],
                    "roof_energy_kwh": array_energy["roof"],
                    "facade_energy_kwh": array_energy["facade"],
                    "parking_energy_kwh": array_energy["parking"],
                    "hard_gate_passed": full["hard_gate_passed"],
                }
            )
    result = pd.DataFrame(rows).sort_values(
        ["facade_orientation_case", "system_loss_fraction"]
    )
    baseline = result.loc[
        (result["facade_orientation_case"] == "equal_cardinal_split")
        & np.isclose(result["system_loss_fraction"], config.system_loss_fraction),
        "daily_pv_energy_kwh",
    ]
    if len(baseline) != 1:
        raise ValueError("sensitivity configuration must include the base case")
    baseline_energy = float(baseline.iloc[0])
    result["energy_change_vs_base_percent"] = (
        result["daily_pv_energy_kwh"] / baseline_energy - 1.0
    ) * 100.0
    return result.reset_index(drop=True)


def write_stage_a_sensitivity(
    sensitivity: pd.DataFrame,
    config: PhysicalPVConfig,
    *,
    project_root: str | Path,
) -> list[Path]:
    root = Path(project_root)
    prefix = root / config.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix.name}_stage_a_sensitivity_{config.study_day.isoformat()}"
    csv_path = prefix.parent / f"{stem}.csv"
    manifest_path = prefix.parent / f"{stem}_manifest.json"
    sensitivity.to_csv(csv_path, index=False)
    config_json = config.model_dump_json().encode("utf-8")
    manifest = {
        "dataset_id": "hkust_gz_physical_pv_stage_a_sensitivity",
        "dataset_version": f"{MODEL_VERSION}:config-{_sha256_bytes(config_json)[:16]}",
        "model_version": MODEL_VERSION,
        "config_id": config.config_id,
        "study_day": config.study_day.isoformat(),
        "base_facade_case": "equal_cardinal_split",
        "base_system_loss_fraction": config.system_loss_fraction,
        "facade_orientation_cases": [
            item.model_dump(mode="json")
            for item in config.sensitivity.facade_orientation_cases
        ],
        "system_loss_fractions": config.sensitivity.system_loss_fractions,
        "row_count": len(sensitivity),
        "all_hard_gates_passed": bool(sensitivity["hard_gate_passed"].all()),
        "simulation_only": config.simulation_only,
        "executable": config.executable,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return [csv_path, manifest_path]


def write_physical_pv_outputs(
    run: PhysicalPVRun, config: PhysicalPVConfig, *, project_root: str | Path
) -> list[Path]:
    root = Path(project_root)
    prefix = root / config.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    date_label = config.study_day.isoformat()
    resolution = f"{config.step_minutes}min"
    written: list[Path] = []

    components_path = prefix.parent / (
        f"{prefix.name}_components_{date_label}_{resolution}.csv"
    )
    run.components.to_csv(components_path, index=False)
    written.append(components_path)

    for scenario in config.scenarios:
        stem = f"{prefix.name}_{scenario.scenario_id}_{date_label}_{resolution}"
        csv_path = prefix.parent / f"{stem}.csv"
        manifest_path = prefix.parent / f"{stem}_manifest.json"
        quality_path = prefix.parent / f"{stem}_quality.json"
        run.scenarios[scenario.scenario_id].to_csv(csv_path, index=False)
        manifest_path.write_text(
            json.dumps(
                run.manifests[scenario.scenario_id], ensure_ascii=False, indent=2
            ),
            encoding="utf-8",
        )
        quality_path.write_text(
            json.dumps(
                run.quality_reports[scenario.scenario_id],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        written.extend((csv_path, manifest_path, quality_path))
    return written
