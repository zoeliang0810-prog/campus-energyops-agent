"""Build a weather-derived HKUST(GZ) PV profile from NASA POWER hourly data."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energyops.scenario_config import load_scenario_config  # noqa: E402


NASA_POWER_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/hourly/point"


def fetch_power_json(*, latitude: float, longitude: float, start: str, end: str):
    query = urllib.parse.urlencode(
        {
            "parameters": "ALLSKY_SFC_SW_DWN,T2M,WS10M",
            "community": "RE",
            "longitude": longitude,
            "latitude": latitude,
            "start": start,
            "end": end,
            "format": "JSON",
            "time-standard": "UTC",
        }
    )
    url = f"{NASA_POWER_ENDPOINT}?{query}"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "EnergyOps-campus-weather-fixture/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read()
    return url, raw, json.loads(raw)


def pv_power_per_kwp(
    ghi_w_m2: float,
    air_temperature_c: float,
    *,
    system_loss_fraction: float = 0.14,
    noct_c: float = 45.0,
    temperature_coefficient_per_c: float = -0.004,
) -> tuple[float, float]:
    """Transparent PVWatts-style proxy using horizontal irradiance.

    Tilt, azimuth, shading, module and inverter details are unavailable, so the
    output is a planning profile rather than measured or bankable production.
    """
    if ghi_w_m2 <= 0:
        return 0.0, air_temperature_c
    cell_temperature_c = air_temperature_c + (noct_c - 20.0) / 800.0 * ghi_w_m2
    temperature_factor = max(
        0.0, 1.0 + temperature_coefficient_per_c * (cell_temperature_c - 25.0)
    )
    dc_per_kwp = ghi_w_m2 / 1000.0 * temperature_factor
    ac_per_kwp = min(1.0, dc_per_kwp * (1.0 - system_loss_fraction))
    return ac_per_kwp, cell_temperature_c


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/scenarios/hkust_gz_campus_baseline.json",
    )
    parser.add_argument(
        "--weather-output",
        type=Path,
        default=PROJECT_ROOT
        / "data/fixtures/hkust_gz_weather_2026-02-19_60min.csv",
    )
    args = parser.parse_args()

    config = load_scenario_config(args.config)
    timezone = ZoneInfo(config.location.timezone)
    study_day = config.study_day
    query_start = study_day - timedelta(days=1)
    query_end = study_day + timedelta(days=1)
    source_url, raw, payload = fetch_power_json(
        latitude=config.location.latitude,
        longitude=config.location.longitude,
        start=query_start.strftime("%Y%m%d"),
        end=query_end.strftime("%Y%m%d"),
    )
    parameters = payload["properties"]["parameter"]
    irradiance = parameters["ALLSKY_SFC_SW_DWN"]
    temperature = parameters["T2M"]
    wind = parameters["WS10M"]
    rows: list[dict[str, object]] = []
    for key, ghi_value in irradiance.items():
        timestamp_utc = datetime.strptime(key, "%Y%m%d%H").replace(tzinfo=UTC)
        timestamp_local = timestamp_utc.astimezone(timezone)
        if timestamp_local.date() != study_day:
            continue
        values = (float(ghi_value), float(temperature[key]), float(wind[key]))
        if any(value <= -900 for value in values):
            raise ValueError(f"NASA POWER fill value encountered at {key}")
        per_kwp, cell_temperature = pv_power_per_kwp(values[0], values[1])
        rows.append(
            {
                "timestamp": timestamp_local.isoformat(),
                "ghi_w_m2": values[0],
                "air_temperature_c": values[1],
                "wind_speed_10m_m_s": values[2],
                "estimated_cell_temperature_c": cell_temperature,
                "pv_power_per_kwp_kw": per_kwp,
                "campus_pv_power_kw": per_kwp * config.pv.capacity_kwp,
            }
        )
    frame = pd.DataFrame(rows).sort_values("timestamp")
    if len(frame) != 24 or frame.isna().any().any():
        raise ValueError("weather-derived PV fixture must contain 24 complete hours")

    pv_path = PROJECT_ROOT / config.pv.csv
    manifest_path = PROJECT_ROOT / config.pv.manifest
    quality_path = PROJECT_ROOT / config.pv.quality
    for path in (args.weather_output, pv_path, manifest_path, quality_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    frame[
        ["timestamp", "ghi_w_m2", "air_temperature_c", "wind_speed_10m_m_s"]
    ].to_csv(args.weather_output, index=False)
    frame.to_csv(pv_path, index=False)

    digest = hashlib.sha256(raw).hexdigest()
    dataset_version = f"sha256:{digest[:16]}:pv-model-v1:{config.pv.capacity_kwp:g}kwp"
    manifest = {
        "dataset_id": config.pv.dataset_id,
        "dataset_version": dataset_version,
        "site_id": config.site_id,
        "study_day": study_day.isoformat(),
        "source_name": "NASA POWER Hourly API",
        "source_url": source_url,
        "source_sha256": digest,
        "source_time_standard": "UTC",
        "output_timezone": config.location.timezone,
        "latitude": config.location.latitude,
        "longitude": config.location.longitude,
        "capacity_kwp": config.pv.capacity_kwp,
        "capacity_status": config.pv.capacity_status,
        "rows": len(frame),
        "resolution_minutes": 60,
        "variable": "campus_pv_power_kw",
        "unit": "kW",
        "model": {
            "irradiance_basis": "ALLSKY_SFC_SW_DWN horizontal irradiance proxy",
            "system_loss_fraction": 0.14,
            "noct_c": 45.0,
            "temperature_coefficient_per_c": -0.004,
            "power_limit_per_kwp_kw": 1.0,
        },
        "assumptions": [
            "光伏装机容量为场景假设，不是校园铭牌",
            "缺少倾角、方位、遮挡、组件和逆变器参数，使用水平辐照度透明近似",
            "NASA POWER为卫星和再分析气象数据，不是校园气象站实测",
        ],
    }
    quality = {
        "status": "warning",
        "evidence_id": f"pv-weather:{digest[:16]}",
        "row_count": len(frame),
        "missing_hour_count": 0,
        "negative_power_count": int((frame["campus_pv_power_kw"] < 0).sum()),
        "capacity_kwp": config.pv.capacity_kwp,
        "daily_pv_energy_kwh": float(frame["campus_pv_power_kw"].sum()),
        "peak_pv_power_kw": float(frame["campus_pv_power_kw"].max()),
        "findings": [
            "天气位置已切换为港科广南沙校园附近网格，不再使用九江工业园光伏曲线。",
            f"光伏容量 {config.pv.capacity_kwp:g} kWp 为未核验场景假设。",
            "光伏功率由NASA POWER天气推算，不是校园光伏实测。",
        ],
        "warnings": [
            "所有结果必须保持simulation_only=true、executable=false。",
            "真实控制或投资分析前必须补充校园光伏铭牌、阵列参数和现场发电数据。",
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    quality_path.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"weather rows: {len(frame)}")
    print(f"PV capacity assumption: {config.pv.capacity_kwp:g} kWp")
    print(f"PV energy: {quality['daily_pv_energy_kwh']:.3f} kWh")
    print(f"PV peak: {quality['peak_pv_power_kw']:.3f} kW")
    print(f"wrote: {pv_path}")


if __name__ == "__main__":
    main()
