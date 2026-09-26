"""Generate auditable roof, facade and parking PV planning scenarios."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energyops.physical_pv import (  # noqa: E402
    build_stage_a_sensitivity,
    load_physical_pv_config,
    run_physical_pv_from_files,
    write_physical_pv_outputs,
    write_stage_a_sensitivity,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/hkust_gz_physical_pv.json",
    )
    args = parser.parse_args()

    config = load_physical_pv_config(args.config)
    run = run_physical_pv_from_files(args.config, project_root=PROJECT_ROOT)
    paths = write_physical_pv_outputs(run, config, project_root=PROJECT_ROOT)
    weather = pd.read_csv(PROJECT_ROOT / config.weather_csv)
    sensitivity = build_stage_a_sensitivity(weather, config)
    paths.extend(
        write_stage_a_sensitivity(sensitivity, config, project_root=PROJECT_ROOT)
    )

    print(f"physical PV model: {config.config_id}")
    print(
        "array capacities: "
        + ", ".join(
            f"{array.array_id}={config.array_capacity_kwp(array.array_id):.1f} kWp"
            for array in config.arrays
        )
    )
    for scenario in config.scenarios:
        quality = run.quality_reports[scenario.scenario_id]
        print(
            f"{scenario.scenario_id}: "
            f"energy={quality['daily_pv_energy_kwh']:.3f} kWh, "
            f"peak={quality['peak_pv_power_kw']:.3f} kW, "
            f"quality={quality['status']}"
        )
    lowest = sensitivity.loc[sensitivity["daily_pv_energy_kwh"].idxmin()]
    highest = sensitivity.loc[sensitivity["daily_pv_energy_kwh"].idxmax()]
    print(
        "sensitivity energy range: "
        f"{lowest['daily_pv_energy_kwh']:.3f}–"
        f"{highest['daily_pv_energy_kwh']:.3f} kWh"
    )
    print(f"wrote {len(paths)} files")


if __name__ == "__main__":
    main()
