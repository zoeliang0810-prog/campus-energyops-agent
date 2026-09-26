"""Run the stage B PV-size, battery-size and economic/carbon matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energyops.stage_b_analysis import (  # noqa: E402
    run_stage_b_analysis,
    write_stage_b_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--physical-config",
        type=Path,
        default=PROJECT_ROOT / "config/hkust_gz_physical_pv.json",
    )
    parser.add_argument(
        "--campus-config",
        type=Path,
        default=PROJECT_ROOT / "config/scenarios/hkust_gz_campus_baseline.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/hkust_gz_stage_b",
    )
    parser.add_argument(
        "--carbon-pattern",
        choices=("flat", "day_low_night_high", "volatile"),
        default="day_low_night_high",
    )
    parser.add_argument("--max-cost-increase-pct", type=float, default=5.0)
    parser.add_argument("--solver")
    args = parser.parse_args()

    run = run_stage_b_analysis(
        project_root=PROJECT_ROOT,
        physical_config_path=args.physical_config,
        campus_config_path=args.campus_config,
        carbon_pattern_id=args.carbon_pattern,
        max_cost_increase_pct=args.max_cost_increase_pct,
        solver_name=args.solver,
    )
    paths = write_stage_b_outputs(run, args.output_dir)

    print(f"analysis version: {run.analysis.analysis_version}")
    print(f"verified plans: {len(run.analysis.plans)}")
    print(f"all verification passed: {run.analysis.all_verification_passed}")
    for baseline in run.analysis.baselines:
        print(
            f"{baseline.pv_scenario_id}: no-battery curtailment="
            f"{baseline.curtailed_energy_kwh:.3f} kWh "
            f"({baseline.curtailment_rate * 100:.2f}%), "
            f"potential export peak={baseline.potential_export_peak_kw:.3f} kW"
        )
    print(f"wrote {len(paths)} files to {args.output_dir}")


if __name__ == "__main__":
    main()
