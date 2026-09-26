"""Run the stage-one synthetic economic/carbon EnergyOps decision demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energyops.contracts import ParsedCampusRequest  # noqa: E402
from energyops.dual_objective_workflow import (  # noqa: E402
    run_dual_objective_workflow,
)
from energyops.scenario import build_dispatch_request, build_hourly_inputs  # noqa: E402
from energyops.scenario_config import load_scenario_config  # noqa: E402
from energyops.synthetic_inputs import (  # noqa: E402
    combine_decision_inputs,
    write_synthetic_fixture_set,
)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/scenarios/hkust_gz_campus_baseline.json",
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=PROJECT_ROOT / "data/fixtures/synthetic_decision",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/hkust_gz_dual_objective_demo",
    )
    parser.add_argument(
        "--forecast-scenario",
        choices=("baseline", "low_pv", "high_evening_load"),
        default="baseline",
    )
    parser.add_argument(
        "--carbon-scenario",
        choices=("flat", "day_low_night_high", "volatile"),
        default="day_low_night_high",
    )
    parser.add_argument("--solver")
    parser.add_argument(
        "request",
        nargs="?",
        default="比较明天的经济和低碳调度，优先减碳，但电费不能比经济方案高出5%。",
    )
    args = parser.parse_args()

    config = load_scenario_config(args.config)
    pv_manifest = read_json(PROJECT_ROOT / config.pv.manifest)
    load_manifest = read_json(PROJECT_ROOT / config.load.manifest)
    source_version = (
        f"pv={pv_manifest['dataset_version']};load={load_manifest['dataset_version']}"
    )
    base_inputs = build_hourly_inputs(
        PROJECT_ROOT / config.pv.csv,
        dataset_version=source_version,
        load_csv=PROJECT_ROOT / config.load.csv,
        buy_price_cny_per_kwh_by_hour=(
            config.tariff.buy_price_cny_per_kwh_by_hour
        ),
        sell_price_cny_per_kwh=config.tariff.sell_price_cny_per_kwh,
    )
    forecasts, carbon = write_synthetic_fixture_set(args.fixture_dir, base_inputs)
    forecast = forecasts[args.forecast_scenario]
    carbon_factors = carbon[args.carbon_scenario]
    inputs = combine_decision_inputs(
        forecast,
        carbon_factors,
        buy_prices=config.tariff.buy_price_cny_per_kwh_by_hour,
        sell_price=config.tariff.sell_price_cny_per_kwh,
    )
    parsed = ParsedCampusRequest(
        scenario_name=config.scenario_name,
        site_id=config.site_id,
        load_dataset_id=forecast.metadata.dataset_id,
        pv_profile_id=forecast.metadata.dataset_id,
        battery_config_id=config.battery.asset_id,
        step_minutes=config.step_minutes,
        horizon_steps=config.horizon_steps,
        objective=config.objective_mode,
        allow_grid_charging=config.grid.allow_grid_charging,
        allow_grid_export=config.grid.allow_grid_export,
    )
    base_request = build_dispatch_request(
        parsed,
        inputs,
        measured_load=True,
        scenario_config=config,
    )
    outcome = run_dual_objective_workflow(
        run_id="run_hkust_gz_dual_objective_synthetic_v1",
        user_text=args.request,
        inputs=inputs,
        base_request=base_request,
        solver_name=args.solver,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "selected_forecast.json", forecast)
    write_json(args.output_dir / "selected_carbon_factors.json", carbon_factors)
    write_json(args.output_dir / "combined_inputs.json", inputs)
    write_json(args.output_dir / "workflow_result.json", outcome)
    if outcome.comparison is not None:
        write_json(args.output_dir / "comparison.json", outcome.comparison)
    if outcome.explanation is not None:
        write_json(args.output_dir / "explanation.json", outcome.explanation)
    for index, plan in enumerate(outcome.plans):
        name = ("economic", "carbon", "carbon_cost_capped")[index]
        write_json(args.output_dir / f"request_{name}.json", plan.request)
        write_json(args.output_dir / f"schedule_{name}.json", plan.dispatch)
        write_json(args.output_dir / f"verification_{name}.json", plan.verification)

    print(f"workflow status: {outcome.status.value}")
    print("provider: mock")
    print(f"forecast scenario: {args.forecast_scenario}")
    print(f"carbon scenario: {args.carbon_scenario}")
    print(f"simulation_only: {outcome.simulation_only}")
    print(f"executable: {outcome.executable}")
    if outcome.failure_code:
        print(f"failure_code: {outcome.failure_code}")
    if outcome.comparison:
        for plan in outcome.comparison.plans:
            print(
                f"{plan.plan_id}: cost={plan.total_cost_cny:.3f} CNY, "
                f"emissions={plan.grid_emissions_kgco2e:.3f} kgCO2e"
            )
        print(f"recommended: {outcome.comparison.recommended_plan_id}")


if __name__ == "__main__":
    main()
