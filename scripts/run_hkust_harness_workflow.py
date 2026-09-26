"""Run a validated 24-hour HKUST(GZ) EnergyOps scenario."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energyops.harness_workflow import run_energyops_workflow  # noqa: E402
from energyops.contracts import ParsedCampusRequest  # noqa: E402
from energyops.providers import DeepSeekAPIProvider, MockProvider  # noqa: E402
from energyops.scenario import build_hourly_inputs  # noqa: E402
from energyops.scenario_comparison import compare_battery_scenarios  # noqa: E402
from energyops.scenario_config import (  # noqa: E402
    load_scenario_config,
    write_scenario_snapshot,
)
from energyops.trace import TraceWriter  # noqa: E402


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_model(path: Path, model) -> None:
    path.write_text(
        json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/scenarios/hkust_gz_campus_baseline.json",
    )
    parser.add_argument(
        "--pv",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--pv-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--pv-quality",
        type=Path,
        default=None,
    )
    parser.add_argument("--load", type=Path)
    parser.add_argument("--load-manifest", type=Path)
    parser.add_argument("--load-quality", type=Path)
    parser.add_argument(
        "--provider", choices=("mock", "deepseek-api"), default="mock"
    )
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--solver")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs/hkust_gz_campus_harness",
    )
    parser.add_argument(
        "request",
        nargs="?",
        default="为港科广真实分表负荷与本地天气光伏、校园储能假设生成24小时仿真调度，并解释验证结果。",
    )
    args = parser.parse_args()

    scenario_config = load_scenario_config(args.config)

    def configured_path(override: Path | None, configured: str) -> Path:
        return override or PROJECT_ROOT / configured

    pv_path = configured_path(args.pv, scenario_config.pv.csv)
    pv_manifest_path = configured_path(args.pv_manifest, scenario_config.pv.manifest)
    pv_quality_path = configured_path(args.pv_quality, scenario_config.pv.quality)
    load_path = configured_path(args.load, scenario_config.load.csv)
    load_manifest_path = configured_path(
        args.load_manifest, scenario_config.load.manifest
    )
    load_quality_path = configured_path(args.load_quality, scenario_config.load.quality)

    pv_manifest = read_json(pv_manifest_path)
    quality_reports: dict[str, dict[str, object]] = {
        "pv": read_json(pv_quality_path)
    }
    measured_load = load_path.exists()
    if measured_load:
        load_manifest = read_json(load_manifest_path)
        quality_reports["load"] = read_json(load_quality_path)
        dataset_version = (
            f"pv={pv_manifest['dataset_version']};load={load_manifest['dataset_version']}"
        )
    else:
        quality_reports["load"] = {
            "status": "warning",
            "evidence_id": "load:synthetic-preintegration-v1",
            "findings": ["校园边界表未批准，当前仅使用透明 synthetic 负荷联调业务流程。"],
        }
        dataset_version = f"pv={pv_manifest['dataset_version']};load=synthetic-preintegration-v1"

    inputs = build_hourly_inputs(
        pv_path,
        dataset_version=dataset_version,
        load_csv=load_path if measured_load else None,
        buy_price_cny_per_kwh_by_hour=(
            scenario_config.tariff.buy_price_cny_per_kwh_by_hour
        ),
        sell_price_cny_per_kwh=scenario_config.tariff.sell_price_cny_per_kwh,
    )
    if args.provider == "deepseek-api":
        provider = DeepSeekAPIProvider(model=args.model)
    else:
        provider = MockProvider(
            ParsedCampusRequest(
                scenario_name=scenario_config.scenario_name,
                site_id=scenario_config.site_id,
                load_dataset_id=scenario_config.load.dataset_id,
                pv_profile_id=scenario_config.pv.dataset_id,
                battery_config_id=scenario_config.battery.asset_id,
                step_minutes=scenario_config.step_minutes,
                horizon_steps=scenario_config.horizon_steps,
                objective=scenario_config.objective_mode,
                allow_grid_charging=scenario_config.grid.allow_grid_charging,
                allow_grid_export=scenario_config.grid.allow_grid_export,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_scenario_snapshot(args.output_dir / "scenario_config.json", scenario_config)
    trace_path = args.output_dir / "trace.jsonl"
    trace_path.write_text("", encoding="utf-8")
    run_id = f"run_{scenario_config.scenario_id}"
    outcome = run_energyops_workflow(
        run_id=run_id,
        user_text=args.request,
        inputs=inputs,
        quality_reports=quality_reports,
        measured_load=measured_load,
        provider=provider,
        trace=TraceWriter(trace_path, run_id=run_id),
        solver_name=args.solver,
        scenario_config=scenario_config,
    )
    write_model(args.output_dir / "workflow_result.json", outcome)
    for filename, model in (
        ("request.json", outcome.dispatch_request),
        ("schedule.json", outcome.dispatch_result),
        ("verification.json", outcome.verification),
        ("evidence.json", outcome.evidence),
        ("explanation.json", outcome.explanation),
    ):
        if model is not None:
            write_model(args.output_dir / filename, model)

    print(f"workflow status: {outcome.status.value}")
    print(f"provider: {provider.name}")
    print(f"simulation_only: {outcome.simulation_only}")
    print(f"executable: {outcome.executable}")
    if outcome.failure_code:
        print(f"failure_code: {outcome.failure_code}")
    if outcome.explanation:
        print(outcome.explanation.summary)
    if outcome.dispatch_request is not None and outcome.verification is not None:
        comparison = compare_battery_scenarios(
            inputs,
            outcome.dispatch_request,
            solver_name=args.solver,
        )
        write_model(args.output_dir / "battery_comparison.json", comparison)
        print("battery scenarios: small, medium, large (simulation assumptions)")


if __name__ == "__main__":
    main()
