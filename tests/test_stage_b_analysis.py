from pathlib import Path

import pandas as pd

from energyops.stage_b_analysis import (
    run_stage_b_analysis,
    write_stage_b_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_CONFIG = PROJECT_ROOT / "config/hkust_gz_physical_pv.json"
CAMPUS_CONFIG = PROJECT_ROOT / "config/scenarios/hkust_gz_campus_baseline.json"


def run_full_build_matrix():
    return run_stage_b_analysis(
        project_root=PROJECT_ROOT,
        physical_config_path=PHYSICAL_CONFIG,
        campus_config_path=CAMPUS_CONFIG,
        pv_scenario_ids=["full_build_20077kwp"],
    )


def test_stage_b_runs_three_verified_objectives_for_each_battery_size():
    run = run_full_build_matrix()

    assert run.analysis.all_verification_passed
    assert len(run.analysis.baselines) == 1
    assert len(run.analysis.plans) == 9
    assert {item.plan_id for item in run.analysis.plans} == {
        "economic",
        "carbon",
        "carbon_cost_capped",
    }
    assert all(item.verification_passed for item in run.analysis.plans)
    assert run.analysis.simulation_only
    assert not run.analysis.executable


def test_stage_b_quantifies_curtailment_and_potential_export():
    run = run_full_build_matrix()
    baseline = run.analysis.baselines[0]

    assert baseline.curtailed_energy_kwh > 30_000
    assert baseline.curtailment_rate > 0.5
    assert baseline.potential_export_peak_kw > 5_000

    medium = next(
        item
        for item in run.analysis.plans
        if item.battery_case_id == "medium_150kw_400kwh"
        and item.plan_id == "carbon_cost_capped"
    )
    assert medium.curtailed_energy_kwh < baseline.curtailed_energy_kwh
    assert medium.avoided_curtailment_vs_no_battery_kwh > 0
    assert medium.residual_surplus_peak_kw > 5_000
    assert medium.soc_at_max_hours > 0


def test_stage_b_reports_marginal_storage_improvement():
    run = run_full_build_matrix()
    carbon = sorted(
        (item for item in run.analysis.plans if item.plan_id == "carbon"),
        key=lambda item: item.battery_capacity_kwh,
    )

    assert [item.battery_capacity_kwh for item in carbon] == [200.0, 400.0, 800.0]
    assert all(item.marginal_avoided_curtailment_kwh > 0 for item in carbon)
    assert carbon[0].avoided_curtailment_vs_no_battery_kwh < carbon[1].avoided_curtailment_vs_no_battery_kwh < carbon[2].avoided_curtailment_vs_no_battery_kwh


def test_stage_b_writes_summary_report_and_verified_schedules(tmp_path: Path):
    run = run_stage_b_analysis(
        project_root=PROJECT_ROOT,
        physical_config_path=PHYSICAL_CONFIG,
        campus_config_path=CAMPUS_CONFIG,
        pv_scenario_ids=["roof_only_10350kwp"],
        battery_case_ids=["medium_150kw_400kwh"],
    )

    paths = write_stage_b_outputs(run, tmp_path)

    assert (tmp_path / "stage_b_analysis.json").exists()
    assert (tmp_path / "verified_plan_summary.csv").exists()
    assert (tmp_path / "STAGE_B_REPORT.md").exists()
    assert len(paths) == 11
    summary = pd.read_csv(tmp_path / "verified_plan_summary.csv")
    assert len(summary) == 3
    assert summary["verification_passed"].all()
