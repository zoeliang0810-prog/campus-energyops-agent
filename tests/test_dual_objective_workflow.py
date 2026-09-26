from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from energyops.contracts import (
    BatteryConfig,
    DispatchInputSeries,
    DispatchRequest,
    DispatchSeriesPoint,
    ObjectiveMode,
    OperatingMode,
    WorkflowStatus,
)
from energyops.dual_objective_workflow import (
    DUAL_OBJECTIVE_TOOL_SEQUENCE,
    parse_mock_decision_intent,
    run_dual_objective_workflow,
)


START = datetime(2026, 2, 19, tzinfo=ZoneInfo("Asia/Shanghai"))


def build_inputs() -> DispatchInputSeries:
    points = []
    for hour in range(24):
        price = 0.20 if hour < 8 else 1.00 if 10 <= hour < 18 else 0.55
        carbon = 0.75 if hour < 8 or 18 <= hour < 22 else 0.25
        points.append(
            DispatchSeriesPoint(
                timestamp=START + timedelta(hours=hour),
                pv_power_kw=0,
                load_power_kw=100,
                buy_price_cny_per_kwh=price,
                sell_price_cny_per_kwh=0,
                carbon_factor_kgco2e_per_kwh=carbon,
            )
        )
    return DispatchInputSeries(
        dataset_version="forecast=synthetic-v1;carbon=synthetic-v1",
        points=points,
        source="synthetic_fixture",
        generated_at=START - timedelta(hours=6),
        power_unit="kW",
        carbon_factor_unit="kgCO2e/kWh",
        forecast_model_version="synthetic-perfect-v1",
        forecast_confidence=0.85,
        carbon_factor_version="synthetic-carbon-v1",
        verified=False,
        simulation_only=True,
        assumptions=["test-only synthetic input"],
    )


def build_request(inputs: DispatchInputSeries) -> DispatchRequest:
    return DispatchRequest(
        request_id="dual_test",
        scenario_name="synthetic dual objective test",
        site_id="hkust_gz",
        dataset_version=inputs.dataset_version,
        operating_mode=OperatingMode.SIMULATION_ONLY,
        horizon_start=START,
        horizon_steps=24,
        step_minutes=60,
        grid_import_limit_kw=150,
        grid_export_limit_kw=0,
        allow_grid_charging=True,
        allow_grid_export=False,
        tariff_region="synthetic",
        tariff_version="synthetic-tou-v1",
        objective_mode=ObjectiveMode.COST,
        battery=BatteryConfig(
            asset_id="synthetic-bess",
            capacity_kwh=100,
            initial_energy_kwh=50,
            min_energy_kwh=10,
            max_energy_kwh=90,
            max_charge_kw=30,
            max_discharge_kw=30,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
            degradation_cost_cny_per_kwh_throughput=0.01,
        ),
        required_terminal_energy_kwh=50,
        assumptions=["synthetic unit-test scenario"],
    )


def test_mock_intent_extracts_carbon_and_cost_limit():
    intent = parse_mock_decision_intent(
        "比较经济与低碳方案，优先减碳，但电费不能高出3.5%。"
    )
    assert intent.intent == "compare_schedules"
    assert intent.objective == ObjectiveMode.CARBON
    assert intent.max_cost_increase_pct == 3.5
    assert intent.cost_limit_explicit


def test_plain_low_carbon_request_has_no_explicit_cost_limit():
    intent = parse_mock_decision_intent("请运行明天的低碳调度")
    assert intent.objective == ObjectiveMode.CARBON
    assert not intent.cost_limit_explicit


def test_dual_objective_workflow_runs_three_verified_plans():
    inputs = build_inputs()
    outcome = run_dual_objective_workflow(
        run_id="dual-run",
        user_text="比较经济和低碳调度，优先减碳，但电费不能高出5%。",
        inputs=inputs,
        base_request=build_request(inputs),
    )
    assert outcome.status == WorkflowStatus.PENDING_HUMAN_APPROVAL
    assert tuple(outcome.tool_sequence) == DUAL_OBJECTIVE_TOOL_SEQUENCE
    assert len(outcome.plans) == 3
    assert all(plan.verification.passed for plan in outcome.plans)
    assert outcome.comparison is not None
    summaries = {plan.plan_id: plan for plan in outcome.comparison.plans}
    assert summaries["carbon"].grid_emissions_kgco2e < summaries["economic"].grid_emissions_kgco2e
    assert summaries["carbon_cost_capped"].total_cost_cny <= summaries["economic"].total_cost_cny * 1.05 + 0.01
    assert not outcome.executable
    assert outcome.explanation is not None
    assert outcome.explanation.provider == "mock"


def test_missing_carbon_factor_safe_terminates_before_optimization():
    inputs = build_inputs()
    points = list(inputs.points)
    points[3] = points[3].model_copy(
        update={"carbon_factor_kgco2e_per_kwh": None}
    )
    broken = inputs.model_copy(update={"points": points})
    outcome = run_dual_objective_workflow(
        run_id="missing-carbon",
        user_text="运行低碳调度",
        inputs=broken,
        base_request=build_request(broken),
    )
    assert outcome.status == WorkflowStatus.SAFE_TERMINATED
    assert outcome.failure_code == "CARBON_FACTOR_MISSING"
    assert "run_economic_dispatch" not in outcome.tool_sequence


def test_stale_inputs_safe_terminate():
    inputs = build_inputs().model_copy(
        update={"generated_at": START - timedelta(hours=48)}
    )
    outcome = run_dual_objective_workflow(
        run_id="stale-inputs",
        user_text="运行低碳调度",
        inputs=inputs,
        base_request=build_request(inputs),
    )
    assert outcome.status == WorkflowStatus.SAFE_TERMINATED
    assert outcome.failure_code == "INPUT_DATA_STALE"


def test_wrong_carbon_unit_safe_terminates():
    inputs = build_inputs().model_copy(update={"carbon_factor_unit": "gCO2/kWh"})
    outcome = run_dual_objective_workflow(
        run_id="bad-unit",
        user_text="运行低碳调度",
        inputs=inputs,
        base_request=build_request(inputs),
    )
    assert outcome.status == WorkflowStatus.SAFE_TERMINATED
    assert outcome.failure_code == "CARBON_FACTOR_UNIT_INVALID"
