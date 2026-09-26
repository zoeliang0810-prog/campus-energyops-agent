"""Safe stage-one economic/carbon decision workflow with synthetic inputs."""

from __future__ import annotations

import re
from datetime import timedelta

from .contracts import (
    DispatchInputSeries,
    DispatchPlanComparison,
    DispatchPlanSummary,
    DispatchRequest,
    DualObjectiveWorkflowResult,
    EnergyDecisionIntent,
    GroundedExplanation,
    ObjectiveMode,
    VerifiedDispatchPlan,
    WorkflowStatus,
)
from .optimization.cvxpy_pv_storage import CvxpyPVStorageOptimizer
from .verifier import verify_dispatch


DUAL_OBJECTIVE_TOOL_SEQUENCE = (
    "get_system_status",
    "get_latest_forecast",
    "get_tariff",
    "get_carbon_factors",
    "run_economic_dispatch",
    "run_carbon_dispatch",
    "verify_dispatch",
    "compare_dispatch_plans",
    "explain_dispatch",
    "submit_for_approval",
)


def parse_mock_decision_intent(user_text: str) -> EnergyDecisionIntent:
    """Deterministically emulate the Agent intent boundary for offline tests."""
    objective = ObjectiveMode.BALANCED
    if any(token in user_text for token in ("省钱", "经济", "电费最低")):
        objective = ObjectiveMode.ECONOMIC
    if any(token in user_text for token in ("低碳", "减碳", "碳排")):
        objective = ObjectiveMode.CARBON
    match = re.search(r"(?:增加|高出|上涨|超过)\s*(\d+(?:\.\d+)?)\s*%", user_text)
    max_increase = float(match.group(1)) if match else 5.0
    return EnergyDecisionIntent(
        intent="compare_schedules" if "比较" in user_text else "create_schedule",
        objective=objective,
        max_cost_increase_pct=max_increase,
        cost_limit_explicit=match is not None,
    )


def _validate_inputs(inputs: DispatchInputSeries) -> None:
    if len(inputs.points) != 24:
        raise ValueError("INPUT_HORIZON_INVALID")
    if inputs.source not in {
        "synthetic_fixture",
        "stage_b_physical_pv_scenario",
    } or inputs.verified:
        raise ValueError("SIMULATION_PROVENANCE_INVALID")
    if not inputs.simulation_only:
        raise ValueError("SIMULATION_POLICY_INVALID")
    if inputs.power_unit != "kW":
        raise ValueError("FORECAST_UNIT_INVALID")
    if inputs.carbon_factor_unit != "kgCO2e/kWh":
        raise ValueError("CARBON_FACTOR_UNIT_INVALID")
    if inputs.generated_at is None:
        raise ValueError("INPUT_GENERATED_AT_MISSING")
    age = inputs.points[0].timestamp - inputs.generated_at
    if age < timedelta(0) or age > timedelta(hours=24):
        raise ValueError("INPUT_DATA_STALE")
    if not inputs.carbon_factor_version:
        raise ValueError("CARBON_FACTOR_VERSION_MISSING")
    for previous, current in zip(inputs.points, inputs.points[1:]):
        if current.timestamp - previous.timestamp != timedelta(hours=1):
            raise ValueError("INPUT_TIMESTAMP_DISCONTINUITY")
    if any(
        point.carbon_factor_kgco2e_per_kwh is None for point in inputs.points
    ):
        raise ValueError("CARBON_FACTOR_MISSING")


def _solve_verified(
    inputs: DispatchInputSeries,
    request: DispatchRequest,
    *,
    solver_name: str | None,
) -> VerifiedDispatchPlan:
    dispatch = CvxpyPVStorageOptimizer(inputs, solver_name=solver_name).solve(request)
    verification = verify_dispatch(request, dispatch)
    if not verification.passed:
        reason = dispatch.reason_codes[0] if dispatch.reason_codes else "VERIFICATION_FAILED"
        raise ValueError(reason)
    return VerifiedDispatchPlan(
        request=request,
        dispatch=dispatch,
        verification=verification,
    )


def _summary(
    plan_id: str,
    plan: VerifiedDispatchPlan,
    economic: VerifiedDispatchPlan,
) -> DispatchPlanSummary:
    metrics = plan.verification.recomputed_metrics
    baseline = economic.verification.recomputed_metrics
    if metrics is None or baseline is None:
        raise ValueError("VERIFIED_METRICS_MISSING")
    cost_change = (
        (metrics.total_cost_cny / baseline.total_cost_cny - 1) * 100
        if baseline.total_cost_cny
        else 0.0
    )
    emissions_change = (
        (metrics.grid_emissions_kgco2e / baseline.grid_emissions_kgco2e - 1) * 100
        if baseline.grid_emissions_kgco2e
        else 0.0
    )
    return DispatchPlanSummary(
        plan_id=plan_id,
        objective=plan.request.objective_mode,
        verification_passed=plan.verification.passed,
        total_cost_cny=metrics.total_cost_cny,
        grid_emissions_kgco2e=metrics.grid_emissions_kgco2e,
        grid_import_energy_kwh=metrics.grid_import_energy_kwh,
        peak_import_kw=metrics.peak_import_kw,
        pv_self_consumption_ratio=metrics.pv_self_consumption_ratio,
        pv_curtailed_energy_kwh=metrics.pv_curtailed_energy_kwh,
        battery_throughput_kwh=metrics.battery_throughput_kwh,
        cost_change_vs_economic_pct=cost_change,
        emissions_change_vs_economic_pct=emissions_change,
    )


def run_dual_objective_workflow(
    *,
    run_id: str,
    user_text: str,
    inputs: DispatchInputSeries,
    base_request: DispatchRequest,
    solver_name: str | None = None,
) -> DualObjectiveWorkflowResult:
    """Run economic, carbon, and cost-capped carbon plans behind hard gates."""
    intent = parse_mock_decision_intent(user_text)
    used_tools: list[str] = []
    plans: list[VerifiedDispatchPlan] = []
    try:
        for tool in DUAL_OBJECTIVE_TOOL_SEQUENCE[:4]:
            used_tools.append(tool)
        _validate_inputs(inputs)

        economic_request = base_request.model_copy(
            update={
                "request_id": f"{base_request.request_id}_economic",
                "objective_mode": ObjectiveMode.ECONOMIC,
                "carbon_factor_version": inputs.carbon_factor_version,
                "max_total_cost_cny": None,
                "max_cost_increase_pct": None,
            }
        )
        used_tools.append("run_economic_dispatch")
        economic = _solve_verified(
            inputs, economic_request, solver_name=solver_name
        )
        plans.append(economic)

        carbon_request = base_request.model_copy(
            update={
                "request_id": f"{base_request.request_id}_carbon",
                "objective_mode": ObjectiveMode.CARBON,
                "carbon_factor_version": inputs.carbon_factor_version,
                "max_total_cost_cny": None,
                "max_cost_increase_pct": None,
            }
        )
        used_tools.append("run_carbon_dispatch")
        carbon = _solve_verified(inputs, carbon_request, solver_name=solver_name)
        plans.append(carbon)

        economic_metrics = economic.verification.recomputed_metrics
        if economic_metrics is None:
            raise ValueError("ECONOMIC_BASELINE_METRICS_MISSING")
        cost_cap = economic_metrics.total_cost_cny * (
            1 + intent.max_cost_increase_pct / 100
        )
        capped_request = carbon_request.model_copy(
            update={
                "request_id": f"{base_request.request_id}_carbon_capped",
                "max_total_cost_cny": cost_cap,
                "max_cost_increase_pct": intent.max_cost_increase_pct,
            }
        )
        capped = _solve_verified(inputs, capped_request, solver_name=solver_name)
        plans.append(capped)

        used_tools.append("verify_dispatch")
        summaries = [
            _summary("economic", economic, economic),
            _summary("carbon", carbon, economic),
            _summary("carbon_cost_capped", capped, economic),
        ]
        if intent.objective == ObjectiveMode.ECONOMIC:
            recommended = "economic"
            basis = "用户要求优先降低电费。"
        elif intent.objective == ObjectiveMode.CARBON and not intent.cost_limit_explicit:
            recommended = "carbon"
            basis = "用户要求纯低碳目标，未设置成本增幅约束。"
        else:
            recommended = "carbon_cost_capped"
            basis = (
                f"在相对经济方案成本增幅不超过{intent.max_cost_increase_pct:g}%的条件下优先减碳。"
            )
        comparison = DispatchPlanComparison(
            dataset_version=inputs.dataset_version,
            plans=summaries,
            recommended_plan_id=recommended,
            recommendation_basis=basis,
        )
        used_tools.append("compare_dispatch_plans")
        selected = next(item for item in summaries if item.plan_id == recommended)
        source_warning = (
            "负荷/PV预测和逐时碳因子包含synthetic模拟输入。"
            if inputs.source == "synthetic_fixture"
            else "负荷为研究日完整观测面板，光伏为规划物理情景，逐时碳因子为synthetic模拟输入。"
        )
        explanation = GroundedExplanation(
            summary=(
                f"推荐 {recommended} 仿真方案；相对经济方案成本变化 "
                f"{selected.cost_change_vs_economic_pct:.2f}%，碳排变化 "
                f"{selected.emissions_change_vs_economic_pct:.2f}%。"
            ),
            warnings=[
                source_warning,
                "结果不代表校园真实节费或减排，不能作为设备控制指令。",
            ],
            evidence_claim_ids=[
                f"comparison:{recommended}:total_cost_cny",
                f"comparison:{recommended}:grid_emissions_kgco2e",
                "verification:all_plans_passed",
            ],
            provider="mock",
        )
        used_tools.extend(["explain_dispatch", "submit_for_approval"])
        return DualObjectiveWorkflowResult(
            run_id=run_id,
            status=WorkflowStatus.PENDING_HUMAN_APPROVAL,
            intent=intent,
            tool_sequence=used_tools,
            plans=plans,
            comparison=comparison,
            explanation=explanation,
        )
    except Exception as error:
        code = str(error) if str(error).isupper() else type(error).__name__
        return DualObjectiveWorkflowResult(
            run_id=run_id,
            status=WorkflowStatus.SAFE_TERMINATED,
            intent=intent,
            tool_sequence=used_tools,
            plans=plans,
            failure_code=code,
        )
