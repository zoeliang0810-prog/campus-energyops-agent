"""Deterministic comparison of three assumed battery sizes."""

from __future__ import annotations

from .contracts import (
    BatteryScenarioComparison,
    BatteryScenarioResult,
    DispatchInputSeries,
    DispatchRequest,
)
from .optimization.cvxpy_pv_storage import CvxpyPVStorageOptimizer
from .verifier import verify_dispatch


BATTERY_SCENARIOS = {
    "small": (75.0, 200.0),
    "medium": (150.0, 400.0),
    "large": (300.0, 800.0),
}


def compare_battery_scenarios(
    inputs: DispatchInputSeries,
    base_request: DispatchRequest,
    *,
    solver_name: str | None = None,
) -> BatteryScenarioComparison:
    results: list[BatteryScenarioResult] = []
    for scenario_id, (power_kw, capacity_kwh) in BATTERY_SCENARIOS.items():
        battery = base_request.battery.model_copy(
            update={
                "asset_id": f"scenario_{scenario_id}_{power_kw:g}kw_{capacity_kwh:g}kwh",
                "capacity_kwh": capacity_kwh,
                "initial_energy_kwh": capacity_kwh * 0.5,
                "min_energy_kwh": capacity_kwh * 0.1,
                "max_energy_kwh": capacity_kwh * 0.9,
                "max_charge_kw": power_kw,
                "max_discharge_kw": power_kw,
            }
        )
        request = base_request.model_copy(
            update={
                "request_id": f"{base_request.request_id}_{scenario_id}",
                "battery": battery,
                "required_terminal_energy_kwh": capacity_kwh * 0.5,
            }
        )
        dispatch = CvxpyPVStorageOptimizer(
            inputs, solver_name=solver_name
        ).solve(request)
        verification = verify_dispatch(request, dispatch)
        metrics = verification.recomputed_metrics
        results.append(
            BatteryScenarioResult(
                scenario_id=scenario_id,
                power_kw=power_kw,
                capacity_kwh=capacity_kwh,
                solver_status=dispatch.status.value,
                verification_passed=verification.passed,
                peak_import_kw=metrics.peak_import_kw if metrics else None,
                peak_reduction_kw=metrics.peak_reduction_kw if metrics else None,
                pv_self_consumption_ratio=(
                    metrics.pv_self_consumption_ratio if metrics else None
                ),
                battery_throughput_kwh=(
                    metrics.battery_charge_energy_kwh
                    + metrics.battery_discharge_energy_kwh
                    if metrics
                    else None
                ),
                evidence_id=verification.evidence_id,
            )
        )
    return BatteryScenarioComparison(
        dataset_version=inputs.dataset_version,
        scenarios=results,
        simulation_only=True,
        executable=False,
    )
