from __future__ import annotations

import unittest
from copy import deepcopy
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from energyops.contracts import (
    BatteryConfig,
    DispatchMetrics,
    DispatchRequest,
    DispatchResult,
    DispatchStatus,
    OperatingMode,
    SchedulePoint,
    SolverMetadata,
)
from energyops.verifier import recompute_metrics, verify_dispatch


def build_request() -> DispatchRequest:
    return DispatchRequest(
        request_id="req_test",
        site_id="site_test",
        dataset_version="dataset_test_v1",
        operating_mode=OperatingMode.SIMULATION_ONLY,
        horizon_start=datetime(2026, 2, 19, tzinfo=ZoneInfo("Asia/Shanghai")),
        horizon_steps=2,
        step_minutes=15,
        grid_import_limit_kw=10,
        grid_export_limit_kw=0,
        tariff_region="demo",
        tariff_version="demo_v1",
        battery=BatteryConfig(
            asset_id="battery_test",
            capacity_kwh=5.12,
            initial_energy_kwh=2.5,
            min_energy_kwh=0.512,
            max_energy_kwh=5.12,
            max_charge_kw=2.5,
            max_discharge_kw=2.5,
            charge_efficiency=0.95,
            discharge_efficiency=0.95,
        ),
        required_terminal_energy_kwh=2.5,
        assumptions=["synthetic unit-test data"],
    )


def build_result(request: DispatchRequest) -> DispatchResult:
    points = []
    for index in range(2):
        points.append(
            SchedulePoint(
                timestamp=request.horizon_start + timedelta(minutes=15 * index),
                pv_power_kw=0,
                load_power_kw=1,
                battery_charge_kw=0,
                battery_discharge_kw=0,
                grid_import_kw=1,
                grid_export_kw=0,
                battery_energy_before_kwh=2.5,
                battery_energy_after_kwh=2.5,
                buy_price_cny_per_kwh=0.5,
                sell_price_cny_per_kwh=0,
            )
        )
    placeholder = DispatchMetrics(
        pv_energy_kwh=0,
        pv_curtailed_energy_kwh=0,
        load_energy_kwh=0,
        battery_charge_energy_kwh=0,
        battery_discharge_energy_kwh=0,
        grid_import_energy_kwh=0,
        grid_export_energy_kwh=0,
        electricity_cost_cny=0,
        degradation_cost_cny=0,
        total_cost_cny=0,
        peak_import_kw=0,
    )
    result = DispatchResult(
        schedule_id="schedule_test",
        request_id=request.request_id,
        dataset_version=request.dataset_version,
        status=DispatchStatus.OPTIMAL,
        points=points,
        metrics=placeholder,
        solver=SolverMetadata(
            solver_name="mock",
            solver_status="optimal",
            solve_time_ms=1,
        ),
        evidence_id="optimizer_test",
    )
    result.metrics = recompute_metrics(request, result)
    return result


class VerifierTests(unittest.TestCase):
    def test_valid_schedule_passes_but_simulation_is_not_executable(self):
        request = build_request()
        result = build_result(request)
        report = verify_dispatch(request, result)
        self.assertTrue(report.passed)
        self.assertFalse(report.executable)

    def test_wrong_cost_is_rejected(self):
        request = build_request()
        result = build_result(request)
        result.metrics.total_cost_cny += 1
        report = verify_dispatch(request, result)
        self.assertFalse(report.passed)
        self.assertIn("METRIC_RECONCILIATION", {item.constraint_id for item in report.violations})

    def test_power_balance_tampering_is_rejected(self):
        request = build_request()
        result = build_result(request)
        tampered = deepcopy(result)
        tampered.points[0].grid_import_kw = 2
        report = verify_dispatch(request, tampered)
        self.assertFalse(report.passed)
        self.assertIn("POWER_BALANCE", {item.constraint_id for item in report.violations})

    def test_grid_charging_policy_is_independently_checked(self):
        request = build_request()
        result = build_result(request)
        result.points[0].battery_charge_kw = 1.0
        result.points[0].grid_import_kw = 2.0
        result.points[0].battery_energy_after_kwh = 2.7375
        result.points[1].battery_energy_before_kwh = 2.7375
        result.points[1].battery_discharge_kw = 0.9025
        result.points[1].grid_import_kw = 0.0975
        result.points[1].battery_energy_after_kwh = 2.5
        result.metrics = recompute_metrics(request, result)
        report = verify_dispatch(request, result)
        self.assertFalse(report.passed)
        self.assertIn(
            "GRID_CHARGING_POLICY",
            {item.constraint_id for item in report.violations},
        )


if __name__ == "__main__":
    unittest.main()
