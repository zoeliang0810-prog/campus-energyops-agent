"""Verified day-ahead PV and battery dispatch using CVXPY.

This module is the refactored replacement for the solver embedded in the legacy
MPC notebook. A later monitoring workflow can call the same operator again with
the latest battery state to provide receding-horizon rescheduling.
"""

from __future__ import annotations

import hashlib
import time

import numpy as np

from energyops.contracts import (
    DispatchInputSeries,
    DispatchMetrics,
    DispatchRequest,
    DispatchResult,
    DispatchStatus,
    ObjectiveMode,
    SchedulePoint,
    SolverMetadata,
)
from energyops.verifier import recompute_metrics


class CvxpyPVStorageOptimizer:
    def __init__(
        self,
        inputs: DispatchInputSeries,
        *,
        solver_name: str | None = None,
    ) -> None:
        self.inputs = inputs
        self.solver_name = solver_name or "auto"

    def solve(self, request: DispatchRequest) -> DispatchResult:
        started = time.perf_counter()
        if self.inputs.dataset_version != request.dataset_version:
            return self._non_optimal(
                request,
                DispatchStatus.ERROR,
                "DATASET_BINDING_MISMATCH",
                started,
            )
        if len(self.inputs.points) != request.horizon_steps:
            return self._non_optimal(
                request,
                DispatchStatus.ERROR,
                "HORIZON_LENGTH_MISMATCH",
                started,
            )

        try:
            import cvxpy as cp
        except ImportError:
            return self._non_optimal(
                request,
                DispatchStatus.ERROR,
                "CVXPY_NOT_INSTALLED",
                started,
            )

        installed_solvers = set(cp.installed_solvers())
        if self.solver_name == "auto":
            solver_name = next(
                (
                    candidate
                    for candidate in (
                        "COPT",
                        "HIGHS",
                        "SCIP",
                        "CBC",
                        "GLPK_MI",
                        "ECOS_BB",
                    )
                    if candidate in installed_solvers
                ),
                None,
            )
            if solver_name is None:
                return self._non_optimal(
                    request,
                    DispatchStatus.ERROR,
                    "MIP_SOLVER_NOT_INSTALLED",
                    started,
                )
        elif self.solver_name not in installed_solvers:
            return self._non_optimal(
                request,
                DispatchStatus.ERROR,
                "REQUESTED_SOLVER_NOT_INSTALLED",
                started,
            )
        else:
            solver_name = self.solver_name

        points = self.inputs.points
        expected_seconds = request.step_minutes * 60
        for index, point in enumerate(points):
            expected = request.horizon_start.timestamp() + index * expected_seconds
            if abs(point.timestamp.timestamp() - expected) > 1e-6:
                return self._non_optimal(
                    request,
                    DispatchStatus.ERROR,
                    "INPUT_TIMESTAMP_DISCONTINUITY",
                    started,
                )

        horizon = request.horizon_steps
        dt = request.step_minutes / 60.0
        battery = request.battery
        pv = np.array([point.pv_power_kw for point in points], dtype=float)
        load = np.array([point.load_power_kw for point in points], dtype=float)
        buy_price = np.array(
            [point.buy_price_cny_per_kwh for point in points], dtype=float
        )
        sell_price = np.array(
            [point.sell_price_cny_per_kwh for point in points], dtype=float
        )
        carbon_values = [
            point.carbon_factor_kgco2e_per_kwh for point in points
        ]
        carbon_required = request.objective_mode in {
            ObjectiveMode.CARBON,
            ObjectiveMode.BALANCED,
        }
        if carbon_required and any(value is None for value in carbon_values):
            return self._non_optimal(
                request,
                DispatchStatus.ERROR,
                "CARBON_FACTOR_REQUIRED",
                started,
            )
        carbon_factor = np.array(
            [0.0 if value is None else value for value in carbon_values],
            dtype=float,
        )

        charge = cp.Variable(horizon, nonneg=True)
        discharge = cp.Variable(horizon, nonneg=True)
        grid_import = cp.Variable(horizon, nonneg=True)
        grid_export = cp.Variable(horizon, nonneg=True)
        pv_curtailment = cp.Variable(horizon, nonneg=True)
        energy = cp.Variable(horizon + 1)
        peak_import = cp.Variable(nonneg=True)
        battery_mode = cp.Variable(horizon, boolean=True)
        grid_mode = cp.Variable(horizon, boolean=True)

        constraints = [energy[0] == battery.initial_energy_kwh]
        for slot in range(horizon):
            constraints.extend(
                [
                    energy[slot + 1]
                    == energy[slot]
                    + charge[slot] * battery.charge_efficiency * dt
                    - discharge[slot] / battery.discharge_efficiency * dt
                    - battery.self_discharge_kwh_per_hour * dt,
                    energy[slot + 1] >= battery.min_energy_kwh,
                    energy[slot + 1] <= battery.max_energy_kwh,
                    charge[slot] <= battery.max_charge_kw * battery_mode[slot],
                    discharge[slot]
                    <= battery.max_discharge_kw * (1 - battery_mode[slot]),
                    grid_import[slot]
                    <= request.grid_import_limit_kw * (1 - grid_mode[slot]),
                    grid_import[slot] <= peak_import,
                    grid_export[slot]
                    <= request.grid_export_limit_kw * grid_mode[slot],
                    pv_curtailment[slot] <= pv[slot],
                    grid_import[slot]
                    - grid_export[slot]
                    + pv[slot]
                    - pv_curtailment[slot]
                    + discharge[slot]
                    == load[slot] + charge[slot],
                ]
            )

        if not request.allow_grid_export:
            constraints.append(grid_export == 0)
        if not request.allow_grid_charging:
            # Conservative first-version rule: the battery may only charge from
            # instantaneous PV surplus. A provenance-flow model can replace this
            # rule when battery export policies are added.
            constraints.append(charge <= np.maximum(pv - load, 0))
        if request.required_terminal_energy_kwh is not None:
            constraints.append(energy[horizon] >= request.required_terminal_energy_kwh)

        electricity_cost = cp.sum(
            cp.multiply(grid_import, buy_price)
            - cp.multiply(grid_export, sell_price)
        ) * dt
        throughput_cost = cp.sum(charge + discharge) * (
            battery.degradation_cost_cny_per_kwh_throughput * dt
        )
        total_cost = electricity_cost + throughput_cost
        grid_emissions = cp.sum(cp.multiply(grid_import, carbon_factor)) * dt
        if request.max_total_cost_cny is not None:
            constraints.append(total_cost <= request.max_total_cost_cny + 1e-6)
        if request.objective_mode == ObjectiveMode.PEAK_AND_PV_SELF_CONSUMPTION:
            objective = (
                request.peak_penalty * peak_import
                + request.curtailment_penalty * cp.sum(pv_curtailment) * dt
                + request.throughput_penalty * cp.sum(charge + discharge) * dt
            )
        elif request.objective_mode == ObjectiveMode.CARBON:
            objective = (
                grid_emissions
                + request.throughput_penalty * cp.sum(charge + discharge) * dt
            )
        elif request.objective_mode == ObjectiveMode.BALANCED:
            objective = total_cost + request.carbon_price_cny_per_kg * grid_emissions
        else:
            objective = total_cost
        problem = cp.Problem(cp.Minimize(objective), constraints)

        try:
            problem.solve(solver=solver_name, verbose=False)
        except Exception:
            return self._non_optimal(
                request,
                DispatchStatus.ERROR,
                "SOLVER_EXCEPTION",
                started,
            )

        elapsed_ms = (time.perf_counter() - started) * 1000
        if problem.status in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE}:
            return self._non_optimal(
                request,
                DispatchStatus.INFEASIBLE,
                "OPTIMIZATION_INFEASIBLE",
                started,
                solver_status=problem.status,
            )
        if problem.status != cp.OPTIMAL:
            return self._non_optimal(
                request,
                DispatchStatus.ERROR,
                "OPTIMIZATION_NOT_OPTIMAL",
                started,
                solver_status=problem.status,
            )

        def clean(value: float) -> float:
            value = float(value)
            return 0.0 if abs(value) < 1e-8 else value

        schedule_points = [
            SchedulePoint(
                timestamp=source.timestamp,
                pv_power_kw=source.pv_power_kw,
                pv_curtailment_kw=clean(pv_curtailment.value[index]),
                load_power_kw=source.load_power_kw,
                battery_charge_kw=clean(charge.value[index]),
                battery_discharge_kw=clean(discharge.value[index]),
                grid_import_kw=clean(grid_import.value[index]),
                grid_export_kw=clean(grid_export.value[index]),
                battery_energy_before_kwh=clean(energy.value[index]),
                battery_energy_after_kwh=clean(energy.value[index + 1]),
                buy_price_cny_per_kwh=source.buy_price_cny_per_kwh,
                sell_price_cny_per_kwh=source.sell_price_cny_per_kwh,
                carbon_factor_kgco2e_per_kwh=(
                    source.carbon_factor_kgco2e_per_kwh
                ),
            )
            for index, source in enumerate(points)
        ]
        evidence_id = self._evidence_id(request, problem.value)
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
            schedule_id=f"schedule_{evidence_id[-12:]}",
            request_id=request.request_id,
            dataset_version=request.dataset_version,
            status=DispatchStatus.OPTIMAL,
            points=schedule_points,
            metrics=placeholder,
            solver=SolverMetadata(
                solver_name=solver_name,
                solver_status=problem.status,
                solve_time_ms=elapsed_ms,
                objective_value=float(problem.value),
            ),
            evidence_id=evidence_id,
        )
        result.metrics = recompute_metrics(request, result)
        return result

    def _non_optimal(
        self,
        request: DispatchRequest,
        status: DispatchStatus,
        reason_code: str,
        started: float,
        *,
        solver_status: str | None = None,
    ) -> DispatchResult:
        elapsed_ms = (time.perf_counter() - started) * 1000
        evidence_id = self._evidence_id(request, reason_code)
        return DispatchResult(
            schedule_id=f"schedule_{evidence_id[-12:]}",
            request_id=request.request_id,
            dataset_version=request.dataset_version,
            status=status,
            solver=SolverMetadata(
                solver_name=self.solver_name,
                solver_status=solver_status or reason_code,
                solve_time_ms=elapsed_ms,
            ),
            reason_codes=[reason_code],
            evidence_id=evidence_id,
        )

    @staticmethod
    def _evidence_id(request: DispatchRequest, value: object) -> str:
        digest = hashlib.sha256(
            f"{request.request_id}:{request.dataset_version}:{value}".encode()
        ).hexdigest()[:16]
        return f"optimizer_{digest}"
