"""Independent constraint, accounting, and evidence verifier."""

from __future__ import annotations

import hashlib
from datetime import timedelta

from .contracts import (
    ConstraintViolation,
    DispatchMetrics,
    DispatchRequest,
    DispatchResult,
    DispatchStatus,
    ObjectiveMode,
    VerificationReport,
)


POWER_TOLERANCE_KW = 1e-5
ENERGY_TOLERANCE_KWH = 1e-5
MONEY_TOLERANCE_CNY = 0.01


def _violation(
    constraint_id: str,
    message: str,
    *,
    timestamp=None,
    actual=None,
    expected=None,
    unit=None,
) -> ConstraintViolation:
    return ConstraintViolation(
        constraint_id=constraint_id,
        message=message,
        timestamp=timestamp,
        actual=actual,
        expected=expected,
        unit=unit,
    )


def recompute_metrics(request: DispatchRequest, result: DispatchResult) -> DispatchMetrics:
    dt = request.step_minutes / 60.0
    points = result.points
    pv = sum(point.pv_power_kw * dt for point in points)
    pv_curtailed = sum(point.pv_curtailment_kw * dt for point in points)
    load = sum(point.load_power_kw * dt for point in points)
    charge = sum(point.battery_charge_kw * dt for point in points)
    discharge = sum(point.battery_discharge_kw * dt for point in points)
    imported = sum(point.grid_import_kw * dt for point in points)
    exported = sum(point.grid_export_kw * dt for point in points)
    electricity_cost = sum(
        (
            point.grid_import_kw * point.buy_price_cny_per_kwh
            - point.grid_export_kw * point.sell_price_cny_per_kwh
        )
        * dt
        for point in points
    )
    degradation_cost = (
        charge + discharge
    ) * request.battery.degradation_cost_cny_per_kwh_throughput
    baseline_peak = max(
        (max(point.load_power_kw - point.pv_power_kw, 0.0) for point in points),
        default=0.0,
    )
    peak_import = max((point.grid_import_kw for point in points), default=0.0)
    pv_self_consumed = max(pv - pv_curtailed - exported, 0.0)
    grid_emissions = sum(
        point.grid_import_kw
        * (point.carbon_factor_kgco2e_per_kwh or 0.0)
        * dt
        for point in points
    )
    return DispatchMetrics(
        pv_energy_kwh=pv,
        pv_curtailed_energy_kwh=pv_curtailed,
        load_energy_kwh=load,
        battery_charge_energy_kwh=charge,
        battery_discharge_energy_kwh=discharge,
        grid_import_energy_kwh=imported,
        grid_export_energy_kwh=exported,
        electricity_cost_cny=electricity_cost,
        degradation_cost_cny=degradation_cost,
        total_cost_cny=electricity_cost + degradation_cost,
        peak_import_kw=peak_import,
        baseline_peak_import_kw=baseline_peak,
        peak_reduction_kw=baseline_peak - peak_import,
        pv_self_consumption_ratio=(pv_self_consumed / pv if pv else 0.0),
        grid_emissions_kgco2e=grid_emissions,
        battery_throughput_kwh=charge + discharge,
    )


def verify_dispatch(
    request: DispatchRequest,
    result: DispatchResult,
) -> VerificationReport:
    violations: list[ConstraintViolation] = []

    if result.request_id != request.request_id:
        violations.append(_violation("REQUEST_BINDING", "结果未绑定当前请求。"))
    if result.dataset_version != request.dataset_version:
        violations.append(_violation("DATASET_BINDING", "结果使用了不同的数据版本。"))
    if result.status != DispatchStatus.OPTIMAL:
        violations.append(
            _violation("SOLVER_STATUS", "只有 optimal 结果可以进入约束验证。")
        )
        return _report(result, violations, None, request)
    if len(result.points) != request.horizon_steps:
        violations.append(
            _violation(
                "HORIZON_LENGTH",
                "调度点数量与请求时域不一致。",
                actual=len(result.points),
                expected=request.horizon_steps,
                unit="slot",
            )
        )

    dt = request.step_minutes / 60.0
    expected_step = timedelta(minutes=request.step_minutes)
    battery = request.battery

    for index, point in enumerate(result.points):
        if (
            request.objective_mode in {ObjectiveMode.CARBON, ObjectiveMode.BALANCED}
            and point.carbon_factor_kgco2e_per_kwh is None
        ):
            violations.append(
                _violation(
                    "CARBON_FACTOR_REQUIRED",
                    "低碳调度时间片缺少动态碳因子。",
                    timestamp=point.timestamp,
                )
            )
        if index == 0:
            if abs(point.battery_energy_before_kwh - battery.initial_energy_kwh) > ENERGY_TOLERANCE_KWH:
                violations.append(
                    _violation(
                        "INITIAL_ENERGY",
                        "首个时间片的电池初始能量不一致。",
                        timestamp=point.timestamp,
                        actual=point.battery_energy_before_kwh,
                        expected=battery.initial_energy_kwh,
                        unit="kWh",
                    )
                )
        else:
            previous = result.points[index - 1]
            if point.timestamp - previous.timestamp != expected_step:
                violations.append(
                    _violation(
                        "TIMESTAMP_CONTINUITY",
                        "调度时间戳不连续。",
                        timestamp=point.timestamp,
                        actual=str(point.timestamp - previous.timestamp),
                        expected=str(expected_step),
                    )
                )
            if abs(point.battery_energy_before_kwh - previous.battery_energy_after_kwh) > ENERGY_TOLERANCE_KWH:
                violations.append(
                    _violation(
                        "ENERGY_STATE_CONTINUITY",
                        "相邻时间片的电池能量状态不连续。",
                        timestamp=point.timestamp,
                        actual=point.battery_energy_before_kwh,
                        expected=previous.battery_energy_after_kwh,
                        unit="kWh",
                    )
                )

        balance = (
            point.grid_import_kw
            - point.grid_export_kw
            + point.pv_power_kw
            - point.pv_curtailment_kw
            + point.battery_discharge_kw
            - point.load_power_kw
            - point.battery_charge_kw
        )
        if abs(balance) > POWER_TOLERANCE_KW:
            violations.append(
                _violation(
                    "POWER_BALANCE",
                    "电力平衡不成立。",
                    timestamp=point.timestamp,
                    actual=balance,
                    expected=0,
                    unit="kW",
                )
            )

        if point.pv_curtailment_kw > point.pv_power_kw + POWER_TOLERANCE_KW:
            violations.append(
                _violation(
                    "PV_CURTAILMENT_LIMIT",
                    "弃光功率超过该时间片可用光伏功率。",
                    timestamp=point.timestamp,
                    actual=point.pv_curtailment_kw,
                    expected=point.pv_power_kw,
                    unit="kW",
                )
            )

        expected_energy = (
            point.battery_energy_before_kwh
            + point.battery_charge_kw * battery.charge_efficiency * dt
            - point.battery_discharge_kw / battery.discharge_efficiency * dt
            - battery.self_discharge_kwh_per_hour * dt
        )
        if abs(expected_energy - point.battery_energy_after_kwh) > ENERGY_TOLERANCE_KWH:
            violations.append(
                _violation(
                    "BATTERY_ENERGY_TRANSITION",
                    "电池能量状态变化与充放电功率不一致。",
                    timestamp=point.timestamp,
                    actual=point.battery_energy_after_kwh,
                    expected=expected_energy,
                    unit="kWh",
                )
            )
        if not battery.min_energy_kwh - ENERGY_TOLERANCE_KWH <= point.battery_energy_after_kwh <= battery.max_energy_kwh + ENERGY_TOLERANCE_KWH:
            violations.append(
                _violation(
                    "BATTERY_ENERGY_BOUNDS",
                    "电池能量超出上下限。",
                    timestamp=point.timestamp,
                    actual=point.battery_energy_after_kwh,
                    expected=f"[{battery.min_energy_kwh}, {battery.max_energy_kwh}]",
                    unit="kWh",
                )
            )
        if point.battery_charge_kw > battery.max_charge_kw + POWER_TOLERANCE_KW:
            violations.append(_violation("CHARGE_POWER_LIMIT", "充电功率超过上限。", timestamp=point.timestamp))
        if point.battery_discharge_kw > battery.max_discharge_kw + POWER_TOLERANCE_KW:
            violations.append(_violation("DISCHARGE_POWER_LIMIT", "放电功率超过上限。", timestamp=point.timestamp))
        if point.battery_charge_kw > POWER_TOLERANCE_KW and point.battery_discharge_kw > POWER_TOLERANCE_KW:
            violations.append(_violation("BATTERY_EXCLUSIVITY", "同一时间片同时充电和放电。", timestamp=point.timestamp))
        if point.grid_import_kw > request.grid_import_limit_kw + POWER_TOLERANCE_KW:
            violations.append(_violation("GRID_IMPORT_LIMIT", "电网购电功率超过上限。", timestamp=point.timestamp))
        if point.grid_export_kw > request.grid_export_limit_kw + POWER_TOLERANCE_KW:
            violations.append(_violation("GRID_EXPORT_LIMIT", "上网功率超过上限。", timestamp=point.timestamp))
        if point.grid_import_kw > POWER_TOLERANCE_KW and point.grid_export_kw > POWER_TOLERANCE_KW:
            violations.append(_violation("GRID_EXCLUSIVITY", "同一时间片同时购电和上网。", timestamp=point.timestamp))
        if not request.allow_grid_export and point.grid_export_kw > POWER_TOLERANCE_KW:
            violations.append(_violation("EXPORT_POLICY", "请求未允许上网，但结果包含上网功率。", timestamp=point.timestamp))
        if not request.allow_grid_charging:
            allowed_pv_surplus = max(point.pv_power_kw - point.load_power_kw, 0.0)
            if point.battery_charge_kw > allowed_pv_surplus + POWER_TOLERANCE_KW:
                violations.append(
                    _violation(
                        "GRID_CHARGING_POLICY",
                        "请求禁止电网充电，但充电功率超过当时光伏余量。",
                        timestamp=point.timestamp,
                        actual=point.battery_charge_kw,
                        expected=allowed_pv_surplus,
                        unit="kW",
                    )
                )

    if result.points and request.required_terminal_energy_kwh is not None:
        terminal = result.points[-1].battery_energy_after_kwh
        if terminal + ENERGY_TOLERANCE_KWH < request.required_terminal_energy_kwh:
            violations.append(
                _violation(
                    "TERMINAL_ENERGY",
                    "终端电池能量低于要求。",
                    timestamp=result.points[-1].timestamp,
                    actual=terminal,
                    expected=request.required_terminal_energy_kwh,
                    unit="kWh",
                )
            )

    recomputed = recompute_metrics(request, result)
    if (
        request.max_total_cost_cny is not None
        and recomputed.total_cost_cny
        > request.max_total_cost_cny + MONEY_TOLERANCE_CNY
    ):
        violations.append(
            _violation(
                "TOTAL_COST_LIMIT",
                "低碳方案超过允许的总成本上限。",
                actual=recomputed.total_cost_cny,
                expected=request.max_total_cost_cny,
                unit="CNY",
            )
        )
    if result.metrics is not None:
        for field_name in DispatchMetrics.model_fields:
            reported = getattr(result.metrics, field_name)
            expected = getattr(recomputed, field_name)
            tolerance = MONEY_TOLERANCE_CNY if field_name.endswith("_cny") else ENERGY_TOLERANCE_KWH
            if field_name == "peak_import_kw":
                tolerance = POWER_TOLERANCE_KW
            if abs(reported - expected) > tolerance:
                violations.append(
                    _violation(
                        "METRIC_RECONCILIATION",
                        f"汇总指标 {field_name} 与时间片复算结果不一致。",
                        actual=reported,
                        expected=expected,
                    )
                )

    return _report(result, violations, recomputed, request)


def _report(result, violations, metrics, request) -> VerificationReport:
    digest = hashlib.sha256(
        f"{result.schedule_id}:{len(violations)}:{request.dataset_version}".encode()
    ).hexdigest()[:12]
    passed = not violations
    executable = passed and request.operating_mode.value == "decision_support"
    return VerificationReport(
        schedule_id=result.schedule_id,
        passed=passed,
        executable=executable,
        violations=violations,
        recomputed_metrics=metrics,
        evidence_id=f"verify_{digest}",
    )
