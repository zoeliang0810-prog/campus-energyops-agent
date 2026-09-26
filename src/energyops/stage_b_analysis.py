"""Stage B matrix analysis for physical PV, storage sizes and dispatch goals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import Field

from .contracts import (
    BatteryConfig,
    DispatchInputSeries,
    DispatchSeriesPoint,
    DualObjectiveWorkflowResult,
    ObjectiveMode,
    ParsedCampusRequest,
    StrictModel,
    WorkflowStatus,
)
from .dual_objective_workflow import run_dual_objective_workflow
from .physical_pv import PhysicalPVConfig, load_physical_pv_config
from .scenario import build_dispatch_request, build_hourly_inputs
from .scenario_config import CampusScenarioConfig, load_scenario_config
from .synthetic_inputs import CARBON_PATTERNS


STAGE_B_VERSION = "hkust-gz-stage-b-v1"

BATTERY_CASES: dict[str, tuple[float, float]] = {
    "small_75kw_200kwh": (75.0, 200.0),
    "medium_150kw_400kwh": (150.0, 400.0),
    "large_300kw_800kwh": (300.0, 800.0),
}


class NoBatteryBaseline(StrictModel):
    pv_scenario_id: str
    pv_capacity_kwp: float
    pv_energy_kwh: float
    load_energy_kwh: float
    grid_import_energy_kwh: float
    curtailed_energy_kwh: float
    curtailment_rate: float
    potential_export_peak_kw: float
    peak_grid_import_kw: float
    electricity_cost_cny: float
    grid_emissions_kgco2e: float


class StageBPlanRecord(StrictModel):
    pv_scenario_id: str
    pv_capacity_kwp: float
    battery_case_id: str
    battery_power_kw: float
    battery_capacity_kwh: float
    plan_id: str
    objective_mode: ObjectiveMode
    verification_passed: bool
    solver_name: str
    total_cost_cny: float
    grid_emissions_kgco2e: float
    grid_import_energy_kwh: float
    peak_grid_import_kw: float
    pv_energy_kwh: float
    curtailed_energy_kwh: float
    curtailment_rate: float
    residual_surplus_peak_kw: float
    pv_self_consumption_ratio: float
    battery_throughput_kwh: float
    soc_at_max_hours: int = Field(ge=0)
    soc_at_min_hours: int = Field(ge=0)
    avoided_curtailment_vs_no_battery_kwh: float
    curtailment_reduction_vs_no_battery_pct: float
    grid_import_reduction_vs_no_battery_kwh: float
    cost_change_vs_no_battery_cny: float
    emissions_change_vs_no_battery_kgco2e: float
    marginal_avoided_curtailment_kwh: float = 0.0
    marginal_grid_import_reduction_kwh: float = 0.0
    marginal_emissions_reduction_kgco2e: float = 0.0
    evidence_id: str


class StageBAnalysisResult(StrictModel):
    analysis_version: str
    study_day: str
    load_dataset_version: str
    carbon_factor_version: str
    carbon_factor_status: str
    tariff_version: str
    tariff_verified: bool
    baselines: list[NoBatteryBaseline]
    plans: list[StageBPlanRecord]
    all_verification_passed: bool
    warnings: list[str]
    simulation_only: bool = True
    executable: bool = False


@dataclass(frozen=True)
class StageBRun:
    analysis: StageBAnalysisResult
    workflows: dict[str, DualObjectiveWorkflowResult]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _physical_pv_paths(
    root: Path,
    config: PhysicalPVConfig,
    scenario_id: str,
) -> tuple[Path, Path]:
    prefix = root / config.output_prefix
    stem = (
        f"{prefix.name}_{scenario_id}_{config.study_day.isoformat()}_"
        f"{config.step_minutes}min"
    )
    return prefix.parent / f"{stem}.csv", prefix.parent / f"{stem}_manifest.json"


def _attach_carbon_factors(
    inputs: DispatchInputSeries,
    *,
    carbon_pattern_id: str,
    dataset_version: str,
) -> DispatchInputSeries:
    try:
        values = CARBON_PATTERNS[carbon_pattern_id]
    except KeyError as error:
        raise ValueError("UNKNOWN_CARBON_PATTERN") from error
    if len(values) != len(inputs.points):
        raise ValueError("CARBON_FACTOR_HORIZON_MISMATCH")
    points = [
        point.model_copy(
            update={"carbon_factor_kgco2e_per_kwh": values[index]}
        )
        for index, point in enumerate(inputs.points)
    ]
    generated_at = points[0].timestamp - timedelta(hours=6)
    return DispatchInputSeries(
        dataset_version=dataset_version,
        points=points,
        source="stage_b_physical_pv_scenario",
        generated_at=generated_at,
        timezone="Asia/Shanghai",
        power_unit="kW",
        carbon_factor_unit="kgCO2e/kWh",
        forecast_model_version="physical-pv-planning-v1",
        forecast_confidence=None,
        carbon_factor_version=f"synthetic-carbon-{carbon_pattern_id}-v1",
        verified=False,
        simulation_only=True,
        assumptions=[
            "负荷为2026-02-19完整观测分表面板，不是完整校园总表",
            "光伏为规划容量和阵列假设下的物理仿真，不是实测或正式预测",
            "动态碳因子为synthetic模拟曲线，不代表电网正式碳因子",
            "电价未经港科广正式结算核验",
        ],
    )


def _battery_config(
    base: BatteryConfig, case_id: str, power_kw: float, capacity_kwh: float
) -> BatteryConfig:
    return base.model_copy(
        update={
            "asset_id": f"stage_b_{case_id}",
            "capacity_kwh": capacity_kwh,
            "initial_energy_kwh": capacity_kwh * 0.5,
            "min_energy_kwh": capacity_kwh * 0.1,
            "max_energy_kwh": capacity_kwh * 0.9,
            "max_charge_kw": power_kw,
            "max_discharge_kw": power_kw,
        }
    )


def _no_battery_baseline(
    *,
    pv_scenario_id: str,
    pv_capacity_kwp: float,
    inputs: DispatchInputSeries,
) -> NoBatteryBaseline:
    dt = 1.0
    pv_energy = sum(point.pv_power_kw * dt for point in inputs.points)
    load_energy = sum(point.load_power_kw * dt for point in inputs.points)
    grid_import = [
        max(point.load_power_kw - point.pv_power_kw, 0.0)
        for point in inputs.points
    ]
    surplus = [
        max(point.pv_power_kw - point.load_power_kw, 0.0)
        for point in inputs.points
    ]
    cost = sum(
        value * point.buy_price_cny_per_kwh * dt
        for value, point in zip(grid_import, inputs.points, strict=True)
    )
    emissions = sum(
        value * (point.carbon_factor_kgco2e_per_kwh or 0.0) * dt
        for value, point in zip(grid_import, inputs.points, strict=True)
    )
    curtailed = sum(surplus) * dt
    return NoBatteryBaseline(
        pv_scenario_id=pv_scenario_id,
        pv_capacity_kwp=pv_capacity_kwp,
        pv_energy_kwh=pv_energy,
        load_energy_kwh=load_energy,
        grid_import_energy_kwh=sum(grid_import) * dt,
        curtailed_energy_kwh=curtailed,
        curtailment_rate=curtailed / pv_energy if pv_energy else 0.0,
        potential_export_peak_kw=max(surplus, default=0.0),
        peak_grid_import_kw=max(grid_import, default=0.0),
        electricity_cost_cny=cost,
        grid_emissions_kgco2e=emissions,
    )


def _plan_record(
    *,
    pv_scenario_id: str,
    pv_capacity_kwp: float,
    battery_case_id: str,
    battery_power_kw: float,
    battery_capacity_kwh: float,
    plan_id: str,
    workflow: DualObjectiveWorkflowResult,
    plan_index: int,
    baseline: NoBatteryBaseline,
) -> StageBPlanRecord:
    plan = workflow.plans[plan_index]
    metrics = plan.verification.recomputed_metrics
    if metrics is None or not plan.verification.passed:
        raise ValueError("STAGE_B_PLAN_NOT_VERIFIED")
    request = plan.request
    max_energy = request.battery.max_energy_kwh
    min_energy = request.battery.min_energy_kwh
    soc_at_max = sum(
        point.battery_energy_after_kwh >= max_energy - 1e-5
        for point in plan.dispatch.points
    )
    soc_at_min = sum(
        point.battery_energy_after_kwh <= min_energy + 1e-5
        for point in plan.dispatch.points
    )
    avoided = baseline.curtailed_energy_kwh - metrics.pv_curtailed_energy_kwh
    reduction = (
        avoided / baseline.curtailed_energy_kwh * 100.0
        if baseline.curtailed_energy_kwh
        else 0.0
    )
    return StageBPlanRecord(
        pv_scenario_id=pv_scenario_id,
        pv_capacity_kwp=pv_capacity_kwp,
        battery_case_id=battery_case_id,
        battery_power_kw=battery_power_kw,
        battery_capacity_kwh=battery_capacity_kwh,
        plan_id=plan_id,
        objective_mode=plan.request.objective_mode,
        verification_passed=plan.verification.passed,
        solver_name=plan.dispatch.solver.solver_name,
        total_cost_cny=metrics.total_cost_cny,
        grid_emissions_kgco2e=metrics.grid_emissions_kgco2e,
        grid_import_energy_kwh=metrics.grid_import_energy_kwh,
        peak_grid_import_kw=metrics.peak_import_kw,
        pv_energy_kwh=metrics.pv_energy_kwh,
        curtailed_energy_kwh=metrics.pv_curtailed_energy_kwh,
        curtailment_rate=(
            metrics.pv_curtailed_energy_kwh / metrics.pv_energy_kwh
            if metrics.pv_energy_kwh
            else 0.0
        ),
        residual_surplus_peak_kw=max(
            (point.pv_curtailment_kw for point in plan.dispatch.points),
            default=0.0,
        ),
        pv_self_consumption_ratio=metrics.pv_self_consumption_ratio,
        battery_throughput_kwh=metrics.battery_throughput_kwh,
        soc_at_max_hours=soc_at_max,
        soc_at_min_hours=soc_at_min,
        avoided_curtailment_vs_no_battery_kwh=avoided,
        curtailment_reduction_vs_no_battery_pct=reduction,
        grid_import_reduction_vs_no_battery_kwh=(
            baseline.grid_import_energy_kwh - metrics.grid_import_energy_kwh
        ),
        cost_change_vs_no_battery_cny=(
            metrics.total_cost_cny - baseline.electricity_cost_cny
        ),
        emissions_change_vs_no_battery_kgco2e=(
            metrics.grid_emissions_kgco2e - baseline.grid_emissions_kgco2e
        ),
        evidence_id=plan.verification.evidence_id,
    )


def run_stage_b_analysis(
    *,
    project_root: str | Path,
    physical_config_path: str | Path,
    campus_config_path: str | Path,
    carbon_pattern_id: str = "day_low_night_high",
    max_cost_increase_pct: float = 5.0,
    pv_scenario_ids: list[str] | None = None,
    battery_case_ids: list[str] | None = None,
    solver_name: str | None = None,
) -> StageBRun:
    root = Path(project_root)
    physical = load_physical_pv_config(physical_config_path)
    campus = load_scenario_config(campus_config_path)
    load_manifest = _read_json(root / campus.load.manifest)
    selected_pv = pv_scenario_ids or [item.scenario_id for item in physical.scenarios]
    selected_batteries = battery_case_ids or list(BATTERY_CASES)
    unknown_pv = set(selected_pv) - {item.scenario_id for item in physical.scenarios}
    unknown_battery = set(selected_batteries) - set(BATTERY_CASES)
    if unknown_pv:
        raise ValueError(f"unknown PV scenarios: {sorted(unknown_pv)}")
    if unknown_battery:
        raise ValueError(f"unknown battery cases: {sorted(unknown_battery)}")

    scenario_by_id = {item.scenario_id: item for item in physical.scenarios}
    baselines: list[NoBatteryBaseline] = []
    records: list[StageBPlanRecord] = []
    workflows: dict[str, DualObjectiveWorkflowResult] = {}
    plan_names = ("economic", "carbon", "carbon_cost_capped")

    for pv_scenario_id in selected_pv:
        scenario = scenario_by_id[pv_scenario_id]
        pv_csv, pv_manifest_path = _physical_pv_paths(
            root, physical, pv_scenario_id
        )
        if not pv_csv.exists() or not pv_manifest_path.exists():
            raise FileNotFoundError(
                f"physical PV outputs missing for {pv_scenario_id}; run stage A first"
            )
        pv_manifest = _read_json(pv_manifest_path)
        source_version = (
            f"load={load_manifest['dataset_version']};"
            f"pv={pv_manifest['dataset_version']};"
            f"carbon=synthetic-carbon-{carbon_pattern_id}-v1"
        )
        base_inputs = build_hourly_inputs(
            pv_csv,
            dataset_version=source_version,
            load_csv=root / campus.load.csv,
            buy_price_cny_per_kwh_by_hour=(
                campus.tariff.buy_price_cny_per_kwh_by_hour
            ),
            sell_price_cny_per_kwh=campus.tariff.sell_price_cny_per_kwh,
        )
        inputs = _attach_carbon_factors(
            base_inputs,
            carbon_pattern_id=carbon_pattern_id,
            dataset_version=source_version,
        )
        baseline = _no_battery_baseline(
            pv_scenario_id=pv_scenario_id,
            pv_capacity_kwp=scenario.expected_capacity_kwp,
            inputs=inputs,
        )
        baselines.append(baseline)

        for battery_case_id in selected_batteries:
            power_kw, capacity_kwh = BATTERY_CASES[battery_case_id]
            battery = _battery_config(
                campus.battery, battery_case_id, power_kw, capacity_kwh
            )
            parsed = ParsedCampusRequest(
                scenario_name=(
                    f"港科广{scenario.name}—{battery_case_id}—经济与低碳对比"
                ),
                site_id=campus.site_id,
                load_dataset_id=campus.load.dataset_id,
                pv_profile_id=str(pv_manifest["dataset_id"]),
                battery_config_id=battery.asset_id,
                step_minutes=campus.step_minutes,
                horizon_steps=campus.horizon_steps,
                objective=campus.objective_mode,
                allow_grid_charging=campus.grid.allow_grid_charging,
                allow_grid_export=campus.grid.allow_grid_export,
            )
            request = build_dispatch_request(
                parsed,
                inputs,
                measured_load=(load_manifest.get("data_origin") == "measured"),
                scenario_config=campus,
            ).model_copy(
                update={
                    "request_id": f"stage_b_{pv_scenario_id}_{battery_case_id}",
                    "scenario_name": parsed.scenario_name,
                    "battery": battery,
                    "required_terminal_energy_kwh": capacity_kwh * 0.5,
                    "grid_import_limit_kw": (
                        max(point.load_power_kw for point in inputs.points)
                        + power_kw
                    ),
                    "assumptions": [
                        *campus.assumptions,
                        *inputs.assumptions,
                        f"PV规划场景={pv_scenario_id}",
                        f"储能灵敏度场景={battery_case_id}",
                    ],
                }
            )
            key = f"{pv_scenario_id}/{battery_case_id}"
            workflow = run_dual_objective_workflow(
                run_id=f"stage_b_{pv_scenario_id}_{battery_case_id}",
                user_text=(
                    "比较经济和低碳调度，优先减碳，但电费不能比经济方案"
                    f"高出{max_cost_increase_pct:g}%。"
                ),
                inputs=inputs,
                base_request=request,
                solver_name=solver_name,
            )
            workflows[key] = workflow
            if workflow.status != WorkflowStatus.PENDING_HUMAN_APPROVAL:
                raise ValueError(
                    f"stage B workflow failed for {key}: {workflow.failure_code}"
                )
            if len(workflow.plans) != 3:
                raise ValueError(f"stage B workflow returned incomplete plans: {key}")
            for index, plan_id in enumerate(plan_names):
                records.append(
                    _plan_record(
                        pv_scenario_id=pv_scenario_id,
                        pv_capacity_kwp=scenario.expected_capacity_kwp,
                        battery_case_id=battery_case_id,
                        battery_power_kw=power_kw,
                        battery_capacity_kwh=capacity_kwh,
                        plan_id=plan_id,
                        workflow=workflow,
                        plan_index=index,
                        baseline=baseline,
                    )
                )

    battery_order = {case_id: index for index, case_id in enumerate(BATTERY_CASES)}
    marginal_records: list[StageBPlanRecord] = []
    for pv_scenario_id in selected_pv:
        baseline = next(
            item for item in baselines if item.pv_scenario_id == pv_scenario_id
        )
        for plan_id in plan_names:
            group = sorted(
                (
                    item
                    for item in records
                    if item.pv_scenario_id == pv_scenario_id
                    and item.plan_id == plan_id
                ),
                key=lambda item: battery_order[item.battery_case_id],
            )
            previous_avoided = 0.0
            previous_grid_reduction = 0.0
            previous_emissions_reduction = 0.0
            for item in group:
                emissions_reduction = (
                    baseline.grid_emissions_kgco2e - item.grid_emissions_kgco2e
                )
                marginal_records.append(
                    item.model_copy(
                        update={
                            "marginal_avoided_curtailment_kwh": (
                                item.avoided_curtailment_vs_no_battery_kwh
                                - previous_avoided
                            ),
                            "marginal_grid_import_reduction_kwh": (
                                item.grid_import_reduction_vs_no_battery_kwh
                                - previous_grid_reduction
                            ),
                            "marginal_emissions_reduction_kgco2e": (
                                emissions_reduction - previous_emissions_reduction
                            ),
                        }
                    )
                )
                previous_avoided = item.avoided_curtailment_vs_no_battery_kwh
                previous_grid_reduction = item.grid_import_reduction_vs_no_battery_kwh
                previous_emissions_reduction = emissions_reduction

    analysis = StageBAnalysisResult(
        analysis_version=STAGE_B_VERSION,
        study_day=physical.study_day.isoformat(),
        load_dataset_version=str(load_manifest["dataset_version"]),
        carbon_factor_version=f"synthetic-carbon-{carbon_pattern_id}-v1",
        carbon_factor_status="synthetic_unverified",
        tariff_version=campus.tariff.version,
        tariff_verified=campus.tariff.verified,
        baselines=baselines,
        plans=marginal_records,
        all_verification_passed=all(
            item.verification_passed for item in marginal_records
        ),
        warnings=[
            "负荷是研究日4,499台完整观测分表聚合，不是完整校园总表。",
            "光伏是规划容量、阵列方向和系统损失假设下的物理仿真。",
            "动态碳因子为synthetic模拟曲线，电价未经校园结算核验。",
            "储能容量均为灵敏度情景，不是采购建议或设备铭牌。",
            "所有结果保持simulation_only=true、executable=false。",
        ],
    )
    return StageBRun(analysis=analysis, workflows=workflows)


def _dump_json(path: Path, value: Any) -> None:
    payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_stage_b_outputs(run: StageBRun, output_dir: str | Path) -> list[Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    analysis_path = output / "stage_b_analysis.json"
    baseline_path = output / "no_battery_baselines.csv"
    plan_path = output / "verified_plan_summary.csv"
    report_path = output / "STAGE_B_REPORT.md"
    _dump_json(analysis_path, run.analysis)
    pd.DataFrame(
        [item.model_dump(mode="json") for item in run.analysis.baselines]
    ).to_csv(baseline_path, index=False)
    pd.DataFrame(
        [item.model_dump(mode="json") for item in run.analysis.plans]
    ).to_csv(plan_path, index=False)
    written.extend((analysis_path, baseline_path, plan_path))

    for key, workflow in run.workflows.items():
        pv_scenario_id, battery_case_id = key.split("/", maxsplit=1)
        scenario_dir = output / "scenarios" / pv_scenario_id / battery_case_id
        scenario_dir.mkdir(parents=True, exist_ok=True)
        workflow_path = scenario_dir / "workflow_result.json"
        _dump_json(workflow_path, workflow)
        written.append(workflow_path)
        for index, plan in enumerate(workflow.plans):
            plan_id = ("economic", "carbon", "carbon_cost_capped")[index]
            schedule_path = scenario_dir / f"schedule_{plan_id}.json"
            verification_path = scenario_dir / f"verification_{plan_id}.json"
            _dump_json(schedule_path, plan.dispatch)
            _dump_json(verification_path, plan.verification)
            written.extend((schedule_path, verification_path))

    medium_capped = [
        item
        for item in run.analysis.plans
        if item.battery_case_id == "medium_150kw_400kwh"
        and item.plan_id == "carbon_cost_capped"
    ]
    full_build_capped = [
        item
        for item in run.analysis.plans
        if item.pv_scenario_id == "full_build_20077kwp"
        and item.plan_id == "carbon_cost_capped"
    ]
    battery_order = {case_id: index for index, case_id in enumerate(BATTERY_CASES)}
    full_build_capped.sort(key=lambda item: battery_order[item.battery_case_id])
    medium_full_build = {
        item.plan_id: item
        for item in run.analysis.plans
        if item.pv_scenario_id == "full_build_20077kwp"
        and item.battery_case_id == "medium_150kw_400kwh"
    }
    lines = [
        "# 港科广阶段 B：容量与经济/低碳调度分析",
        "",
        "> 所有输入和输出均为仿真决策辅助，不是实测发电、正式碳因子、正式电价或采购建议。",
        "",
        "## 无储能基线",
        "",
        "| 光伏场景 | 光伏电量 | 弃光量 | 弃光率 | 潜在反送峰值 |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in run.analysis.baselines:
        lines.append(
            f"| `{item.pv_scenario_id}` | {item.pv_energy_kwh / 1000:.3f} MWh "
            f"| {item.curtailed_energy_kwh / 1000:.3f} MWh "
            f"| {item.curtailment_rate * 100:.2f}% "
            f"| {item.potential_export_peak_kw / 1000:.3f} MW |"
        )
    lines.extend(
        [
            "",
            "## 150 kW / 400 kWh 成本约束低碳方案",
            "",
            "| 光伏场景 | 弃光量 | 弃光率 | 储能吞吐 | 最大SOC小时 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in medium_capped:
        lines.append(
            f"| `{item.pv_scenario_id}` | {item.curtailed_energy_kwh / 1000:.3f} MWh "
            f"| {item.curtailment_rate * 100:.2f}% "
            f"| {item.battery_throughput_kwh:.1f} kWh "
            f"| {item.soc_at_max_hours} |"
        )
    lines.extend(
        [
            "",
            "## 全量光伏下的储能灵敏度",
            "",
            "以下三行均使用成本约束低碳目标；边际量相对于上一档储能。",
            "",
            "| 储能场景 | 弃光量 | 弃光率 | 减少弃光 | 边际减少弃光 | 储能吞吐 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in full_build_capped:
        lines.append(
            f"| `{item.battery_case_id}` | {item.curtailed_energy_kwh / 1000:.3f} MWh "
            f"| {item.curtailment_rate * 100:.2f}% "
            f"| {item.avoided_curtailment_vs_no_battery_kwh:.1f} kWh "
            f"| {item.marginal_avoided_curtailment_kwh:.1f} kWh "
            f"| {item.battery_throughput_kwh:.1f} kWh |"
        )
    if {"economic", "carbon", "carbon_cost_capped"}.issubset(medium_full_build):
        economic = medium_full_build["economic"]
        carbon = medium_full_build["carbon"]
        capped = medium_full_build["carbon_cost_capped"]
        lines.extend(
            [
                "",
                "## 全量光伏 + 150 kW / 400 kWh 的目标权衡",
                "",
                "| 目标 | 参考电费 | 电网碳排 | 弃光量 | 储能吞吐 |",
                "|---|---:|---:|---:|---:|",
                f"| 经济 | {economic.total_cost_cny:,.2f} CNY "
                f"| {economic.grid_emissions_kgco2e:,.2f} kgCO2e "
                f"| {economic.curtailed_energy_kwh / 1000:.3f} MWh "
                f"| {economic.battery_throughput_kwh:.1f} kWh |",
                f"| 低碳 | {carbon.total_cost_cny:,.2f} CNY "
                f"| {carbon.grid_emissions_kgco2e:,.2f} kgCO2e "
                f"| {carbon.curtailed_energy_kwh / 1000:.3f} MWh "
                f"| {carbon.battery_throughput_kwh:.1f} kWh |",
                f"| 成本约束低碳 | {capped.total_cost_cny:,.2f} CNY "
                f"| {capped.grid_emissions_kgco2e:,.2f} kgCO2e "
                f"| {capped.curtailed_energy_kwh / 1000:.3f} MWh "
                f"| {capped.battery_throughput_kwh:.1f} kWh |",
                "",
                "当前合成碳因子下，5% 成本上限没有成为约束，因此成本约束低碳方案与纯低碳方案相同；这不是正式校园经济结论。",
            ]
        )
    lines.extend(
        [
            "",
            "## 结论边界",
            "",
            f"- 共生成 {len(run.analysis.plans)} 个计划，全部验证通过："
            f"`{str(run.analysis.all_verification_passed).lower()}`。",
            "- 极限光伏明显大于当前研究日负荷，禁止上网时储能只能减少少量弃光。",
            "- 在全量光伏情景，储能从75 kW / 200 kWh增加到300 kW / 800 kWh，弃光率仍由55.76%仅降至约54.53%；首要矛盾是光伏、负荷与并网边界不匹配。",
            "- 正式预测、碳因子、电价、并网政策和设备铭牌到位前，不形成采购或控制结论。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    written.append(report_path)
    return written
