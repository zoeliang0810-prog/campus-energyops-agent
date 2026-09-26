"""Read-only scenario, quality, comparison, and evidence tools for EnergyOps."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from .agent_runtime import ToolDefinition, ToolRegistry
from .business_tools import (
    VerifiedDispatchContext,
    register_verified_dispatch_tool,
)
from .contracts import (
    DispatchMetrics,
    DispatchRequest,
    DispatchResult,
    EvidenceClaim,
    EvidencePack,
    ObjectiveMode,
    StrictModel,
    VerificationReport,
)
from .verifier import verify_dispatch


GET_AVAILABLE_SCENARIOS = "get_available_scenarios"
GET_DATA_QUALITY_REPORT = "get_data_quality_report"
COMPARE_VERIFIED_PLANS = "compare_verified_plans"
EXPLAIN_EVIDENCE = "explain_evidence"


class BusinessQueryError(RuntimeError):
    """Stable failure raised by read-only EnergyOps business tools."""


class GetAvailableScenariosArgs(StrictModel):
    site_id: Literal["hkust_gz"] = "hkust_gz"


class AvailableScenario(StrictModel):
    scenario_id: str
    scenario_name: str
    site_id: str
    study_day: str
    load_dataset_id: str
    pv_profile_id: str
    battery_config_id: str
    available_objectives: list[ObjectiveMode]
    unavailable_objectives: dict[str, str]
    simulation_only: Literal[True] = True
    executable: Literal[False] = False


class AvailableScenariosResult(StrictModel):
    status: Literal["completed", "safe_terminated"]
    scenarios: list[AvailableScenario] = Field(default_factory=list)
    failure_code: str | None = None
    simulation_only: Literal[True] = True
    executable: Literal[False] = False


class GetDataQualityReportArgs(StrictModel):
    dataset: Literal["all", "load", "pv"] = "all"


class DataQualityToolResult(StrictModel):
    status: Literal["completed", "safe_terminated"]
    dataset_version: str
    reports: dict[str, dict[str, Any]] = Field(default_factory=dict)
    economic_dispatch_ready: bool = False
    carbon_dispatch_ready: bool = False
    failure_code: str | None = None
    simulation_only: Literal[True] = True
    executable: Literal[False] = False


class CompareVerifiedPlansArgs(StrictModel):
    dispatch_run_ids: list[str] = Field(min_length=2, max_length=5)
    baseline_run_id: str

    @model_validator(mode="after")
    def validate_run_ids(self) -> "CompareVerifiedPlansArgs":
        if len(set(self.dispatch_run_ids)) != len(self.dispatch_run_ids):
            raise ValueError("dispatch_run_ids must be unique")
        if self.baseline_run_id not in self.dispatch_run_ids:
            raise ValueError("baseline_run_id must be included in dispatch_run_ids")
        return self


class VerifiedPlanComparisonRow(StrictModel):
    dispatch_run_id: str
    schedule_id: str
    objective: ObjectiveMode
    metrics: DispatchMetrics
    cost_change_vs_baseline_cny: float
    emissions_change_vs_baseline_kgco2e: float | None = None
    evidence_id: str
    verification_passed: Literal[True] = True


class CompareVerifiedPlansResult(StrictModel):
    status: Literal["pending_human_approval", "safe_terminated"]
    baseline_run_id: str
    dataset_version: str | None = None
    plans: list[VerifiedPlanComparisonRow] = Field(default_factory=list)
    failure_code: str | None = None
    simulation_only: Literal[True] = True
    executable: Literal[False] = False


class ExplainEvidenceArgs(StrictModel):
    dispatch_run_id: str
    claim_ids: list[str] = Field(default_factory=list, max_length=20)


class ExplainEvidenceResult(StrictModel):
    status: Literal["completed", "safe_terminated"]
    dispatch_run_id: str
    schedule_id: str | None = None
    claims: list[EvidenceClaim] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    failure_code: str | None = None
    simulation_only: Literal[True] = True
    executable: Literal[False] = False


def _status_value(report: dict[str, Any]) -> str:
    status = report.get("status", "unknown")
    return str(getattr(status, "value", status))


def _failure_code(error: Exception) -> str:
    value = str(error)
    return value if value.isupper() and " " not in value else type(error).__name__


class EnergyOpsBusinessQueryTools:
    """Read and compare only artifacts created by the verified dispatch tool."""

    def __init__(self, context: VerifiedDispatchContext) -> None:
        self.context = context

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name=GET_AVAILABLE_SCENARIOS,
                description=(
                    "List the configured HKUST(GZ) simulation scenario and only the "
                    "objectives supported by the currently loaded inputs."
                ),
                args_model=GetAvailableScenariosArgs,
                handler=self.get_available_scenarios,
            ),
            ToolDefinition(
                name=GET_DATA_QUALITY_REPORT,
                description=(
                    "Return versioned load and PV quality findings and whether economic "
                    "or carbon dispatch is currently ready."
                ),
                args_model=GetDataQualityReportArgs,
                handler=self.get_data_quality_report,
            ),
            ToolDefinition(
                name=COMPARE_VERIFIED_PLANS,
                description=(
                    "Re-verify and compare two to five dispatch runs created by "
                    "create_verified_dispatch. Arbitrary files and unverified plans are rejected."
                ),
                args_model=CompareVerifiedPlansArgs,
                handler=self.compare_verified_plans,
            ),
            ToolDefinition(
                name=EXPLAIN_EVIDENCE,
                description=(
                    "Return selected grounded Evidence claims and mandatory warnings for one "
                    "verified dispatch run. It never calculates new energy values."
                ),
                args_model=ExplainEvidenceArgs,
                handler=self.explain_evidence,
            ),
        ]

    def _trace(
        self,
        event_type: str,
        status: str,
        *,
        query_run_id: str,
        payload: dict[str, Any] | None = None,
        evidence_id: str | None = None,
    ) -> None:
        self.context.trace.append(
            event_type=event_type,
            component="energyops_business_tools",
            status=status,
            payload={"query_run_id": query_run_id, **(payload or {})},
            evidence_id=evidence_id,
        )

    def _carbon_ready(self) -> bool:
        return bool(self.context.inputs.carbon_factor_version) and all(
            point.carbon_factor_kgco2e_per_kwh is not None
            for point in self.context.inputs.points
        )

    def _economic_ready(self) -> bool:
        pv = _status_value(self.context.quality_reports.get("pv", {}))
        load = _status_value(self.context.quality_reports.get("load", {}))
        return pv != "fail" and (
            not self.context.measured_load or load == "pass"
        )

    def _dispatch_directory(self, run_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,180}", run_id):
            raise BusinessQueryError("INVALID_DISPATCH_RUN_ID")
        root = self.context.artifact_root.resolve()
        directory = (root / run_id).resolve()
        if directory.parent != root or not directory.is_dir():
            raise BusinessQueryError("DISPATCH_RUN_NOT_FOUND")
        return directory

    def _load_verified_run(
        self, run_id: str
    ) -> tuple[DispatchRequest, DispatchResult, VerificationReport, EvidencePack]:
        directory = self._dispatch_directory(run_id)
        try:
            request = DispatchRequest.model_validate_json(
                (directory / "request.json").read_text(encoding="utf-8")
            )
            dispatch = DispatchResult.model_validate_json(
                (directory / "schedule.json").read_text(encoding="utf-8")
            )
            stored_verification = VerificationReport.model_validate_json(
                (directory / "verification.json").read_text(encoding="utf-8")
            )
            evidence = EvidencePack.model_validate_json(
                (directory / "evidence.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError) as error:
            raise BusinessQueryError("DISPATCH_ARTIFACT_INVALID") from error

        if (
            request.request_id != dispatch.request_id
            or request.dataset_version != dispatch.dataset_version
            or dispatch.schedule_id != stored_verification.schedule_id
            or dispatch.schedule_id != evidence.schedule_id
        ):
            raise BusinessQueryError("DISPATCH_ARTIFACT_BINDING_FAILED")
        if not request.simulation_only or request.executable:
            raise BusinessQueryError("DISPATCH_POLICY_INVALID")

        current_verification = verify_dispatch(request, dispatch)
        if (
            not stored_verification.passed
            or stored_verification.executable
            or not current_verification.passed
            or current_verification.executable
        ):
            raise BusinessQueryError("DISPATCH_REVERIFICATION_FAILED")
        return request, dispatch, current_verification, evidence

    def get_available_scenarios(
        self, args: GetAvailableScenariosArgs
    ) -> AvailableScenariosResult:
        query_id = f"{self.context.trace.run_id}_scenarios_{uuid4().hex[:8]}"
        try:
            config = self.context.scenario_config
            available = [ObjectiveMode.COST, ObjectiveMode.ECONOMIC]
            unavailable: dict[str, str] = {}
            if self._carbon_ready():
                available.extend([ObjectiveMode.CARBON, ObjectiveMode.BALANCED])
            else:
                unavailable.update(
                    {
                        "carbon": "CARBON_FACTOR_REQUIRED",
                        "balanced": "CARBON_FACTOR_REQUIRED",
                    }
                )
            scenario = AvailableScenario(
                scenario_id=config.scenario_id,
                scenario_name=config.scenario_name,
                site_id=args.site_id,
                study_day=config.study_day.isoformat(),
                load_dataset_id=config.load.dataset_id,
                pv_profile_id=config.pv.dataset_id,
                battery_config_id=config.battery.asset_id,
                available_objectives=available,
                unavailable_objectives=unavailable,
            )
            self._trace(
                "available_scenarios_returned",
                "completed",
                query_run_id=query_id,
                payload={
                    "scenario_ids": [scenario.scenario_id],
                    "available_objectives": [item.value for item in available],
                },
            )
            return AvailableScenariosResult(status="completed", scenarios=[scenario])
        except Exception as error:
            code = _failure_code(error)
            self._trace(
                "available_scenarios_terminated",
                "safe_terminated",
                query_run_id=query_id,
                payload={"failure_code": code},
            )
            return AvailableScenariosResult(
                status="safe_terminated", failure_code=code
            )

    def get_data_quality_report(
        self, args: GetDataQualityReportArgs
    ) -> DataQualityToolResult:
        query_id = f"{self.context.trace.run_id}_quality_{uuid4().hex[:8]}"
        selected = (
            self.context.quality_reports
            if args.dataset == "all"
            else {args.dataset: self.context.quality_reports.get(args.dataset, {})}
        )
        economic_ready = self._economic_ready()
        carbon_ready = economic_ready and self._carbon_ready()
        self._trace(
            "data_quality_returned",
            "completed",
            query_run_id=query_id,
            payload={
                "datasets": sorted(selected),
                "economic_dispatch_ready": economic_ready,
                "carbon_dispatch_ready": carbon_ready,
                "dataset_version": self.context.inputs.dataset_version,
            },
        )
        return DataQualityToolResult(
            status="completed",
            dataset_version=self.context.inputs.dataset_version,
            reports=selected,
            economic_dispatch_ready=economic_ready,
            carbon_dispatch_ready=carbon_ready,
        )

    def compare_verified_plans(
        self, args: CompareVerifiedPlansArgs
    ) -> CompareVerifiedPlansResult:
        query_id = f"{self.context.trace.run_id}_compare_{uuid4().hex[:8]}"
        try:
            loaded = {
                run_id: self._load_verified_run(run_id)
                for run_id in args.dispatch_run_ids
            }
            versions = {item[0].dataset_version for item in loaded.values()}
            sites = {item[0].site_id for item in loaded.values()}
            horizons = {
                (item[0].horizon_start, item[0].horizon_steps, item[0].step_minutes)
                for item in loaded.values()
            }
            if len(versions) != 1 or len(sites) != 1 or len(horizons) != 1:
                raise BusinessQueryError("PLAN_COMPARISON_INPUT_MISMATCH")

            baseline_request, _, baseline_verification, _ = loaded[
                args.baseline_run_id
            ]
            baseline_metrics = baseline_verification.recomputed_metrics
            if baseline_metrics is None:
                raise BusinessQueryError("VERIFIED_METRICS_MISSING")
            baseline_has_carbon = bool(baseline_request.carbon_factor_version)
            rows: list[VerifiedPlanComparisonRow] = []
            for run_id in args.dispatch_run_ids:
                request, _, verification, _ = loaded[run_id]
                metrics = verification.recomputed_metrics
                if metrics is None:
                    raise BusinessQueryError("VERIFIED_METRICS_MISSING")
                emissions_delta = None
                if baseline_has_carbon and request.carbon_factor_version:
                    emissions_delta = (
                        metrics.grid_emissions_kgco2e
                        - baseline_metrics.grid_emissions_kgco2e
                    )
                rows.append(
                    VerifiedPlanComparisonRow(
                        dispatch_run_id=run_id,
                        schedule_id=verification.schedule_id,
                        objective=request.objective_mode,
                        metrics=metrics,
                        cost_change_vs_baseline_cny=(
                            metrics.total_cost_cny - baseline_metrics.total_cost_cny
                        ),
                        emissions_change_vs_baseline_kgco2e=emissions_delta,
                        evidence_id=verification.evidence_id,
                    )
                )
            dataset_version = versions.pop()
            self._trace(
                "verified_plans_compared",
                "pending_human_approval",
                query_run_id=query_id,
                payload={
                    "dispatch_run_ids": args.dispatch_run_ids,
                    "baseline_run_id": args.baseline_run_id,
                    "dataset_version": dataset_version,
                    "plan_count": len(rows),
                },
            )
            return CompareVerifiedPlansResult(
                status="pending_human_approval",
                baseline_run_id=args.baseline_run_id,
                dataset_version=dataset_version,
                plans=rows,
            )
        except Exception as error:
            code = _failure_code(error)
            self._trace(
                "verified_plans_comparison_terminated",
                "safe_terminated",
                query_run_id=query_id,
                payload={"failure_code": code},
            )
            return CompareVerifiedPlansResult(
                status="safe_terminated",
                baseline_run_id=args.baseline_run_id,
                failure_code=code,
            )

    def explain_evidence(
        self, args: ExplainEvidenceArgs
    ) -> ExplainEvidenceResult:
        query_id = f"{self.context.trace.run_id}_evidence_{uuid4().hex[:8]}"
        try:
            _, _, verification, evidence = self._load_verified_run(
                args.dispatch_run_id
            )
            by_id = {claim.claim_id: claim for claim in evidence.claims}
            requested = args.claim_ids or [
                "verification_status",
                "executable_status",
                "release_status",
                "objective_mode",
                "dataset_version",
                "total_cost_cny",
                "grid_emissions_kgco2e",
                "grid_emissions_status",
                "peak_import_kw",
                "pv_energy_kwh",
                "pv_curtailed_energy_kwh",
                "pv_self_consumption_ratio",
                "battery_throughput_kwh",
                "load_quality_status",
                "pv_quality_status",
                "tariff_verified",
            ]
            unknown = [claim_id for claim_id in requested if claim_id not in by_id]
            unavailable_defaults = {"grid_emissions_kgco2e"}
            real_unknown = [item for item in unknown if item not in unavailable_defaults]
            if real_unknown:
                raise BusinessQueryError("EVIDENCE_CLAIM_NOT_FOUND")
            claims = [by_id[claim_id] for claim_id in requested if claim_id in by_id]
            warnings = [
                "结果仅用于仿真和决策辅助，不可作为设备控制指令。",
                "所有数值只能引用返回 Claim 的 evidence_id 与 path。",
            ]
            for suffix in ("findings", "warnings"):
                for dataset in ("load", "pv"):
                    claim = by_id.get(f"{dataset}_quality_{suffix}")
                    if claim is not None and isinstance(claim.value, list):
                        warnings.extend(str(item) for item in claim.value)
            if by_id.get("tariff_verified") and not by_id["tariff_verified"].value:
                warnings.append("当前电价未经港科广正式结算核验。")
            self._trace(
                "verified_evidence_returned",
                "completed",
                query_run_id=query_id,
                payload={
                    "dispatch_run_id": args.dispatch_run_id,
                    "schedule_id": verification.schedule_id,
                    "claim_ids": [claim.claim_id for claim in claims],
                },
                evidence_id=verification.evidence_id,
            )
            return ExplainEvidenceResult(
                status="completed",
                dispatch_run_id=args.dispatch_run_id,
                schedule_id=verification.schedule_id,
                claims=claims,
                warnings=list(dict.fromkeys(warnings)),
            )
        except Exception as error:
            code = _failure_code(error)
            self._trace(
                "verified_evidence_terminated",
                "safe_terminated",
                query_run_id=query_id,
                payload={"failure_code": code},
            )
            return ExplainEvidenceResult(
                status="safe_terminated",
                dispatch_run_id=args.dispatch_run_id,
                failure_code=code,
            )


def register_energyops_business_tools(
    registry: ToolRegistry,
    context: VerifiedDispatchContext,
) -> None:
    """Register exactly the five approved EnergyOps business tools."""

    register_verified_dispatch_tool(registry, context)
    for definition in EnergyOpsBusinessQueryTools(context).definitions():
        registry.register(definition)
