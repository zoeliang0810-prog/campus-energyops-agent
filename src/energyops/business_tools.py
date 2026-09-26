"""Policy-bounded EnergyOps business tools for the self-owned AgentLoop."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import Field, model_validator

from .agent_runtime import ToolDefinition, ToolRegistry
from .contracts import (
    DispatchInputSeries,
    DispatchMetrics,
    DispatchRequest,
    DispatchResult,
    EvidencePack,
    ObjectiveMode,
    ParsedCampusRequest,
    StrictModel,
    VerificationReport,
    WorkflowStatus,
)
from .evidence import build_evidence_pack
from .optimization.cvxpy_pv_storage import CvxpyPVStorageOptimizer
from .scenario import build_dispatch_request
from .scenario_config import CampusScenarioConfig
from .trace import TraceWriter
from .verifier import verify_dispatch


CREATE_VERIFIED_DISPATCH = "create_verified_dispatch"


class CreateVerifiedDispatchArgs(StrictModel):
    """Only business choices exposed to the language model."""

    scenario_id: Literal["hkust_gz_weather_pv_storage_baseline_v1"] = (
        "hkust_gz_weather_pv_storage_baseline_v1"
    )
    objective: Literal["cost", "economic", "carbon", "balanced"] = "cost"
    max_cost_increase_pct: float | None = Field(default=None, ge=0, le=100)
    simulation_only: Literal[True] = True
    allow_grid_export: Literal[False] = False

    @model_validator(mode="after")
    def validate_cost_limit(self) -> "CreateVerifiedDispatchArgs":
        if self.max_cost_increase_pct is not None and self.objective not in {
            "carbon",
            "balanced",
        }:
            raise ValueError(
                "max_cost_increase_pct is only valid for carbon-aware objectives"
            )
        return self


class VerifiedDispatchArtifacts(StrictModel):
    request: str
    schedule: str
    verification: str
    evidence: str


class CreateVerifiedDispatchResult(StrictModel):
    tool_name: Literal["create_verified_dispatch"] = CREATE_VERIFIED_DISPATCH
    run_id: str
    status: WorkflowStatus
    scenario_id: str
    objective: ObjectiveMode
    dataset_version: str
    quality_status: dict[str, str] = Field(default_factory=dict)
    schedule_id: str | None = None
    verification_passed: bool = False
    metrics: DispatchMetrics | None = None
    evidence: EvidencePack | None = None
    artifacts: VerifiedDispatchArtifacts | None = None
    failure_code: str | None = None
    simulation_only: Literal[True] = True
    executable: Literal[False] = False


@dataclass(frozen=True)
class VerifiedDispatchContext:
    inputs: DispatchInputSeries
    quality_reports: dict[str, dict[str, Any]]
    measured_load: bool
    scenario_config: CampusScenarioConfig
    trace: TraceWriter
    artifact_root: Path
    solver_name: str | None = None


class CompositeDispatchError(RuntimeError):
    """A stable failure code raised inside the deterministic composite tool."""


def _status_value(report: dict[str, Any]) -> str:
    status = report.get("status", "unknown")
    return str(getattr(status, "value", status))


def _failure_code(error: Exception) -> str:
    value = str(error)
    return value if value.isupper() and " " not in value else type(error).__name__


def _write_model(path: Path, value: StrictModel) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


class CreateVerifiedDispatchTool:
    """Execute the complete deterministic dispatch chain behind one tool call."""

    def __init__(self, context: VerifiedDispatchContext) -> None:
        self.context = context

    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name=CREATE_VERIFIED_DISPATCH,
            description=(
                "Create one simulation-only HKUST(GZ) dispatch plan. This composite "
                "tool always runs data-quality checks, deterministic optimization, "
                "independent verification, Evidence Pack creation, artifact persistence, "
                "and audit tracing. It never controls equipment."
            ),
            args_model=CreateVerifiedDispatchArgs,
            handler=self,
        )

    def _trace(
        self,
        event_type: str,
        status: str,
        *,
        run_id: str,
        payload: dict[str, Any] | None = None,
        evidence_id: str | None = None,
    ) -> None:
        merged = {"dispatch_run_id": run_id, **(payload or {})}
        try:
            self.context.trace.append(
                event_type=event_type,
                component=CREATE_VERIFIED_DISPATCH,
                status=status,
                payload=merged,
                evidence_id=evidence_id,
            )
        except OSError as error:
            raise CompositeDispatchError("TRACE_WRITE_FAILED") from error

    def _quality_gate(self) -> dict[str, str]:
        if len(self.context.inputs.points) != 24:
            raise CompositeDispatchError("PREPARED_DATA_MUST_HAVE_24_HOURLY_POINTS")
        if not self.context.inputs.simulation_only:
            raise CompositeDispatchError("SIMULATION_POLICY_INVALID")
        if self.context.inputs.power_unit != "kW":
            raise CompositeDispatchError("INPUT_POWER_UNIT_INVALID")

        pv_status = _status_value(self.context.quality_reports.get("pv", {}))
        load_status = _status_value(self.context.quality_reports.get("load", {}))
        if pv_status == "fail":
            raise CompositeDispatchError("PV_DATA_QUALITY_FAILED")
        if self.context.measured_load and load_status != "pass":
            raise CompositeDispatchError("MEASURED_LOAD_QUALITY_NOT_PASSED")
        return {"pv": pv_status, "load": load_status}

    def _base_request(self) -> DispatchRequest:
        config = self.context.scenario_config
        parsed = ParsedCampusRequest(
            scenario_name=config.scenario_name,
            site_id=config.site_id,
            load_dataset_id=config.load.dataset_id,
            pv_profile_id=config.pv.dataset_id,
            battery_config_id=config.battery.asset_id,
            step_minutes=config.step_minutes,
            horizon_steps=config.horizon_steps,
            objective=config.objective_mode,
            allow_grid_charging=config.grid.allow_grid_charging,
            allow_grid_export=config.grid.allow_grid_export,
            simulation_only=True,
            executable=False,
        )
        return build_dispatch_request(
            parsed,
            self.context.inputs,
            measured_load=self.context.measured_load,
            scenario_config=config,
        )

    def _request_for_objective(
        self,
        base: DispatchRequest,
        *,
        objective: ObjectiveMode,
        run_id: str,
        max_total_cost_cny: float | None = None,
        max_cost_increase_pct: float | None = None,
    ) -> DispatchRequest:
        payload = base.model_dump(mode="python")
        payload.update(
            {
                "request_id": f"{base.request_id}_{objective.value}_{run_id[-8:]}",
                "objective_mode": objective,
                "carbon_factor_version": self.context.inputs.carbon_factor_version,
                "max_total_cost_cny": max_total_cost_cny,
                "max_cost_increase_pct": max_cost_increase_pct,
                "simulation_only": True,
                "executable": False,
            }
        )
        return DispatchRequest.model_validate(payload)

    def _solve_and_verify(
        self,
        request: DispatchRequest,
        *,
        run_id: str,
        purpose: str,
    ) -> tuple[DispatchResult, VerificationReport]:
        dispatch = CvxpyPVStorageOptimizer(
            self.context.inputs,
            solver_name=self.context.solver_name,
        ).solve(request)
        self._trace(
            "composite_optimization_completed",
            dispatch.status.value,
            run_id=run_id,
            payload={
                "purpose": purpose,
                "objective": request.objective_mode.value,
                "schedule_id": dispatch.schedule_id,
                "solver": dispatch.solver.model_dump(mode="json"),
                "reason_codes": dispatch.reason_codes,
            },
            evidence_id=dispatch.evidence_id,
        )

        verification = verify_dispatch(request, dispatch)
        self._trace(
            "composite_verification_completed",
            "pass" if verification.passed else "fail",
            run_id=run_id,
            payload={
                "purpose": purpose,
                "schedule_id": dispatch.schedule_id,
                "violation_count": len(verification.violations),
                "executable": verification.executable,
            },
            evidence_id=verification.evidence_id,
        )
        if not verification.passed:
            if dispatch.reason_codes:
                raise CompositeDispatchError(dispatch.reason_codes[0])
            raise CompositeDispatchError("VERIFICATION_FAILED")
        if verification.executable:
            raise CompositeDispatchError("SIMULATION_RESULT_MARKED_EXECUTABLE")
        return dispatch, verification

    def _persist(
        self,
        *,
        run_id: str,
        request: DispatchRequest,
        dispatch: DispatchResult,
        verification: VerificationReport,
        evidence: EvidencePack,
    ) -> VerifiedDispatchArtifacts:
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", run_id)
        directory = self.context.artifact_root.resolve() / safe_id
        paths = {
            "request": directory / "request.json",
            "schedule": directory / "schedule.json",
            "verification": directory / "verification.json",
            "evidence": directory / "evidence.json",
        }
        try:
            directory.mkdir(parents=True, exist_ok=False)
            _write_model(paths["request"], request)
            _write_model(paths["schedule"], dispatch)
            _write_model(paths["verification"], verification)
            _write_model(paths["evidence"], evidence)
        except OSError as error:
            raise CompositeDispatchError("ARTIFACT_WRITE_FAILED") from error
        return VerifiedDispatchArtifacts(
            **{name: str(path) for name, path in paths.items()}
        )

    def __call__(
        self, args: CreateVerifiedDispatchArgs
    ) -> CreateVerifiedDispatchResult:
        run_id = f"{self.context.trace.run_id}_dispatch_{uuid4().hex[:8]}"
        objective = ObjectiveMode(args.objective)
        quality_status: dict[str, str] = {}
        try:
            self._trace(
                "composite_dispatch_started",
                "running",
                run_id=run_id,
                payload={
                    "scenario_id": args.scenario_id,
                    "objective": objective.value,
                    "simulation_only": True,
                    "executable": False,
                },
            )
            if args.scenario_id != self.context.scenario_config.scenario_id:
                raise CompositeDispatchError("SCENARIO_NOT_AVAILABLE")
            if args.allow_grid_export:
                raise CompositeDispatchError("GRID_EXPORT_NOT_ALLOWED")

            quality_status = self._quality_gate()
            self._trace(
                "composite_quality_gate_completed",
                "pass",
                run_id=run_id,
                payload={
                    "quality_status": quality_status,
                    "measured_load": self.context.measured_load,
                    "dataset_version": self.context.inputs.dataset_version,
                },
            )

            if objective in {ObjectiveMode.CARBON, ObjectiveMode.BALANCED}:
                if not self.context.inputs.carbon_factor_version or any(
                    point.carbon_factor_kgco2e_per_kwh is None
                    for point in self.context.inputs.points
                ):
                    raise CompositeDispatchError("CARBON_FACTOR_REQUIRED")

            base = self._base_request()
            max_total_cost_cny = None
            if args.max_cost_increase_pct is not None:
                economic_request = self._request_for_objective(
                    base,
                    objective=ObjectiveMode.ECONOMIC,
                    run_id=run_id,
                )
                _, economic_verification = self._solve_and_verify(
                    economic_request,
                    run_id=run_id,
                    purpose="economic_cost_cap_baseline",
                )
                economic_metrics = economic_verification.recomputed_metrics
                if economic_metrics is None:
                    raise CompositeDispatchError("ECONOMIC_BASELINE_METRICS_MISSING")
                max_total_cost_cny = economic_metrics.total_cost_cny * (
                    1 + args.max_cost_increase_pct / 100
                )

            request = self._request_for_objective(
                base,
                objective=objective,
                run_id=run_id,
                max_total_cost_cny=max_total_cost_cny,
                max_cost_increase_pct=args.max_cost_increase_pct,
            )
            self._trace(
                "composite_request_validated",
                "pass",
                run_id=run_id,
                payload={
                    "request_id": request.request_id,
                    "objective": request.objective_mode.value,
                    "max_total_cost_cny": request.max_total_cost_cny,
                    "simulation_only": request.simulation_only,
                    "executable": request.executable,
                },
            )

            dispatch, verification = self._solve_and_verify(
                request,
                run_id=run_id,
                purpose="requested_dispatch",
            )
            evidence = build_evidence_pack(
                run_id,
                verification,
                request=request,
                result=dispatch,
                data_quality=self.context.quality_reports,
            )
            self._trace(
                "composite_evidence_built",
                "pass",
                run_id=run_id,
                payload={"claim_count": len(evidence.claims)},
                evidence_id=verification.evidence_id,
            )
            artifacts = self._persist(
                run_id=run_id,
                request=request,
                dispatch=dispatch,
                verification=verification,
                evidence=evidence,
            )
            self._trace(
                "composite_artifacts_persisted",
                "pass",
                run_id=run_id,
                payload={
                    "artifact_names": list(VerifiedDispatchArtifacts.model_fields)
                },
                evidence_id=verification.evidence_id,
            )
            self._trace(
                "composite_dispatch_completed",
                WorkflowStatus.PENDING_HUMAN_APPROVAL.value,
                run_id=run_id,
                payload={
                    "schedule_id": dispatch.schedule_id,
                    "verification_passed": True,
                    "simulation_only": True,
                    "executable": False,
                },
                evidence_id=verification.evidence_id,
            )
            return CreateVerifiedDispatchResult(
                run_id=run_id,
                status=WorkflowStatus.PENDING_HUMAN_APPROVAL,
                scenario_id=args.scenario_id,
                objective=objective,
                dataset_version=self.context.inputs.dataset_version,
                quality_status=quality_status,
                schedule_id=dispatch.schedule_id,
                verification_passed=True,
                metrics=verification.recomputed_metrics,
                evidence=evidence,
                artifacts=artifacts,
            )
        except Exception as error:
            code = _failure_code(error)
            try:
                self._trace(
                    "composite_dispatch_terminated",
                    WorkflowStatus.SAFE_TERMINATED.value,
                    run_id=run_id,
                    payload={
                        "failure_code": code,
                        "simulation_only": True,
                        "executable": False,
                    },
                )
            except CompositeDispatchError:
                code = "TRACE_WRITE_FAILED"
            return CreateVerifiedDispatchResult(
                run_id=run_id,
                status=WorkflowStatus.SAFE_TERMINATED,
                scenario_id=args.scenario_id,
                objective=objective,
                dataset_version=self.context.inputs.dataset_version,
                quality_status=quality_status,
                failure_code=code,
            )


def register_verified_dispatch_tool(
    registry: ToolRegistry,
    context: VerifiedDispatchContext,
) -> None:
    """Register the first EnergyOps business tool and nothing else."""

    registry.register(CreateVerifiedDispatchTool(context).definition())
