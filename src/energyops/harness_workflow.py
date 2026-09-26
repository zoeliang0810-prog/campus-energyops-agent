"""Fixed-order EnergyOps workflow wrapped by replaceable agent providers."""

from __future__ import annotations

from typing import Any

from .contracts import (
    DispatchInputSeries,
    WorkflowRunResult,
    WorkflowStatus,
)
from .evidence import build_evidence_pack
from .optimization.cvxpy_pv_storage import CvxpyPVStorageOptimizer
from .providers import AgentProvider
from .scenario import build_dispatch_request
from .trace import TraceWriter
from .verifier import verify_dispatch
from .scenario_config import CampusScenarioConfig


FIXED_TOOL_SEQUENCE = (
    "prepare_campus_data",
    "get_data_quality_report",
    "build_simulation_scenario",
    "create_energy_schedule",
    "validate_energy_schedule",
    "build_evidence_pack",
    "explain_verified_result",
    "get_run_trace",
)


def _status_value(report: dict[str, Any]) -> str:
    status = report.get("status", "unknown")
    return getattr(status, "value", status)


def _trace_tool(
    trace: TraceWriter,
    name: str,
    status: str,
    payload: dict[str, Any] | None = None,
    *,
    evidence_id: str | None = None,
) -> None:
    trace.append(
        event_type="tool_completed",
        component=name,
        status=status,
        payload=payload or {},
        evidence_id=evidence_id,
    )


def run_energyops_workflow(
    *,
    run_id: str,
    user_text: str,
    inputs: DispatchInputSeries,
    quality_reports: dict[str, dict[str, Any]],
    measured_load: bool,
    provider: AgentProvider,
    trace: TraceWriter,
    solver_name: str | None = None,
    scenario_config: CampusScenarioConfig | None = None,
) -> WorkflowRunResult:
    """Run the first simulation workflow without allowing model-selected tools."""
    used_tools: list[str] = []
    parsed = None
    request = None
    dispatch = None
    verification = None
    evidence = None
    explanation = None

    trace.append(
        event_type="workflow_started",
        component="energyops_harness",
        status=WorkflowStatus.CREATED.value,
        payload={"provider": provider.name, "measured_load": measured_load},
    )
    try:
        parsed = provider.parse_request(user_text)
        trace.append(
            event_type="state_transition",
            component="energyops_harness",
            status=WorkflowStatus.PARSED.value,
            payload={"scenario_name": parsed.scenario_name},
        )
        # ParsedCampusRequest has already crossed the strict local Pydantic gate.
        trace.append(
            event_type="state_transition",
            component="energyops_harness",
            status=WorkflowStatus.VALIDATED.value,
            payload={"simulation_only": parsed.simulation_only, "executable": False},
        )

        tool = FIXED_TOOL_SEQUENCE[0]
        used_tools.append(tool)
        if len(inputs.points) != 24:
            raise ValueError("PREPARED_DATA_MUST_HAVE_24_HOURLY_POINTS")
        _trace_tool(
            trace,
            tool,
            "pass",
            {"dataset_version": inputs.dataset_version, "points": len(inputs.points)},
        )

        tool = FIXED_TOOL_SEQUENCE[1]
        used_tools.append(tool)
        pv_status = _status_value(quality_reports.get("pv", {}))
        load_status = _status_value(quality_reports.get("load", {}))
        if pv_status == "fail":
            raise ValueError("PV_DATA_QUALITY_FAILED")
        if measured_load and load_status != "pass":
            raise ValueError("MEASURED_LOAD_QUALITY_NOT_PASSED")
        _trace_tool(
            trace,
            tool,
            "pass" if measured_load else "warning",
            {"pv_status": pv_status, "load_status": load_status},
        )
        trace.append(
            event_type="state_transition",
            component="energyops_harness",
            status=WorkflowStatus.QUALITY_CHECKED.value,
            payload={"measured_load": measured_load},
        )

        tool = FIXED_TOOL_SEQUENCE[2]
        used_tools.append(tool)
        request = build_dispatch_request(
            parsed,
            inputs,
            measured_load=measured_load,
            scenario_config=scenario_config,
        )
        _trace_tool(
            trace,
            tool,
            "pass",
            {
                "request_id": request.request_id,
                "objective": request.objective_mode.value,
                "simulation_only": True,
                "executable": False,
            },
        )

        tool = FIXED_TOOL_SEQUENCE[3]
        used_tools.append(tool)
        optimizer = CvxpyPVStorageOptimizer(inputs, solver_name=solver_name)
        dispatch = optimizer.solve(request)
        _trace_tool(
            trace,
            tool,
            dispatch.status.value,
            {"schedule_id": dispatch.schedule_id, "solver": dispatch.solver.model_dump()},
            evidence_id=dispatch.evidence_id,
        )
        if dispatch.status.value != "optimal":
            raise ValueError("OPTIMIZATION_NOT_OPTIMAL")
        trace.append(
            event_type="state_transition",
            component="energyops_harness",
            status=WorkflowStatus.OPTIMIZED.value,
            payload={"schedule_id": dispatch.schedule_id},
        )

        tool = FIXED_TOOL_SEQUENCE[4]
        used_tools.append(tool)
        verification = verify_dispatch(request, dispatch)
        _trace_tool(
            trace,
            tool,
            "pass" if verification.passed else "fail",
            {"violation_count": len(verification.violations), "executable": False},
            evidence_id=verification.evidence_id,
        )
        if not verification.passed:
            raise ValueError("VERIFICATION_FAILED")
        trace.append(
            event_type="state_transition",
            component="energyops_harness",
            status=WorkflowStatus.VERIFIED.value,
            payload={"executable": False},
        )

        tool = FIXED_TOOL_SEQUENCE[5]
        used_tools.append(tool)
        evidence = build_evidence_pack(
            run_id,
            verification,
            request=request,
            result=dispatch,
            data_quality=quality_reports,
        )
        _trace_tool(
            trace,
            tool,
            "pass",
            {"claim_count": len(evidence.claims)},
            evidence_id=verification.evidence_id,
        )

        tool = FIXED_TOOL_SEQUENCE[6]
        used_tools.append(tool)
        explanation = provider.explain_result(parsed, evidence)
        allowed_claims = {claim.claim_id for claim in evidence.claims}
        if not explanation.evidence_claim_ids or not set(
            explanation.evidence_claim_ids
        ).issubset(allowed_claims):
            raise ValueError("EXPLANATION_EVIDENCE_BINDING_FAILED")
        _trace_tool(
            trace,
            tool,
            "pass",
            {
                "provider": explanation.provider,
                "evidence_claim_ids": explanation.evidence_claim_ids,
            },
        )
        trace.append(
            event_type="state_transition",
            component="energyops_harness",
            status=WorkflowStatus.EXPLANATION_READY.value,
            payload={},
        )

        tool = FIXED_TOOL_SEQUENCE[7]
        used_tools.append(tool)
        _trace_tool(trace, tool, "pass", {"trace_path": str(trace.path)})
        trace.append(
            event_type="workflow_completed",
            component="energyops_harness",
            status=WorkflowStatus.PENDING_HUMAN_APPROVAL.value,
            payload={"simulation_only": True, "executable": False},
        )
        return WorkflowRunResult(
            run_id=run_id,
            status=WorkflowStatus.PENDING_HUMAN_APPROVAL,
            tool_sequence=used_tools,
            parsed_request=parsed,
            dispatch_request=request,
            dispatch_result=dispatch,
            verification=verification,
            evidence=evidence,
            explanation=explanation,
            simulation_only=True,
            executable=False,
        )
    except Exception as error:
        failure_code = str(error) if str(error).isupper() else type(error).__name__
        trace.append(
            event_type="workflow_terminated",
            component="energyops_harness",
            status=WorkflowStatus.SAFE_TERMINATED.value,
            payload={"failure_code": failure_code},
        )
        return WorkflowRunResult(
            run_id=run_id,
            status=WorkflowStatus.SAFE_TERMINATED,
            tool_sequence=used_tools,
            parsed_request=parsed,
            dispatch_request=request,
            dispatch_result=dispatch,
            verification=verification,
            evidence=evidence,
            explanation=explanation,
            failure_code=failure_code,
            simulation_only=True,
            executable=False,
        )
