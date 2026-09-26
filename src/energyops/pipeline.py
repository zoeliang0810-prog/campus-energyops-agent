"""Fixed verified workflow used by the first EnergyOps harness version."""

from __future__ import annotations

from typing import Protocol

from .contracts import DispatchRequest, DispatchResult, EvidencePack, VerificationReport
from .evidence import build_evidence_pack
from .trace import TraceWriter
from .verifier import verify_dispatch


class DispatchOptimizer(Protocol):
    def solve(self, request: DispatchRequest) -> DispatchResult: ...


def run_verified_dispatch(
    *,
    run_id: str,
    request: DispatchRequest,
    optimizer: DispatchOptimizer,
    trace: TraceWriter,
) -> tuple[DispatchResult, VerificationReport, EvidencePack]:
    trace.append(
        event_type="workflow_started",
        component="verified_dispatch",
        status="started",
        payload={
            "request_id": request.request_id,
            "dataset_version": request.dataset_version,
            "operating_mode": request.operating_mode.value,
        },
    )
    result = optimizer.solve(request)
    trace.append(
        event_type="tool_completed",
        component="pv_storage_dispatch_operator",
        status=result.status.value,
        payload={
            "schedule_id": result.schedule_id,
            "solver": result.solver.model_dump(),
        },
        evidence_id=result.evidence_id,
    )
    report = verify_dispatch(request, result)
    trace.append(
        event_type="verifier_completed",
        component="constraint_and_accounting_verifier",
        status="pass" if report.passed else "fail",
        payload={
            "schedule_id": result.schedule_id,
            "violation_count": len(report.violations),
            "executable": report.executable,
        },
        evidence_id=report.evidence_id,
    )
    evidence = build_evidence_pack(
        run_id,
        report,
        request=request,
        result=result,
    )
    return result, report, evidence
