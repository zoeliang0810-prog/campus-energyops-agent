from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import energyops.business_tools as business_tools
from energyops.agent_runtime import (
    AgentLoop,
    AgentModelStep,
    AgentPolicy,
    AgentToolCall,
    ToolRegistry,
)
from energyops.business_tools import (
    CREATE_VERIFIED_DISPATCH,
    CreateVerifiedDispatchArgs,
    CreateVerifiedDispatchTool,
    VerifiedDispatchContext,
    register_verified_dispatch_tool,
)
from energyops.business_query_tools import (
    COMPARE_VERIFIED_PLANS,
    EXPLAIN_EVIDENCE,
    GET_AVAILABLE_SCENARIOS,
    GET_DATA_QUALITY_REPORT,
    CompareVerifiedPlansArgs,
    EnergyOpsBusinessQueryTools,
    ExplainEvidenceArgs,
    GetAvailableScenariosArgs,
    GetDataQualityReportArgs,
    register_energyops_business_tools,
)
from energyops.contracts import (
    DispatchResult,
    DispatchStatus,
    SolverMetadata,
    WorkflowStatus,
)
from energyops.scenario import build_hourly_inputs
from energyops.scenario_config import load_scenario_config
from energyops.trace import TraceWriter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_context(
    tmp_path: Path,
    *,
    load_status: str = "pass",
) -> VerifiedDispatchContext:
    config = load_scenario_config(
        PROJECT_ROOT / "config/scenarios/hkust_gz_campus_baseline.json"
    )
    pv_manifest = read_json(PROJECT_ROOT / config.pv.manifest)
    load_manifest = read_json(PROJECT_ROOT / config.load.manifest)
    inputs = build_hourly_inputs(
        PROJECT_ROOT / config.pv.csv,
        dataset_version=(
            f"pv={pv_manifest['dataset_version']};"
            f"load={load_manifest['dataset_version']}"
        ),
        load_csv=PROJECT_ROOT / config.load.csv,
        buy_price_cny_per_kwh_by_hour=(
            config.tariff.buy_price_cny_per_kwh_by_hour
        ),
        sell_price_cny_per_kwh=config.tariff.sell_price_cny_per_kwh,
    )
    return VerifiedDispatchContext(
        inputs=inputs,
        quality_reports={
            "pv": {"status": "warning", "evidence_id": "quality:pv"},
            "load": {
                "status": load_status,
                "evidence_id": "quality:load",
                "findings": ["test fixture"],
            },
        },
        measured_load=True,
        scenario_config=config,
        trace=TraceWriter(tmp_path / "trace.jsonl", "business-tool-test"),
        artifact_root=tmp_path / "dispatches",
    )


class ScriptedTransport:
    def __init__(self, steps: list[AgentModelStep]) -> None:
        self.steps = list(steps)
        self.tool_schemas: list[dict] = []

    def complete(self, messages, tools):
        self.tool_schemas = tools
        return self.steps.pop(0)


def test_create_verified_dispatch_runs_all_mandatory_gates(tmp_path: Path):
    context = build_context(tmp_path)
    result = CreateVerifiedDispatchTool(context)(CreateVerifiedDispatchArgs())

    assert result.status == WorkflowStatus.PENDING_HUMAN_APPROVAL
    assert result.verification_passed
    assert result.schedule_id
    assert result.metrics is not None
    assert result.evidence is not None
    claims = {claim.claim_id: claim.value for claim in result.evidence.claims}
    assert claims["verification_status"] is True
    assert claims["executable_status"] is False
    assert claims["release_status"] == "pending_human_approval"
    assert result.simulation_only
    assert not result.executable

    assert result.artifacts is not None
    for path in result.artifacts.model_dump().values():
        assert Path(path).is_file()
    verification = json.loads(Path(result.artifacts.verification).read_text())
    assert verification["passed"] is True
    assert verification["executable"] is False

    events = [
        json.loads(line)
        for line in context.trace.path.read_text(encoding="utf-8").splitlines()
    ]
    event_types = [event["event_type"] for event in events]
    assert event_types == [
        "composite_dispatch_started",
        "composite_quality_gate_completed",
        "composite_request_validated",
        "composite_optimization_completed",
        "composite_verification_completed",
        "composite_evidence_built",
        "composite_artifacts_persisted",
        "composite_dispatch_completed",
    ]


def test_create_verified_dispatch_stops_at_failed_quality_gate(tmp_path: Path):
    context = build_context(tmp_path, load_status="fail")
    result = CreateVerifiedDispatchTool(context)(CreateVerifiedDispatchArgs())

    assert result.status == WorkflowStatus.SAFE_TERMINATED
    assert result.failure_code == "MEASURED_LOAD_QUALITY_NOT_PASSED"
    assert not result.verification_passed
    assert result.artifacts is None
    events = [
        json.loads(line)["event_type"]
        for line in context.trace.path.read_text(encoding="utf-8").splitlines()
    ]
    assert "composite_optimization_completed" not in events
    assert events[-1] == "composite_dispatch_terminated"


def test_carbon_dispatch_fails_closed_without_carbon_factors(tmp_path: Path):
    context = build_context(tmp_path)
    result = CreateVerifiedDispatchTool(context)(
        CreateVerifiedDispatchArgs(objective="carbon")
    )

    assert result.status == WorkflowStatus.SAFE_TERMINATED
    assert result.failure_code == "CARBON_FACTOR_REQUIRED"
    assert result.artifacts is None


def test_cost_capped_carbon_dispatch_uses_verified_economic_baseline(
    tmp_path: Path,
):
    context = build_context(tmp_path)
    points = [
        point.model_copy(
            update={"carbon_factor_kgco2e_per_kwh": 0.3 + index * 0.01}
        )
        for index, point in enumerate(context.inputs.points)
    ]
    carbon_inputs = context.inputs.model_copy(
        update={
            "points": points,
            "carbon_factor_version": "synthetic-test-carbon-v1",
            "carbon_factor_unit": "kgCO2e/kWh",
        }
    )
    context = VerifiedDispatchContext(
        inputs=carbon_inputs,
        quality_reports=context.quality_reports,
        measured_load=context.measured_load,
        scenario_config=context.scenario_config,
        trace=context.trace,
        artifact_root=context.artifact_root,
    )
    result = CreateVerifiedDispatchTool(context)(
        CreateVerifiedDispatchArgs(
            objective="carbon",
            max_cost_increase_pct=5,
        )
    )

    assert result.status == WorkflowStatus.PENDING_HUMAN_APPROVAL
    assert result.verification_passed
    assert result.evidence is not None
    claims = {claim.claim_id: claim.value for claim in result.evidence.claims}
    assert claims["max_cost_increase_pct"] == 5
    assert claims["max_total_cost_cny"] is not None
    assert "grid_emissions_kgco2e" in claims
    events = [
        json.loads(line)["event_type"]
        for line in context.trace.path.read_text(encoding="utf-8").splitlines()
    ]
    assert events.count("composite_optimization_completed") == 2
    assert events.count("composite_verification_completed") == 2


def test_verifier_failure_cannot_produce_artifacts(tmp_path: Path, monkeypatch):
    context = build_context(tmp_path)
    original_verify = business_tools.verify_dispatch

    def reject_dispatch(request, dispatch):
        report = original_verify(request, dispatch)
        return report.model_copy(update={"passed": False, "executable": False})

    monkeypatch.setattr(business_tools, "verify_dispatch", reject_dispatch)
    result = CreateVerifiedDispatchTool(context)(CreateVerifiedDispatchArgs())

    assert result.status == WorkflowStatus.SAFE_TERMINATED
    assert result.failure_code == "VERIFICATION_FAILED"
    assert result.artifacts is None
    assert not context.artifact_root.exists()


def test_infeasible_optimizer_result_safe_terminates_without_artifacts(
    tmp_path: Path, monkeypatch
):
    context = build_context(tmp_path)

    def infeasible(_optimizer, request):
        return DispatchResult(
            schedule_id="schedule_infeasible_test",
            request_id=request.request_id,
            dataset_version=request.dataset_version,
            status=DispatchStatus.INFEASIBLE,
            solver=SolverMetadata(
                solver_name="test",
                solver_status="infeasible",
                solve_time_ms=1,
            ),
            reason_codes=["OPTIMIZATION_INFEASIBLE"],
            evidence_id="optimizer_infeasible_test",
        )

    monkeypatch.setattr(
        business_tools.CvxpyPVStorageOptimizer,
        "solve",
        infeasible,
    )
    result = CreateVerifiedDispatchTool(context)(CreateVerifiedDispatchArgs())

    assert result.status == WorkflowStatus.SAFE_TERMINATED
    assert result.failure_code == "OPTIMIZATION_INFEASIBLE"
    assert result.artifacts is None
    events = [
        json.loads(line)["event_type"]
        for line in context.trace.path.read_text(encoding="utf-8").splitlines()
    ]
    assert "composite_verification_completed" in events


def test_tool_schema_rejects_unapproved_scenario_and_extra_fields():
    with pytest.raises(ValidationError):
        CreateVerifiedDispatchArgs.model_validate(
            {"scenario_id": "arbitrary", "shell_command": "rm -rf /"}
        )


def test_agent_loop_can_only_call_composite_dispatch_tool(tmp_path: Path):
    registry = ToolRegistry()
    register_verified_dispatch_tool(registry, build_context(tmp_path))
    transport = ScriptedTransport(
        [
            AgentModelStep(
                tool_calls=[
                    AgentToolCall(
                        call_id="dispatch-1",
                        name=CREATE_VERIFIED_DISPATCH,
                        arguments={},
                    )
                ],
                finish_reason="tool_calls",
            ),
            AgentModelStep(content="已生成并验证仿真计划，等待人工审批。"),
        ]
    )
    result = AgentLoop(
        transport=transport,
        registry=registry,
        policy=AgentPolicy(
            allowed_tools={CREATE_VERIFIED_DISPATCH},
            max_tool_calls_per_step=1,
            tool_timeout_seconds=120,
            max_tool_result_chars=50_000,
        ),
    ).run("生成经济调度", session_id="business-agent-test")

    assert result.status == "completed"
    assert result.steps == 2
    tool_messages = [item for item in result.messages if item["role"] == "tool"]
    payload = json.loads(tool_messages[0]["content"])
    assert payload["status"] == "pending_human_approval"
    assert payload["verification_passed"] is True
    assert set(registry.names) == {CREATE_VERIFIED_DISPATCH}
    assert transport.tool_schemas[0]["function"]["name"] == CREATE_VERIFIED_DISPATCH


def test_trace_writer_continues_existing_sequence(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    first = TraceWriter(path, "trace-run")
    first.append(event_type="first", component="test", status="pass")
    second = TraceWriter(path, "trace-run")
    second.append(event_type="second", component="test", status="pass")

    events = [json.loads(line) for line in path.read_text().splitlines()]
    assert [event["sequence"] for event in events] == [1, 2]


def test_full_registry_contains_exactly_five_energyops_tools(tmp_path: Path):
    registry = ToolRegistry()
    register_energyops_business_tools(registry, build_context(tmp_path))

    assert registry.names == {
        CREATE_VERIFIED_DISPATCH,
        GET_AVAILABLE_SCENARIOS,
        GET_DATA_QUALITY_REPORT,
        COMPARE_VERIFIED_PLANS,
        EXPLAIN_EVIDENCE,
    }


def test_scenario_and_quality_tools_expose_current_readiness(tmp_path: Path):
    context = build_context(tmp_path)
    tools = EnergyOpsBusinessQueryTools(context)

    scenarios = tools.get_available_scenarios(GetAvailableScenariosArgs())
    assert scenarios.status == "completed"
    assert scenarios.scenarios[0].available_objectives == [
        "cost",
        "economic",
    ]
    assert scenarios.scenarios[0].unavailable_objectives == {
        "carbon": "CARBON_FACTOR_REQUIRED",
        "balanced": "CARBON_FACTOR_REQUIRED",
    }

    quality = tools.get_data_quality_report(GetDataQualityReportArgs())
    assert quality.status == "completed"
    assert quality.economic_dispatch_ready
    assert not quality.carbon_dispatch_ready
    assert set(quality.reports) == {"load", "pv"}


def test_compare_and_explain_only_verified_dispatch_artifacts(tmp_path: Path):
    context = build_context(tmp_path)
    creator = CreateVerifiedDispatchTool(context)
    first = creator(CreateVerifiedDispatchArgs(objective="cost"))
    second = creator(CreateVerifiedDispatchArgs(objective="economic"))
    assert first.verification_passed and second.verification_passed

    tools = EnergyOpsBusinessQueryTools(context)
    comparison = tools.compare_verified_plans(
        CompareVerifiedPlansArgs(
            dispatch_run_ids=[first.run_id, second.run_id],
            baseline_run_id=first.run_id,
        )
    )
    assert comparison.status == "pending_human_approval"
    assert len(comparison.plans) == 2
    assert all(plan.verification_passed for plan in comparison.plans)
    assert comparison.plans[0].cost_change_vs_baseline_cny == pytest.approx(0)
    assert comparison.plans[0].emissions_change_vs_baseline_kgco2e is None

    explanation = tools.explain_evidence(
        ExplainEvidenceArgs(dispatch_run_id=first.run_id)
    )
    assert explanation.status == "completed"
    claim_ids = {claim.claim_id for claim in explanation.claims}
    assert {"verification_status", "executable_status", "total_cost_cny"} <= claim_ids
    assert "grid_emissions_status" in claim_ids
    assert any("不可作为设备控制指令" in warning for warning in explanation.warnings)


def test_query_tools_reject_path_traversal_and_unknown_claims(tmp_path: Path):
    context = build_context(tmp_path)
    creator = CreateVerifiedDispatchTool(context)
    created = creator(CreateVerifiedDispatchArgs())
    tools = EnergyOpsBusinessQueryTools(context)

    traversal = tools.explain_evidence(
        ExplainEvidenceArgs(dispatch_run_id="../outside")
    )
    assert traversal.status == "safe_terminated"
    assert traversal.failure_code == "INVALID_DISPATCH_RUN_ID"

    unknown = tools.explain_evidence(
        ExplainEvidenceArgs(
            dispatch_run_id=created.run_id,
            claim_ids=["invented_number"],
        )
    )
    assert unknown.status == "safe_terminated"
    assert unknown.failure_code == "EVIDENCE_CLAIM_NOT_FOUND"


def test_tampered_schedule_is_rejected_during_reverification(tmp_path: Path):
    context = build_context(tmp_path)
    created = CreateVerifiedDispatchTool(context)(CreateVerifiedDispatchArgs())
    assert created.artifacts is not None
    schedule_path = Path(created.artifacts.schedule)
    schedule = json.loads(schedule_path.read_text())
    schedule["points"][0]["grid_import_kw"] += 1
    schedule_path.write_text(json.dumps(schedule), encoding="utf-8")

    result = EnergyOpsBusinessQueryTools(context).explain_evidence(
        ExplainEvidenceArgs(dispatch_run_id=created.run_id)
    )
    assert result.status == "safe_terminated"
    assert result.failure_code == "DISPATCH_REVERIFICATION_FAILED"
