"""Factory for the policy-bounded EnergyOps AgentLoop used by CLI and HTTP."""

from __future__ import annotations

import json
from pathlib import Path

from .agent_runtime import (
    AgentLoop,
    AgentModelTransport,
    AgentPolicy,
    JsonAgentSessionStore,
    ToolRegistry,
)
from .business_query_tools import register_energyops_business_tools
from .business_tools import VerifiedDispatchContext
from .providers import DeepSeekAPIProvider
from .scenario import build_hourly_inputs
from .scenario_config import load_scenario_config
from .trace import TraceWriter


ENERGYOPS_SYSTEM_PROMPT = (
    "You are the language and orchestration layer of the Campus EnergyOps "
    "simulation. Use only the five supplied EnergyOps business tools. Request at "
    "most four tool calls per step; they are executed sequentially. Before every "
    "final answer, call at least one relevant business tool in the current turn. "
    "For questions about actual-versus-planned deviations or input availability, "
    "call get_data_quality_report first. Actual operating load, PV and battery "
    "SOC telemetry are not connected in this demo; the public synthetic load is "
    "not execution feedback. Do not invent actual values or deviations. For a direct "
    "scheduling request, call create_verified_dispatch exactly once. Use scenario "
    "and quality tools when the user asks what is available or whether inputs are "
    "ready. Compare only through compare_verified_plans and ground explanations "
    "through explain_evidence. Never calculate energy values yourself. Always state "
    "that dispatch results are simulation-only, non-executable, and pending human "
    "approval. Do not claim that a failed tool call produced a schedule. Reply in "
    "concise Chinese."
)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def build_energyops_agent_loop(
    *,
    project_root: Path,
    config_path: Path,
    state_dir: Path,
    session_id: str,
    model: str = "deepseek-flash",
    solver_name: str | None = None,
    transport: AgentModelTransport | None = None,
) -> AgentLoop:
    """Build one request-scoped loop while retaining durable local sessions."""
    project_root = project_root.resolve()
    config = load_scenario_config(config_path.resolve())

    def project_path(relative: str) -> Path:
        return project_root / relative

    pv_manifest = _read_json(project_path(config.pv.manifest))
    load_manifest = _read_json(project_path(config.load.manifest))
    quality_reports = {
        "pv": _read_json(project_path(config.pv.quality)),
        "load": _read_json(project_path(config.load.quality)),
    }
    inputs = build_hourly_inputs(
        project_path(config.pv.csv),
        dataset_version=(
            f"pv={pv_manifest['dataset_version']};"
            f"load={load_manifest['dataset_version']}"
        ),
        load_csv=project_path(config.load.csv),
        buy_price_cny_per_kwh_by_hour=(
            config.tariff.buy_price_cny_per_kwh_by_hour
        ),
        sell_price_cny_per_kwh=config.tariff.sell_price_cny_per_kwh,
    )

    state_dir = state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    trace = TraceWriter(state_dir / "trace.jsonl", session_id)
    registry = ToolRegistry()
    measured_load = load_manifest.get("data_origin") == "measured"
    register_energyops_business_tools(
        registry,
        VerifiedDispatchContext(
            inputs=inputs,
            quality_reports=quality_reports,
            measured_load=measured_load,
            scenario_config=config,
            trace=trace,
            artifact_root=state_dir / "dispatches",
            solver_name=solver_name,
        ),
    )
    return AgentLoop(
        transport=transport or DeepSeekAPIProvider(model=model),
        registry=registry,
        policy=AgentPolicy(
            max_steps=8,
            max_tool_calls_per_step=4,
            tool_timeout_seconds=120,
            max_tool_result_chars=50_000,
            allowed_tools=registry.names,
        ),
        system_prompt=ENERGYOPS_SYSTEM_PROMPT,
        trace=trace,
        session_store=JsonAgentSessionStore(state_dir / "sessions"),
    )
