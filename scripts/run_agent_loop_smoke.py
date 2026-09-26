"""Run a minimal real-API smoke test of the self-owned AgentLoop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energyops.agent_runtime import (  # noqa: E402
    AgentLoop,
    AgentPolicy,
    JsonAgentSessionStore,
    ToolDefinition,
    ToolRegistry,
)
from energyops.providers import DeepSeekAPIProvider  # noqa: E402
from energyops.trace import TraceWriter  # noqa: E402


class RuntimeStatusArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")


def get_runtime_status(_args: RuntimeStatusArgs) -> dict[str, object]:
    return {
        "agent_runtime": "energyops-self-owned",
        "agent_loop_available": True,
        "energyops_business_tools_registered": False,
        "stage": "C-runtime-smoke",
        "simulation_only": True,
        "executable": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--session-id", default="energyops-stage-c-smoke")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "agent_runtime_smoke",
    )
    parser.add_argument(
        "request",
        nargs="?",
        default=(
            "请先调用 get_agent_runtime_status，然后只根据工具结果说明当前 Agent "
            "运行层状态。不要编造调度结果。"
        ),
    )
    args = parser.parse_args()

    args.state_dir.mkdir(parents=True, exist_ok=True)
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="get_agent_runtime_status",
            description=(
                "Return the implementation status of the self-owned EnergyOps AgentLoop. "
                "Call this before answering runtime-status questions."
            ),
            args_model=RuntimeStatusArgs,
            handler=get_runtime_status,
        )
    )
    trace = TraceWriter(args.state_dir / "trace.jsonl", args.session_id)
    loop = AgentLoop(
        transport=DeepSeekAPIProvider(model=args.model),
        registry=registry,
        policy=AgentPolicy(
            max_steps=4,
            max_tool_calls_per_step=1,
            allowed_tools={"get_agent_runtime_status"},
        ),
        system_prompt=(
            "You are testing the self-owned EnergyOps AgentLoop. For every user request, "
            "you must call get_agent_runtime_status exactly once before answering. Use only "
            "the tool result. Never claim business dispatch tools are registered. Return a "
            "brief Chinese final answer."
        ),
        trace=trace,
        session_store=JsonAgentSessionStore(args.state_dir / "sessions"),
    )
    result = loop.run(args.request, session_id=args.session_id)
    result_path = args.state_dir / "result.json"
    result_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    print(f"status: {result.status}")
    print(f"steps: {result.steps}")
    if result.failure_code:
        print(f"failure_code: {result.failure_code}")
    if result.final_response:
        print(result.final_response)
    print(f"result: {result_path}")
    print(
        json.dumps(
            {"simulation_only": result.simulation_only, "executable": result.executable},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
