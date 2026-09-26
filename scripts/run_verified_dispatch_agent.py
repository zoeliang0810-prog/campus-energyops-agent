"""Run the first real EnergyOps business tool through the self-owned AgentLoop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energyops.agent_app import build_energyops_agent_loop  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/scenarios/hkust_gz_campus_baseline.json",
    )
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--solver")
    parser.add_argument("--session-id", default="energyops-stage-d-dispatch")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "verified_dispatch_agent",
    )
    parser.add_argument(
        "request",
        nargs="?",
        default=(
            "请调用 create_verified_dispatch，为港科广基准场景生成经济目标的24小时"
            "仿真调度。只根据工具返回的 Evidence 解释结果，明确它不可执行。"
        ),
    )
    args = parser.parse_args()

    loop = build_energyops_agent_loop(
        project_root=PROJECT_ROOT,
        config_path=args.config,
        state_dir=args.state_dir,
        session_id=args.session_id,
        model=args.model,
        solver_name=args.solver,
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
