"""Generate portable, synthetic Agent artifacts for the public dashboard demo."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energyops.business_tools import (  # noqa: E402
    CreateVerifiedDispatchArgs,
    CreateVerifiedDispatchTool,
    VerifiedDispatchContext,
)
from energyops.scenario import build_hourly_inputs  # noqa: E402
from energyops.scenario_config import load_scenario_config  # noqa: E402
from energyops.trace import TraceWriter  # noqa: E402


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/scenarios/hkust_gz_campus_baseline.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            PROJECT_ROOT / "data/fixtures/public_verified_dispatch_agent"
        ),
    )
    args = parser.parse_args()

    config = load_scenario_config(args.config.resolve())
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def project_path(value: str) -> Path:
        return PROJECT_ROOT / value

    pv_manifest = read_json(project_path(config.pv.manifest))
    load_manifest = read_json(project_path(config.load.manifest))
    quality_reports = {
        "pv": read_json(project_path(config.pv.quality)),
        "load": read_json(project_path(config.load.quality)),
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
    trace = TraceWriter(output_dir / "trace.jsonl", "public-demo")
    tool_result = CreateVerifiedDispatchTool(
        VerifiedDispatchContext(
            inputs=inputs,
            quality_reports=quality_reports,
            measured_load=False,
            scenario_config=config,
            trace=trace,
            artifact_root=output_dir / "dispatches",
        )
    )(CreateVerifiedDispatchArgs())
    if not tool_result.verification_passed or tool_result.artifacts is None:
        raise RuntimeError(tool_result.failure_code or "public demo generation failed")

    payload = tool_result.model_dump(mode="json")
    payload["artifacts"] = {
        name: str(Path(path).resolve().relative_to(output_dir))
        for name, path in payload["artifacts"].items()
    }
    result = {
        "status": "completed",
        "steps": 2,
        "final_response": (
            "公开版合成数据已生成验证通过的仿真调度；结果不可执行，"
            "仍需人工审批。"
        ),
        "failure_code": None,
        "simulation_only": True,
        "executable": False,
        "messages": [
            {
                "role": "tool",
                "name": "create_verified_dispatch",
                "content": json.dumps(payload, ensure_ascii=False),
            }
        ],
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(output_dir / "result.json")


if __name__ == "__main__":
    main()
