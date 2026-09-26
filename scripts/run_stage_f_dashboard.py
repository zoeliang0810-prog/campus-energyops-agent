"""Serve the Stage F dashboard and loopback-only Agent API."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energyops.agent_http import EnergyOpsAgentService, serve_dashboard  # noqa: E402


def load_local_deepseek_env(path: Path) -> None:
    """Load only approved DeepSeek settings from a local, git-ignored .env file."""
    if not path.is_file():
        return
    allowed = {"DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in allowed or key in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if value:
            os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost", "::1"])
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--model", default="deepseek-flash")
    parser.add_argument("--solver")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path.home() / ".config" / "energyops" / "secrets.env",
        help="local secrets file; only DeepSeek settings are loaded",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config/scenarios/hkust_gz_campus_baseline.json",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "dashboard_agent",
    )
    args = parser.parse_args()
    load_local_deepseek_env(args.env_file.expanduser())
    load_local_deepseek_env(PROJECT_ROOT / ".env")
    service = EnergyOpsAgentService(
        project_root=PROJECT_ROOT,
        config_path=args.config,
        state_dir=args.state_dir,
        model=args.model,
        solver_name=args.solver,
    )
    serve_dashboard(
        service=service,
        dist_dir=PROJECT_ROOT / "apps" / "energyops-dashboard" / "dist",
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
