"""Refresh and build the Stage F dashboard with the installed Data plugin."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_ROOT = PROJECT_ROOT / "apps" / "energyops-dashboard"
DATA_PLUGIN_CACHE = (
    Path.home()
    / ".codex"
    / "plugins"
    / "cache"
    / "openai-curated-remote"
    / "data-analytics"
)


def _find_builder() -> Path:
    candidates = sorted(
        DATA_PLUGIN_CACHE.glob("*/scripts/data-app.mjs"), reverse=True
    )
    if not candidates:
        raise FileNotFoundError(
            "DATA_APP_BUILDER_NOT_FOUND: install or enable the Data plugin in Codex"
        )
    return candidates[0]


def main() -> None:
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("NODE_NOT_FOUND: the local Data App builder needs Node.js")
    subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_stage_f_dashboard_snapshot.py"),
            "--sync-dashboard",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        [
            node,
            str(_find_builder()),
            "build",
            "--project-dir",
            str(DASHBOARD_ROOT),
            "--separate-data",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )
    print(DASHBOARD_ROOT / "dist" / "index.html")


if __name__ == "__main__":
    main()
