"""Run the verified campus PV-storage demo and persist all evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from energyops.campus_demo import build_demo_inputs, build_demo_request  # noqa: E402
from energyops.optimization.cvxpy_pv_storage import (  # noqa: E402
    CvxpyPVStorageOptimizer,
)
from energyops.pipeline import run_verified_dispatch  # noqa: E402
from energyops.trace import TraceWriter  # noqa: E402


def write_model(path: Path, model) -> None:
    path.write_text(
        json.dumps(model.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pv",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "fixtures"
        / "jiujiang_campus_pv_2026-02-19_15min.csv",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "fixtures" / "jiujiang_campus_pv_manifest.json",
    )
    parser.add_argument(
        "--load",
        type=Path,
        default=None,
        help="Optional measured 15-minute load CSV with timestamp and load_power_kw.",
    )
    parser.add_argument(
        "--load-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "fixtures" / "jiujiang_campus_load_manifest.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "jiujiang_campus_demo",
    )
    parser.add_argument("--solver", default=None)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    dataset_version = manifest["dataset_version"]
    if args.load is not None:
        if not args.load_manifest.exists():
            raise FileNotFoundError(
                f"measured load requires its manifest: {args.load_manifest}"
            )
        load_manifest = json.loads(args.load_manifest.read_text(encoding="utf-8"))
        dataset_version = (
            f"pv={manifest['dataset_version']};load={load_manifest['dataset_version']}"
        )
    inputs = build_demo_inputs(
        args.pv,
        dataset_version=dataset_version,
        load_csv=args.load,
    )
    request = build_demo_request(inputs, measured_load=args.load is not None)
    optimizer = CvxpyPVStorageOptimizer(inputs, solver_name=args.solver)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trace = TraceWriter(args.output_dir / "trace.jsonl", run_id="run_jiujiang_campus_demo")
    result, verification, evidence = run_verified_dispatch(
        run_id="run_jiujiang_campus_demo",
        request=request,
        optimizer=optimizer,
        trace=trace,
    )
    write_model(args.output_dir / "request.json", request)
    write_model(args.output_dir / "schedule.json", result)
    write_model(args.output_dir / "verification.json", verification)
    write_model(args.output_dir / "evidence.json", evidence)

    print(f"dispatch status: {result.status.value}")
    print(f"verification passed: {verification.passed}")
    print(f"executable: {verification.executable}")
    if result.reason_codes:
        print("reason codes: " + ", ".join(result.reason_codes))
    if verification.recomputed_metrics:
        print(
            json.dumps(
                verification.recomputed_metrics.model_dump(),
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
