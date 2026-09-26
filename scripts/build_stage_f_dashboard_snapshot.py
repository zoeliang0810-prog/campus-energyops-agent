"""Build the reviewed, simulation-only snapshot for the Stage F dashboard."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGENT_OUTPUT = PROJECT_ROOT / "outputs" / "verified_dispatch_agent"
DEFAULT_DASHBOARD_AGENT_OUTPUT = PROJECT_ROOT / "outputs" / "dashboard_agent"
DEFAULT_PUBLIC_AGENT_OUTPUT = (
    PROJECT_ROOT / "data" / "fixtures" / "public_verified_dispatch_agent"
)
DEFAULT_SNAPSHOT = (
    PROJECT_ROOT / "data" / "reviewed" / "stage_f_dashboard_snapshot.json"
)
DEFAULT_DASHBOARD_DATA = (
    PROJECT_ROOT / "apps" / "energyops-dashboard" / "src" / "data.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one JSON object")
    return value


def _latest_create_payload(result: dict[str, Any]) -> dict[str, Any]:
    for message in reversed(result.get("messages", [])):
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message.get("content", ""))
        except json.JSONDecodeError:
            continue
        if (
            isinstance(payload, dict)
            and payload.get("tool_name") == "create_verified_dispatch"
            and payload.get("status") == "pending_human_approval"
        ):
            return payload
    raise ValueError("NO_VERIFIED_DISPATCH_TOOL_RESULT")


def _source(
    *,
    label: str,
    files: list[Path],
    component_ids: list[str],
    definition: str,
    caveats: list[str],
    generated_at: str,
    methods: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    relative_files = [str(path.resolve().relative_to(PROJECT_ROOT)) for path in files]
    return {
        "label": label,
        "files": relative_files,
        "executedAt": generated_at,
        "caveats": caveats,
        "evidenceFlow": [
            {
                "title": "Local verified artifact",
                "detail": (
                    "Read from the simulation-only EnergyOps output directory; no live "
                    "database or device endpoint was queried."
                ),
            }
        ],
        "metricDefinitions": [
            {
                "label": label,
                "definition": definition,
                "componentIds": component_ids,
                "sourceLineage": [{"files": relative_files}],
            }
        ],
        "methods": methods or [],
    }


def build_snapshot(agent_output: Path | None = None) -> dict[str, Any]:
    if agent_output is None:
        if (DEFAULT_DASHBOARD_AGENT_OUTPUT / "result.json").is_file():
            agent_output = DEFAULT_DASHBOARD_AGENT_OUTPUT
        elif (DEFAULT_AGENT_OUTPUT / "result.json").is_file():
            agent_output = DEFAULT_AGENT_OUTPUT
        else:
            agent_output = DEFAULT_PUBLIC_AGENT_OUTPUT
    result_path = agent_output / "result.json"
    trace_path = agent_output / "trace.jsonl"
    agent_result = _read_json(result_path)
    tool_payload = _latest_create_payload(agent_result)
    artifact_paths = {}
    for name, value in tool_payload["artifacts"].items():
        path = Path(value)
        artifact_paths[name] = path if path.is_absolute() else agent_output / path
    request = _read_json(artifact_paths["request"])
    schedule = _read_json(artifact_paths["schedule"])
    verification = _read_json(artifact_paths["verification"])
    evidence = _read_json(artifact_paths["evidence"])
    if not verification.get("passed") or verification.get("executable"):
        raise ValueError("DASHBOARD_REQUIRES_VERIFIED_NON_EXECUTABLE_PLAN")
    if not (
        request["request_id"] == schedule["request_id"]
        and request["dataset_version"] == schedule["dataset_version"]
        and schedule["schedule_id"]
        == verification["schedule_id"]
        == evidence["schedule_id"]
    ):
        raise ValueError("DASHBOARD_ARTIFACT_BINDING_FAILED")

    generated_at = datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()
    metrics = verification["recomputed_metrics"]
    claims = {claim["claim_id"]: claim for claim in evidence["claims"]}
    quality_reports = {
        "load": _read_json(
            PROJECT_ROOT / "data/fixtures/hkust_gz_selected_meter_load_quality.json"
        ),
        "pv": _read_json(
            PROJECT_ROOT / "data/fixtures/hkust_gz_weather_pv_quality.json"
        ),
    }
    summary_rows = [
        {
            "dispatchRunId": tool_payload["run_id"],
            "scheduleId": schedule["schedule_id"],
            "scenarioName": request["scenario_name"],
            "objective": request["objective_mode"],
            "studyDay": schedule["points"][0]["timestamp"][:10],
            "datasetVersion": request["dataset_version"],
            "solver": schedule["solver"]["solver_name"],
            "solverStatus": schedule["solver"]["solver_status"],
            "verificationPassed": verification["passed"],
            "releaseStatus": claims["release_status"]["value"],
            "simulationOnly": request["simulation_only"],
            "executable": request["executable"],
            "totalCostCny": metrics["total_cost_cny"],
            "peakImportKw": metrics["peak_import_kw"],
            "loadEnergyKwh": metrics["load_energy_kwh"],
            "pvEnergyKwh": metrics["pv_energy_kwh"],
            "pvCurtailedEnergyKwh": metrics["pv_curtailed_energy_kwh"],
            "pvSelfConsumptionRatio": metrics["pv_self_consumption_ratio"],
            "gridImportEnergyKwh": metrics["grid_import_energy_kwh"],
            "batteryThroughputKwh": metrics["battery_throughput_kwh"],
            "carbonStatus": claims.get("grid_emissions_status", {}).get(
                "value", "available"
            ),
            "tariffVerified": request["tariff_verified"],
            "loadQualityStatus": quality_reports["load"]["status"],
            "pvQualityStatus": quality_reports["pv"]["status"],
        }
    ]
    schedule_rows = []
    for index, point in enumerate(schedule["points"]):
        schedule_rows.append(
            {
                "slot": index + 1,
                "timestamp": point["timestamp"],
                "hour": point["timestamp"][11:16],
                "plannedLoadKw": point["load_power_kw"],
                "plannedPvKw": point["pv_power_kw"],
                "plannedGridImportKw": point["grid_import_kw"],
                "plannedBatteryChargeKw": point["battery_charge_kw"],
                "plannedBatteryDischargeKw": point["battery_discharge_kw"],
                "plannedBatteryEnergyKwh": point["battery_energy_after_kwh"],
                "plannedPvCurtailmentKw": point["pv_curtailment_kw"],
                "actualLoadKw": None,
                "actualPvKw": None,
                "actualGridImportKw": None,
                "actualBatteryEnergyKwh": None,
                "loadDeviationPct": None,
                "pvDeviationPct": None,
                "socDeviationKwh": None,
                "monitoringStatus": "awaiting_actual_telemetry",
            }
        )

    quality_rows: list[dict[str, Any]] = []
    for dataset, report in quality_reports.items():
        findings = report.get("findings") or ["No detailed findings supplied."]
        for index, finding in enumerate(findings, start=1):
            quality_rows.append(
                {
                    "dataset": dataset,
                    "status": report.get("status", "unknown"),
                    "findingId": f"{dataset}-{index}",
                    "finding": str(finding),
                    "evidenceId": report.get("evidence_id"),
                }
            )

    trace_rows: list[dict[str, Any]] = []
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("run_id") != agent_result.get("session_id"):
                continue
            trace_rows.append(
                {
                    "sequence": event["sequence"],
                    "timestamp": event["timestamp"],
                    "eventType": event["event_type"],
                    "component": event["component"],
                    "status": event["status"],
                    "evidenceId": event.get("evidence_id"),
                }
            )

    monitoring_rules = [
        {
            "ruleId": "load-deviation",
            "signal": "负荷功率",
            "threshold": "±10%",
            "persistence": "连续2个时间片",
            "action": "建议重新计算调度，不自动执行",
            "status": "draft_policy",
        },
        {
            "ruleId": "pv-deviation",
            "signal": "光伏功率",
            "threshold": "±15%",
            "persistence": "连续2个时间片",
            "action": "建议刷新光伏预测并重新调度",
            "status": "draft_policy",
        },
        {
            "ruleId": "soc-deviation",
            "signal": "储能电量",
            "threshold": "±20 kWh",
            "persistence": "任一时间片",
            "action": "建议核对 BMS 状态并重新调度",
            "status": "draft_policy",
        },
        {
            "ruleId": "data-gap",
            "signal": "实际数据覆盖率",
            "threshold": "低于90%",
            "persistence": "当前监测窗口",
            "action": "暂停重新调度并报告数据缺失",
            "status": "draft_policy",
        },
    ]
    tools = [
        ("get_available_scenarios", "查询可用场景和目标"),
        ("get_data_quality_report", "检查负荷与光伏数据质量"),
        ("create_verified_dispatch", "生成并独立验证仿真调度"),
        ("compare_verified_plans", "重新验证后比较多个方案"),
        ("explain_evidence", "读取带证据路径的结论"),
    ]

    artifact_files = list(artifact_paths.values())
    fixed_caveats = [
        "调度计划为 simulation_only，executable=false。",
        "负荷来自末端分表聚合，不是校园总表。",
        "300 kWp 光伏、150 kW / 400 kWh 储能和参考电价均未完成现场核验。",
        "实际负荷、实际光伏和实际 SOC 尚未接入；缺失值保留为 null。",
    ]
    return {
        "id": "energyops-stage-f-dashboard",
        "surface": "dashboard",
        "title": "港科广 EnergyOps 决策台",
        "generatedAt": generated_at,
        "status": "reviewed_simulation",
        "buildStatus": "creating",
        "queries": {
            "dispatch_summary": {
                "rows": summary_rows,
                "source": _source(
                    label="已验证调度摘要",
                    files=artifact_files,
                    component_ids=[
                        "plan-status",
                        "total-cost",
                        "peak-import",
                        "pv-utilization",
                        "battery-throughput",
                    ],
                    definition=(
                        "由独立 Verifier 复算的单日仿真调度汇总；费用为未核验参考电价下的模拟值。"
                    ),
                    caveats=fixed_caveats,
                    generated_at=generated_at,
                ),
            },
            "dispatch_schedule": {
                "rows": schedule_rows,
                "source": _source(
                    label="逐时计划与实际值接入状态",
                    files=[artifact_paths["schedule"], artifact_paths["verification"]],
                    component_ids=[
                        "power-plan-chart",
                        "storage-plan-chart",
                        "hourly-plan",
                        "actual-coverage",
                        "deviation-status",
                        "plan-actual-chart",
                        "plan-actual-table",
                    ],
                    definition=(
                        "24 个小时点的已验证计划；actual* 与 deviation* 字段在实际遥测接入前保持 null。"
                    ),
                    caveats=fixed_caveats,
                    generated_at=generated_at,
                    methods=[
                        {
                            "language": "python",
                            "code": (
                                "Copy verified schedule points; initialize unavailable actual "
                                "and deviation fields to null rather than zero."
                            ),
                        }
                    ],
                ),
            },
            "quality_findings": {
                "rows": quality_rows,
                "source": _source(
                    label="负荷与光伏数据质量结论",
                    files=[
                        PROJECT_ROOT
                        / "data/fixtures/hkust_gz_selected_meter_load_quality.json",
                        PROJECT_ROOT
                        / "data/fixtures/hkust_gz_weather_pv_quality.json",
                    ],
                    component_ids=["quality-findings"],
                    definition="输入数据质量报告中的状态与原始 findings。",
                    caveats=fixed_caveats,
                    generated_at=generated_at,
                ),
            },
            "monitoring_rules": {
                "rows": monitoring_rules,
                "source": {
                    "label": "偏差监测草案策略",
                    "assumptions": [
                        "阈值是阶段 F 骨架的可修改默认值，尚未由校园运维或设备供应商批准。"
                    ],
                    "metricDefinitions": [
                        {
                            "label": "重新调度触发规则",
                            "definition": (
                                "仅产生重新调度建议；不会向 PLC、BMS、EMS 或储能设备发送控制指令。"
                            ),
                            "componentIds": [
                                "reschedule-status",
                                "reschedule-rules",
                                "reschedule-decision",
                            ],
                        }
                    ],
                },
            },
            "agent_tools": {
                "rows": [
                    {"tool": name, "purpose": purpose, "access": "whitelist"}
                    for name, purpose in tools
                ],
                "source": {
                    "label": "EnergyOps Agent 白名单",
                    "files": [
                        "src/energyops/business_tools.py",
                        "src/energyops/business_query_tools.py",
                    ],
                    "metricDefinitions": [
                        {
                            "label": "可用业务工具",
                            "definition": "AgentLoop 当前注册的五个受限 EnergyOps 工具。",
                            "componentIds": ["whitelist-tools"],
                            "sourceLineage": [
                                {
                                    "files": [
                                        "src/energyops/business_tools.py",
                                        "src/energyops/business_query_tools.py",
                                    ]
                                }
                            ],
                        }
                    ],
                },
            },
            "agent_trace": {
                "rows": trace_rows,
                "source": _source(
                    label="阶段 D 真实多工具验收 Trace",
                    files=[trace_path],
                    component_ids=["agent-trace"],
                    definition="真实 DeepSeek API 多工具 turn 的脱敏有序事件。",
                    caveats=[
                        "Trace 不记录 API Key、Authorization Header、完整环境变量或隐藏推理。"
                    ],
                    generated_at=generated_at,
                ),
            },
        },
    }


def sync_dashboard_data(
    snapshot: dict[str, Any], dashboard_data_path: Path = DEFAULT_DASHBOARD_DATA
) -> None:
    """Refresh reviewed dashboard data without changing its stable app identity."""
    dashboard = _read_json(dashboard_data_path)
    dashboard.update(
        {
            "surface": "dashboard",
            "title": snapshot["title"],
            "generatedAt": snapshot["generatedAt"],
            "status": snapshot["status"],
            "queries": snapshot["queries"],
        }
    )
    dashboard_data_path.write_text(
        json.dumps(dashboard, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the reviewed Stage F dashboard snapshot."
    )
    parser.add_argument(
        "--sync-dashboard",
        action="store_true",
        help="also refresh the dashboard's reviewed rows while preserving its app id",
    )
    args = parser.parse_args()
    snapshot = build_snapshot()
    DEFAULT_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_SNAPSHOT.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.sync_dashboard:
        sync_dashboard_data(snapshot)
        print(DEFAULT_DASHBOARD_DATA)
    print(DEFAULT_SNAPSHOT)


if __name__ == "__main__":
    main()
