"""Build a bounded evidence pack from independently verified results."""

from __future__ import annotations

from typing import Any

from .contracts import (
    DispatchRequest,
    DispatchResult,
    EvidenceClaim,
    EvidencePack,
    VerificationReport,
)


def build_evidence_pack(
    run_id: str,
    report: VerificationReport,
    *,
    request: DispatchRequest | None = None,
    result: DispatchResult | None = None,
    data_quality: dict[str, Any] | None = None,
) -> EvidencePack:
    claims: list[EvidenceClaim] = [
        EvidenceClaim(
            claim_id="verification_status",
            claim_type="verifier_fact",
            value=report.passed,
            evidence_id=report.evidence_id,
            path="passed",
        ),
        EvidenceClaim(
            claim_id="executable_status",
            claim_type="policy_fact",
            value=report.executable,
            evidence_id=report.evidence_id,
            path="executable",
        ),
    ]
    if request is not None:
        request_claims = {
            "scenario_name": request.scenario_name,
            "step_minutes": request.step_minutes,
            "horizon_steps": request.horizon_steps,
            "dataset_version": request.dataset_version,
            "objective_mode": request.objective_mode.value,
            "allow_grid_charging": request.allow_grid_charging,
            "allow_grid_export": request.allow_grid_export,
            "tariff_version": request.tariff_version,
            "tariff_verified": request.tariff_verified,
            "carbon_factor_version": request.carbon_factor_version,
            "max_total_cost_cny": request.max_total_cost_cny,
            "max_cost_increase_pct": request.max_cost_increase_pct,
            "simulation_only": request.simulation_only,
            "critical_load_locked": request.critical_load_locked,
        }
        for field_name, value in request_claims.items():
            claims.append(
                EvidenceClaim(
                    claim_id=field_name,
                    claim_type="validated_request_fact",
                    value=value,
                    evidence_id=f"request:{request.request_id}",
                    path=field_name,
                )
            )
    if result is not None:
        claims.extend(
            [
                EvidenceClaim(
                    claim_id="solver_name",
                    claim_type="solver_fact",
                    value=result.solver.solver_name,
                    evidence_id=result.evidence_id,
                    path="solver.solver_name",
                ),
                EvidenceClaim(
                    claim_id="solver_status",
                    claim_type="solver_fact",
                    value=result.solver.solver_status,
                    evidence_id=result.evidence_id,
                    path="solver.solver_status",
                ),
            ]
        )
    for dataset_name, quality in sorted((data_quality or {}).items()):
        claims.append(
            EvidenceClaim(
                claim_id=f"{dataset_name}_quality_status",
                claim_type="data_quality_fact",
                value=quality.get("status", "unknown"),
                evidence_id=str(quality.get("evidence_id", f"quality:{dataset_name}")),
                path="status",
            )
        )
        for detail_name in ("findings", "warnings"):
            detail = quality.get(detail_name)
            if detail:
                claims.append(
                    EvidenceClaim(
                        claim_id=f"{dataset_name}_quality_{detail_name}",
                        claim_type="data_quality_fact",
                        value=detail,
                        evidence_id=str(
                            quality.get("evidence_id", f"quality:{dataset_name}")
                        ),
                        path=detail_name,
                    )
                )
    metrics = report.recomputed_metrics
    if metrics is not None:
        claim_fields = {
            "total_cost_cny": "CNY",
            "peak_import_kw": "kW",
            "pv_energy_kwh": "kWh",
            "pv_curtailed_energy_kwh": "kWh",
            "grid_import_energy_kwh": "kWh",
            "grid_export_energy_kwh": "kWh",
            "load_energy_kwh": "kWh",
            "battery_charge_energy_kwh": "kWh",
            "battery_discharge_energy_kwh": "kWh",
            "baseline_peak_import_kw": "kW",
            "peak_reduction_kw": "kW",
            "pv_self_consumption_ratio": None,
            "battery_throughput_kwh": "kWh",
        }
        if request is not None and request.carbon_factor_version:
            claim_fields["grid_emissions_kgco2e"] = "kgCO2e"
        else:
            claims.append(
                EvidenceClaim(
                    claim_id="grid_emissions_status",
                    claim_type="data_availability_fact",
                    value="unavailable_missing_carbon_factor",
                    evidence_id=report.evidence_id,
                    path="recomputed_metrics.grid_emissions_kgco2e",
                )
            )
        for field_name, unit in claim_fields.items():
            claims.append(
                EvidenceClaim(
                    claim_id=field_name,
                    claim_type="recomputed_tool_fact",
                    value=getattr(metrics, field_name),
                    unit=unit,
                    evidence_id=report.evidence_id,
                    path=f"recomputed_metrics.{field_name}",
                )
            )
    claims.append(
        EvidenceClaim(
            claim_id="release_status",
            claim_type="policy_fact",
            value="pending_human_approval" if report.passed else "safe_terminated",
            evidence_id=report.evidence_id,
            path="release_status",
        )
    )
    return EvidencePack(run_id=run_id, schedule_id=report.schedule_id, claims=claims)
