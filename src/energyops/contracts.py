"""Strict contracts shared by the optimizer, harness, and verifiers."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QualityStatus(StrEnum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


class OperatingMode(StrEnum):
    SIMULATION_ONLY = "simulation_only"
    DECISION_SUPPORT = "decision_support"


class ObjectiveMode(StrEnum):
    COST = "cost"
    ECONOMIC = "economic"
    CARBON = "carbon"
    BALANCED = "balanced"
    PEAK_AND_PV_SELF_CONSUMPTION = "peak_and_pv_self_consumption"


class WorkflowStatus(StrEnum):
    CREATED = "created"
    PARSED = "parsed"
    VALIDATED = "validated"
    QUALITY_CHECKED = "quality_checked"
    OPTIMIZED = "optimized"
    VERIFIED = "verified"
    EXPLANATION_READY = "explanation_ready"
    PENDING_HUMAN_APPROVAL = "pending_human_approval"
    SAFE_TERMINATED = "safe_terminated"


class DispatchStatus(StrEnum):
    OPTIMAL = "optimal"
    INFEASIBLE = "infeasible"
    ERROR = "error"


class DatasetManifest(StrictModel):
    dataset_id: str
    dataset_version: str
    site_id: str
    source_sha256: str
    source_name: str
    source_timezone: str
    variable: str = "pv_power_kw"
    unit: str = "kW"
    source_resolution_minutes: int = Field(gt=0)
    dispatch_resolution_minutes: int = Field(gt=0)
    rows: int = Field(ge=0)
    start_time: datetime
    end_time: datetime
    transformations: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)


class DataQualityReport(StrictModel):
    dataset_version: str
    status: QualityStatus
    row_count: int = Field(ge=0)
    parsed_timestamp_count: int = Field(ge=0)
    duplicate_timestamp_count: int = Field(ge=0)
    missing_timestamp_count: int = Field(ge=0)
    null_power_count: int = Field(ge=0)
    negative_power_count: int = Field(ge=0)
    gap_count: int = Field(ge=0)
    maximum_gap_minutes: float = Field(ge=0)
    inferred_resolution_minutes: int | None = Field(default=None, gt=0)
    findings: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)
    evidence_id: str


class BatteryConfig(StrictModel):
    asset_id: str
    capacity_kwh: float = Field(gt=0)
    initial_energy_kwh: float = Field(ge=0)
    min_energy_kwh: float = Field(ge=0)
    max_energy_kwh: float = Field(gt=0)
    max_charge_kw: float = Field(gt=0)
    max_discharge_kw: float = Field(gt=0)
    charge_efficiency: float = Field(gt=0, le=1)
    discharge_efficiency: float = Field(gt=0, le=1)
    self_discharge_kwh_per_hour: float = Field(default=0, ge=0)
    degradation_cost_cny_per_kwh_throughput: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_energy_bounds(self) -> "BatteryConfig":
        if self.max_energy_kwh > self.capacity_kwh:
            raise ValueError("max_energy_kwh cannot exceed capacity_kwh")
        if self.min_energy_kwh > self.max_energy_kwh:
            raise ValueError("min_energy_kwh cannot exceed max_energy_kwh")
        if not self.min_energy_kwh <= self.initial_energy_kwh <= self.max_energy_kwh:
            raise ValueError("initial_energy_kwh must be inside configured bounds")
        return self


class DispatchRequest(StrictModel):
    request_id: str
    scenario_name: str = "unnamed_simulation"
    site_id: str
    dataset_version: str
    operating_mode: OperatingMode
    horizon_start: datetime
    horizon_steps: int = Field(gt=0)
    step_minutes: int = Field(gt=0)
    grid_import_limit_kw: float = Field(gt=0)
    grid_export_limit_kw: float = Field(ge=0)
    allow_grid_charging: bool = False
    allow_grid_export: bool = False
    tariff_region: str
    tariff_version: str
    tariff_verified: bool = False
    objective_mode: ObjectiveMode = ObjectiveMode.COST
    peak_penalty: float = Field(default=1.0, ge=0)
    curtailment_penalty: float = Field(default=0.1, ge=0)
    throughput_penalty: float = Field(default=0.001, ge=0)
    carbon_price_cny_per_kg: float = Field(default=0.0, ge=0)
    carbon_factor_version: str | None = None
    max_total_cost_cny: float | None = Field(default=None, ge=0)
    max_cost_increase_pct: float | None = Field(default=None, ge=0)
    battery: BatteryConfig
    required_terminal_energy_kwh: float | None = Field(default=None, ge=0)
    assumptions: list[str] = Field(default_factory=list)
    critical_load_locked: bool = True
    simulation_only: bool = True
    executable: bool = False

    @model_validator(mode="after")
    def validate_horizon_and_mode(self) -> "DispatchRequest":
        if self.horizon_start.tzinfo is None:
            raise ValueError("horizon_start must include timezone information")
        if 60 % self.step_minutes != 0:
            raise ValueError("step_minutes must divide one hour exactly")
        if self.required_terminal_energy_kwh is not None:
            if self.required_terminal_energy_kwh > self.battery.max_energy_kwh:
                raise ValueError("terminal energy cannot exceed battery maximum")
        if self.operating_mode == OperatingMode.DECISION_SUPPORT and self.assumptions:
            raise ValueError("decision_support requests cannot contain unresolved assumptions")
        if self.simulation_only and self.executable:
            raise ValueError("simulation-only requests cannot be executable")
        if not self.critical_load_locked:
            raise ValueError("critical loads must remain locked")
        if self.max_cost_increase_pct is not None and self.max_total_cost_cny is None:
            raise ValueError(
                "max_cost_increase_pct requires a derived max_total_cost_cny"
            )
        return self


class DispatchSeriesPoint(StrictModel):
    timestamp: datetime
    pv_power_kw: float = Field(ge=0)
    load_power_kw: float = Field(ge=0)
    buy_price_cny_per_kwh: float = Field(ge=0)
    sell_price_cny_per_kwh: float = Field(ge=0)
    carbon_factor_kgco2e_per_kwh: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_timestamp(self) -> "DispatchSeriesPoint":
        if self.timestamp.tzinfo is None:
            raise ValueError("input-series timestamps must include timezone information")
        return self


class DispatchInputSeries(StrictModel):
    dataset_version: str
    points: list[DispatchSeriesPoint]
    source: str = "unspecified"
    generated_at: datetime | None = None
    timezone: str = "Asia/Shanghai"
    power_unit: str = "kW"
    carbon_factor_unit: str | None = None
    forecast_model_version: str | None = None
    forecast_confidence: float | None = Field(default=None, ge=0, le=1)
    carbon_factor_version: str | None = None
    verified: bool = False
    simulation_only: bool = True
    assumptions: list[str] = Field(default_factory=list)


class SchedulePoint(StrictModel):
    timestamp: datetime
    pv_power_kw: float = Field(ge=0)
    pv_curtailment_kw: float = Field(default=0, ge=0)
    load_power_kw: float = Field(ge=0)
    battery_charge_kw: float = Field(ge=0)
    battery_discharge_kw: float = Field(ge=0)
    grid_import_kw: float = Field(ge=0)
    grid_export_kw: float = Field(ge=0)
    battery_energy_before_kwh: float = Field(ge=0)
    battery_energy_after_kwh: float = Field(ge=0)
    buy_price_cny_per_kwh: float = Field(ge=0)
    sell_price_cny_per_kwh: float = Field(ge=0)
    carbon_factor_kgco2e_per_kwh: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_timestamp(self) -> "SchedulePoint":
        if self.timestamp.tzinfo is None:
            raise ValueError("schedule timestamps must include timezone information")
        return self


class DispatchMetrics(StrictModel):
    pv_energy_kwh: float
    pv_curtailed_energy_kwh: float = 0
    load_energy_kwh: float
    battery_charge_energy_kwh: float
    battery_discharge_energy_kwh: float
    grid_import_energy_kwh: float
    grid_export_energy_kwh: float
    electricity_cost_cny: float
    degradation_cost_cny: float
    total_cost_cny: float
    peak_import_kw: float
    baseline_peak_import_kw: float = 0
    peak_reduction_kw: float = 0
    pv_self_consumption_ratio: float = 0
    grid_emissions_kgco2e: float = 0
    battery_throughput_kwh: float = 0


class SolverMetadata(StrictModel):
    solver_name: str
    solver_status: str
    solve_time_ms: float = Field(ge=0)
    objective_value: float | None = None
    mip_gap: float | None = Field(default=None, ge=0)


class DispatchResult(StrictModel):
    schedule_id: str
    request_id: str
    dataset_version: str
    status: DispatchStatus
    points: list[SchedulePoint] = Field(default_factory=list)
    metrics: DispatchMetrics | None = None
    solver: SolverMetadata
    reason_codes: list[str] = Field(default_factory=list)
    evidence_id: str

    @model_validator(mode="after")
    def validate_status_payload(self) -> "DispatchResult":
        if self.status == DispatchStatus.OPTIMAL:
            if not self.points or self.metrics is None:
                raise ValueError("optimal results require points and metrics")
        elif self.points:
            raise ValueError("non-optimal results cannot expose executable schedule points")
        return self


class ConstraintViolation(StrictModel):
    constraint_id: str
    message: str
    timestamp: datetime | None = None
    actual: float | str | None = None
    expected: float | str | None = None
    unit: str | None = None


class VerificationReport(StrictModel):
    schedule_id: str
    passed: bool
    executable: bool
    violations: list[ConstraintViolation]
    recomputed_metrics: DispatchMetrics | None
    evidence_id: str


class EvidenceClaim(StrictModel):
    claim_id: str
    claim_type: str
    value: Any
    unit: str | None = None
    evidence_id: str
    path: str


class EvidencePack(StrictModel):
    run_id: str
    schedule_id: str
    claims: list[EvidenceClaim]


class ParsedCampusRequest(StrictModel):
    """Small, validated request that a language-model provider may produce."""

    intent: str = "create_schedule"
    scenario_name: str
    site_id: str = "hkust_gz"
    operating_mode: OperatingMode = OperatingMode.SIMULATION_ONLY
    load_dataset_id: str
    pv_profile_id: str
    battery_config_id: str
    step_minutes: int = 60
    horizon_steps: int = 24
    objective: ObjectiveMode = ObjectiveMode.COST
    max_cost_increase_pct: float | None = Field(default=None, ge=0)
    allow_grid_charging: bool = True
    allow_grid_export: bool = False
    critical_load_locked: bool = True
    simulation_only: bool = True
    executable: bool = False

    @model_validator(mode="after")
    def enforce_simulation_policy(self) -> "ParsedCampusRequest":
        if self.intent not in {"create_schedule", "compare_schedules"}:
            raise ValueError("unsupported campus EnergyOps intent")
        if self.step_minutes != 60 or self.horizon_steps != 24:
            raise ValueError("the first HKUST-GZ scenario must use 24 hourly points")
        if not self.allow_grid_charging:
            raise ValueError("the reference TOU simulation requires grid charging")
        if self.allow_grid_export:
            raise ValueError("grid export is disabled")
        if self.max_cost_increase_pct is not None and self.objective not in {
            ObjectiveMode.CARBON,
            ObjectiveMode.BALANCED,
        }:
            raise ValueError("cost-increase limits apply only to carbon-aware objectives")
        if not self.critical_load_locked:
            raise ValueError("critical loads must remain locked")
        if not self.simulation_only or self.executable:
            raise ValueError("this cross-site scenario is simulation-only and non-executable")
        return self


class GroundedExplanation(StrictModel):
    summary: str
    warnings: list[str] = Field(default_factory=list)
    evidence_claim_ids: list[str] = Field(default_factory=list)
    provider: str


class WorkflowRunResult(StrictModel):
    run_id: str
    status: WorkflowStatus
    tool_sequence: list[str]
    parsed_request: ParsedCampusRequest | None = None
    dispatch_request: DispatchRequest | None = None
    dispatch_result: DispatchResult | None = None
    verification: VerificationReport | None = None
    evidence: EvidencePack | None = None
    explanation: GroundedExplanation | None = None
    failure_code: str | None = None
    simulation_only: bool = True
    executable: bool = False


class BatteryScenarioResult(StrictModel):
    scenario_id: str
    power_kw: float
    capacity_kwh: float
    solver_status: str
    verification_passed: bool
    peak_import_kw: float | None = None
    peak_reduction_kw: float | None = None
    pv_self_consumption_ratio: float | None = None
    battery_throughput_kwh: float | None = None
    evidence_id: str


class BatteryScenarioComparison(StrictModel):
    dataset_version: str
    scenarios: list[BatteryScenarioResult]
    simulation_only: bool = True
    executable: bool = False


class SyntheticDatasetMetadata(StrictModel):
    dataset_id: str
    source: str = "synthetic_fixture"
    version: str
    generated_at: datetime
    timezone: str = "Asia/Shanghai"
    unit: str
    verified: bool = False
    simulation_only: bool = True
    assumptions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def enforce_synthetic_policy(self) -> "SyntheticDatasetMetadata":
        if self.generated_at.tzinfo is None:
            raise ValueError("generated_at must include timezone information")
        if self.source != "synthetic_fixture" or self.verified:
            raise ValueError("synthetic fixtures must remain unverified")
        if not self.simulation_only:
            raise ValueError("synthetic fixtures must remain simulation-only")
        return self


class ForecastSeriesPoint(StrictModel):
    timestamp: datetime
    load_kw: float = Field(ge=0)
    pv_kw: float = Field(ge=0)
    load_confidence: float = Field(ge=0, le=1)
    pv_confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_timestamp(self) -> "ForecastSeriesPoint":
        if self.timestamp.tzinfo is None:
            raise ValueError("forecast timestamps must include timezone information")
        return self


class ForecastDataset(StrictModel):
    metadata: SyntheticDatasetMetadata
    model_version: str
    points: list[ForecastSeriesPoint]


class CarbonFactorPoint(StrictModel):
    timestamp: datetime
    carbon_factor_kgco2e_per_kwh: float = Field(ge=0)
    quality: str = "synthetic"

    @model_validator(mode="after")
    def validate_timestamp(self) -> "CarbonFactorPoint":
        if self.timestamp.tzinfo is None:
            raise ValueError("carbon-factor timestamps must include timezone information")
        return self


class CarbonFactorDataset(StrictModel):
    metadata: SyntheticDatasetMetadata
    points: list[CarbonFactorPoint]


class DispatchPlanSummary(StrictModel):
    plan_id: str
    objective: ObjectiveMode
    verification_passed: bool
    total_cost_cny: float
    grid_emissions_kgco2e: float
    grid_import_energy_kwh: float
    peak_import_kw: float
    pv_self_consumption_ratio: float
    pv_curtailed_energy_kwh: float
    battery_throughput_kwh: float
    cost_change_vs_economic_pct: float
    emissions_change_vs_economic_pct: float


class DispatchPlanComparison(StrictModel):
    dataset_version: str
    plans: list[DispatchPlanSummary]
    recommended_plan_id: str
    recommendation_basis: str
    simulation_only: bool = True
    executable: bool = False
    status: WorkflowStatus = WorkflowStatus.PENDING_HUMAN_APPROVAL


class EnergyDecisionIntent(StrictModel):
    intent: str = "compare_schedules"
    objective: ObjectiveMode = ObjectiveMode.BALANCED
    max_cost_increase_pct: float = Field(default=5.0, ge=0)
    cost_limit_explicit: bool = False
    simulation_only: bool = True
    executable: bool = False

    @model_validator(mode="after")
    def enforce_decision_policy(self) -> "EnergyDecisionIntent":
        if self.intent not in {"create_schedule", "compare_schedules"}:
            raise ValueError("unsupported decision intent")
        if not self.simulation_only or self.executable:
            raise ValueError("first-stage decisions must remain simulation-only")
        return self


class VerifiedDispatchPlan(StrictModel):
    request: DispatchRequest
    dispatch: DispatchResult
    verification: VerificationReport


class DualObjectiveWorkflowResult(StrictModel):
    run_id: str
    status: WorkflowStatus
    intent: EnergyDecisionIntent
    tool_sequence: list[str]
    plans: list[VerifiedDispatchPlan] = Field(default_factory=list)
    comparison: DispatchPlanComparison | None = None
    explanation: GroundedExplanation | None = None
    failure_code: str | None = None
    simulation_only: bool = True
    executable: bool = False
