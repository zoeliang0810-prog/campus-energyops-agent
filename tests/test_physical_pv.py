from pathlib import Path

import pandas as pd
import pytest

from energyops.physical_pv import (
    build_stage_a_sensitivity,
    calculate_physical_pv,
    load_physical_pv_config,
    write_physical_pv_outputs,
    write_stage_a_sensitivity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config/hkust_gz_physical_pv.json"
WEATHER_PATH = (
    PROJECT_ROOT / "data/fixtures/hkust_gz_weather_2026-02-19_60min.csv"
)


def load_run():
    config = load_physical_pv_config(CONFIG_PATH)
    weather = pd.read_csv(WEATHER_PATH)
    return config, calculate_physical_pv(weather, config)


def test_planning_config_matches_module_counts_and_capacities():
    config = load_physical_pv_config(CONFIG_PATH)

    assert config.array_capacity_kwp("roof") == pytest.approx(10350.6)
    assert config.array_capacity_kwp("facade") == pytest.approx(9422.4)
    assert config.array_capacity_kwp("parking") == pytest.approx(304.2)
    assert sum(array.module_count for array in config.arrays) == 30_888
    assert config.module.nominal_efficiency == pytest.approx(0.24065, rel=1e-4)
    assert not config.executable
    assert config.simulation_only


def test_physical_model_builds_three_auditable_scenarios():
    config, run = load_run()

    assert set(run.scenarios) == {
        "roof_only_10350kwp",
        "roof_parking_10654kwp",
        "full_build_20077kwp",
    }
    assert len(run.components) == 24
    assert not run.components.isna().any().any()

    full = run.scenarios["full_build_20077kwp"]
    expected = (
        full["roof_ac_power_kw"]
        + full["facade_ac_power_kw"]
        + full["parking_ac_power_kw"]
    )
    pd.testing.assert_series_equal(
        full["campus_pv_power_kw"], expected, check_names=False
    )
    assert full["campus_pv_power_kw"].sum() > 0


def test_facade_orientation_responds_to_morning_and_afternoon_sun():
    _, run = load_run()
    components = run.components.set_index("timestamp")

    morning = components.loc["2026-02-19T09:00:00+08:00"]
    afternoon = components.loc["2026-02-19T16:00:00+08:00"]
    assert morning["facade_east_ac_power_kw"] > morning["facade_west_ac_power_kw"]
    assert afternoon["facade_west_ac_power_kw"] > afternoon["facade_east_ac_power_kw"]


def test_quality_gates_enforce_night_zero_and_capacity_limit():
    config, run = load_run()

    for scenario in config.scenarios:
        quality = run.quality_reports[scenario.scenario_id]
        assert quality["hard_gate_passed"]
        assert quality["checks"] == {
            "timestamps_complete_hourly": True,
            "nonnegative_power": True,
            "capacity_limit": True,
            "night_power_zero": True,
        }
        assert quality["night_max_power_kw"] == pytest.approx(0.0)
        assert quality["daily_pv_energy_kwh"] > 0
        assert quality["peak_pv_power_kw"] < scenario.expected_capacity_kwp


def test_missing_weather_hour_is_rejected():
    config = load_physical_pv_config(CONFIG_PATH)
    weather = pd.read_csv(WEATHER_PATH).drop(index=12)

    with pytest.raises(ValueError, match="exactly 24 hourly rows"):
        calculate_physical_pv(weather, config)


def test_outputs_include_component_and_scenario_evidence(tmp_path: Path):
    config, run = load_run()

    paths = write_physical_pv_outputs(run, config, project_root=tmp_path)

    assert len(paths) == 10
    assert all(path.exists() for path in paths)
    assert any("components" in path.name for path in paths)
    assert sum(path.name.endswith("_manifest.json") for path in paths) == 3
    assert sum(path.name.endswith("_quality.json") for path in paths) == 3


def test_stage_a_sensitivity_has_three_orientation_and_loss_cases():
    config = load_physical_pv_config(CONFIG_PATH)
    weather = pd.read_csv(WEATHER_PATH)

    sensitivity = build_stage_a_sensitivity(weather, config)

    assert len(sensitivity) == 9
    assert sensitivity["facade_orientation_case"].nunique() == 3
    assert sensitivity["system_loss_fraction"].nunique() == 3
    assert sensitivity["hard_gate_passed"].all()
    by_loss = sensitivity.groupby("system_loss_fraction")[
        "daily_pv_energy_kwh"
    ].mean()
    assert by_loss.loc[0.10] > by_loss.loc[0.14] > by_loss.loc[0.18]


def test_stage_a_sensitivity_outputs_are_versioned(tmp_path: Path):
    config = load_physical_pv_config(CONFIG_PATH)
    weather = pd.read_csv(WEATHER_PATH)
    sensitivity = build_stage_a_sensitivity(weather, config)

    paths = write_stage_a_sensitivity(
        sensitivity, config, project_root=tmp_path
    )

    assert len(paths) == 2
    assert paths[0].suffix == ".csv"
    assert paths[1].name.endswith("_manifest.json")
    assert all(path.exists() for path in paths)
