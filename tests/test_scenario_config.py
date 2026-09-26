from pathlib import Path

from energyops.contracts import ObjectiveMode
from energyops.scenario_config import load_scenario_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_campus_baseline_config_is_explicitly_simulation_only():
    config = load_scenario_config(
        PROJECT_ROOT / "config/scenarios/hkust_gz_campus_baseline.json"
    )
    assert config.site_id == "hkust_gz"
    assert config.pv.capacity_kwp == 300.0
    assert config.pv.capacity_status == "scenario_assumption"
    assert config.battery.max_charge_kw == 150.0
    assert config.battery.capacity_kwh == 400.0
    assert config.objective_mode == ObjectiveMode.COST
    assert config.grid.allow_grid_charging
    assert not config.grid.allow_grid_export
    assert not config.tariff.verified
    assert config.simulation_only
    assert not config.executable
