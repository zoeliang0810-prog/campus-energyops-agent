from __future__ import annotations

import tempfile
import unittest
import json
import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from energyops.boundary import validate_boundary_selection
from energyops.contracts import ObjectiveMode, WorkflowStatus
from energyops.harness_workflow import FIXED_TOOL_SEQUENCE, run_energyops_workflow
from energyops.providers import DeepSeekAPIProvider, MockProvider, ProviderError
from energyops.scenario import build_hourly_inputs
from energyops.scenario_comparison import compare_battery_scenarios
from energyops.trace import TraceWriter


class HarnessWorkflowTests(unittest.TestCase):
    def _inputs(self, root: Path):
        path = root / "pv.csv"
        timestamps = pd.date_range(
            "2026-02-19T00:00:00+08:00", periods=96, freq="15min"
        )
        pd.DataFrame(
            {"timestamp": timestamps, "campus_pv_power_kw": [20.0] * 96}
        ).to_csv(path, index=False)
        return build_hourly_inputs(path, dataset_version="combined:test")

    def test_mock_provider_completes_fixed_non_executable_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcome = run_energyops_workflow(
                run_id="test-run",
                user_text="生成仿真调度",
                inputs=self._inputs(root),
                quality_reports={
                    "pv": {"status": "pass", "evidence_id": "pv-q"},
                    "load": {
                        "status": "warning",
                        "evidence_id": "load-q",
                        "findings": ["研究日有42台表被报告为日期异常。"],
                    },
                },
                measured_load=False,
                provider=MockProvider(),
                trace=TraceWriter(root / "trace.jsonl", "test-run"),
            )
            self.assertEqual(outcome.status, WorkflowStatus.PENDING_HUMAN_APPROVAL)
            self.assertEqual(tuple(outcome.tool_sequence), FIXED_TOOL_SEQUENCE)
            self.assertFalse(outcome.executable)
            self.assertIsNotNone(outcome.dispatch_request)
            assert outcome.dispatch_request is not None
            self.assertEqual(
                outcome.dispatch_request.objective_mode,
                ObjectiveMode.COST,
            )
            self.assertTrue(outcome.dispatch_request.allow_grid_charging)
            self.assertFalse(outcome.dispatch_request.allow_grid_export)
            self.assertTrue(outcome.verification and outcome.verification.passed)
            assert outcome.verification is not None
            assert outcome.verification.recomputed_metrics is not None
            self.assertGreater(
                outcome.verification.recomputed_metrics.battery_charge_energy_kwh,
                0,
            )
            self.assertGreater(
                outcome.verification.recomputed_metrics.battery_discharge_energy_kwh,
                0,
            )
            assert outcome.explanation is not None
            self.assertIn(
                "研究日有42台表被报告为日期异常。",
                outcome.explanation.warnings,
            )
            self.assertIn(
                "load_quality_findings", outcome.explanation.evidence_claim_ids
            )
            assert outcome.evidence is not None
            claims = {claim.claim_id: claim for claim in outcome.evidence.claims}
            self.assertNotIn("grid_emissions_kgco2e", claims)
            self.assertEqual(
                claims["grid_emissions_status"].value,
                "unavailable_missing_carbon_factor",
            )
            comparison = compare_battery_scenarios(inputs=self._inputs(root), base_request=outcome.dispatch_request)
            self.assertEqual([item.scenario_id for item in comparison.scenarios], ["small", "medium", "large"])
            self.assertTrue(all(item.verification_passed for item in comparison.scenarios))
            self.assertFalse(comparison.executable)

    def test_failed_measured_load_quality_stops_before_optimization(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outcome = run_energyops_workflow(
                run_id="test-fail",
                user_text="生成仿真调度",
                inputs=self._inputs(root),
                quality_reports={
                    "pv": {"status": "pass"},
                    "load": {"status": "fail"},
                },
                measured_load=True,
                provider=MockProvider(),
                trace=TraceWriter(root / "trace.jsonl", "test-fail"),
            )
            self.assertEqual(outcome.status, WorkflowStatus.SAFE_TERMINATED)
            self.assertEqual(outcome.failure_code, "MEASURED_LOAD_QUALITY_NOT_PASSED")
            self.assertNotIn("create_energy_schedule", outcome.tool_sequence)


class BoundarySelectionTests(unittest.TestCase):
    def test_parent_and_child_cannot_both_be_selected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = root / "inventory.csv"
            boundary = root / "boundary.csv"
            pd.DataFrame(
                {
                    "ssuid": ["parent", "child"],
                    "chno": ["1", "1"],
                    "site_name_basis": ["user_confirmed", "user_confirmed"],
                }
            ).to_csv(inventory, index=False)
            pd.DataFrame(
                {
                    "building_or_zone": ["W1", "W1"],
                    "selected_ssuid": ["parent", "child"],
                    "channel": ["1", "1"],
                    "parent_ssuid": ["", "parent"],
                    "inclusion_reason": ["进线", "下游"],
                    "reviewer": ["reviewer", "reviewer"],
                    "effective_date": ["2026-09-14", "2026-09-14"],
                    "include": ["true", "true"],
                }
            ).to_csv(boundary, index=False)
            report = validate_boundary_selection(boundary, inventory)
            self.assertFalse(report.passed)
            self.assertTrue(any("parent" in error for error in report.errors))

    def test_user_confirmed_terminal_submeters_do_not_require_parent_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = root / "inventory.csv"
            boundary = root / "boundary.csv"
            pd.DataFrame(
                {
                    "ssuid": ["terminal"],
                    "chno": ["1"],
                    "site_name_basis": ["user_confirmed"],
                }
            ).to_csv(inventory, index=False)
            pd.DataFrame(
                {
                    "building_or_zone": ["PUBLIC-ZONE"],
                    "selected_ssuid": ["terminal"],
                    "channel": ["1"],
                    "parent_ssuid": [""],
                    "inclusion_reason": ["末端分表"],
                    "reviewer": ["reviewer"],
                    "effective_date": ["2026-09-15"],
                    "include": ["true"],
                    "boundary_basis": ["user_confirmed_terminal_submeter"],
                }
            ).to_csv(boundary, index=False)
            report = validate_boundary_selection(boundary, inventory)
            self.assertTrue(report.passed)
            self.assertFalse(any("parent_ssuid" in warning for warning in report.warnings))


class DeepSeekAPIProviderBoundaryTests(unittest.TestCase):
    def test_real_provider_requires_explicit_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ProviderError):
                DeepSeekAPIProvider()

    def test_real_provider_calls_chat_completions_directly(self):
        captured: dict[str, object] = {}
        payload = {
            "scenario_name": "港科广负荷—九江光储配置仿真",
            "load_dataset_id": "hkust_gz_selected_meter_load",
            "pv_profile_id": "jiujiang_campus_pv_2026-02-19",
            "battery_config_id": "jiujiang_50kw_100kwh",
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(payload, ensure_ascii=False)
                                }
                            }
                        ]
                    },
                    ensure_ascii=False,
                ).encode("utf-8")

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["authorization"] = request.get_header("Authorization")
            captured["body"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        with patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "test-only"}, clear=True
        ), patch("energyops.providers.urlopen", side_effect=fake_urlopen):
            provider = DeepSeekAPIProvider()
            parsed = provider.parse_request("生成调度")
            self.assertEqual(parsed.horizon_steps, 24)
            self.assertEqual(
                captured["url"], "https://api.deepseek.com/chat/completions"
            )
            self.assertEqual(captured["authorization"], "Bearer test-only")
            self.assertEqual(captured["body"]["model"], "deepseek-flash")
            self.assertEqual(
                captured["body"]["response_format"], {"type": "json_object"}
            )

    def test_real_provider_parses_function_tool_call(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {
                        "model": "deepseek-flash",
                        "choices": [
                            {
                                "finish_reason": "tool_calls",
                                "message": {
                                    "content": None,
                                    "tool_calls": [
                                        {
                                            "id": "call-api-1",
                                            "type": "function",
                                            "function": {
                                                "name": "echo",
                                                "arguments": '{"text":"ok"}',
                                            },
                                        }
                                    ],
                                },
                            }
                        ],
                    }
                ).encode("utf-8")

        with patch.dict(
            os.environ, {"DEEPSEEK_API_KEY": "test-only"}, clear=True
        ), patch("energyops.providers.urlopen", return_value=FakeResponse()):
            provider = DeepSeekAPIProvider()
            step = provider.complete(
                [{"role": "user", "content": "调用 echo"}],
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "description": "test",
                            "parameters": {"type": "object"},
                        },
                    }
                ],
            )
        self.assertEqual(step.finish_reason, "tool_calls")
        self.assertEqual(step.tool_calls[0].name, "echo")
        self.assertEqual(step.tool_calls[0].arguments, {"text": "ok"})


if __name__ == "__main__":
    unittest.main()
