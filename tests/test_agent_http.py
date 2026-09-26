from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from energyops.agent_http import (
    AgentAPIRequest,
    EnergyOpsAgentService,
    _origin_is_loopback,
)
from energyops.agent_runtime import AgentTurnResult
from scripts.run_stage_f_dashboard import load_local_deepseek_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeLoop:
    def __init__(self, result: AgentTurnResult) -> None:
        self.result = result

    def run(self, user_text: str, *, session_id: str) -> AgentTurnResult:
        assert user_text == "生成经济调度"
        assert session_id == "web-session-1"
        return self.result


def service_with_result(tmp_path: Path, result: AgentTurnResult) -> EnergyOpsAgentService:
    return EnergyOpsAgentService(
        project_root=PROJECT_ROOT,
        config_path=PROJECT_ROOT / "config/scenarios/hkust_gz_campus_baseline.json",
        state_dir=tmp_path,
        loop_factory=lambda _session_id: FakeLoop(result),
    )


def test_agent_http_returns_only_bounded_public_result(tmp_path: Path) -> None:
    result = AgentTurnResult(
        session_id="web-session-1",
        status="completed",
        steps=2,
        final_response="已生成仿真计划，等待人工审批。",
        messages=[
            {"role": "system", "content": "private system prompt"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "create_verified_dispatch",
                            "arguments": "{}",
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": '{"run_id":"dispatch-safe-1"}',
            },
        ],
    )
    response = service_with_result(tmp_path, result).run(
        AgentAPIRequest(message="生成经济调度", session_id="web-session-1")
    )

    assert response.status == "completed"
    assert response.tool_calls == ["create_verified_dispatch"]
    assert response.dispatch_run_ids == ["dispatch-safe-1"]
    assert response.simulation_only is True
    assert response.executable is False
    assert "messages" not in response.model_dump()
    assert "private system prompt" not in response.model_dump_json()
    persisted = tmp_path / "result.json"
    assert persisted.is_file()
    assert "dispatch-safe-1" in persisted.read_text(encoding="utf-8")


def test_agent_http_fails_closed_without_api_key(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    service = EnergyOpsAgentService(
        project_root=PROJECT_ROOT,
        config_path=PROJECT_ROOT / "config/scenarios/hkust_gz_campus_baseline.json",
        state_dir=tmp_path,
    )

    assert service.health()["status"] == "unavailable"
    response = service.run(
        AgentAPIRequest(message="检查场景", session_id="web-session-2")
    )
    assert response.status == "safe_terminated"
    assert response.failure_code == "DEEPSEEK_API_KEY_NOT_CONFIGURED"


def test_agent_http_rejects_completed_response_without_tool_evidence(
    tmp_path: Path,
) -> None:
    result = AgentTurnResult(
        session_id="web-session-1",
        status="completed",
        steps=1,
        final_response="模型直接给出的无工具结论。",
        messages=[{"role": "assistant", "content": "模型直接给出的无工具结论。"}],
    )
    response = service_with_result(tmp_path, result).run(
        AgentAPIRequest(message="生成经济调度", session_id="web-session-1")
    )

    assert response.status == "safe_terminated"
    assert response.failure_code == "UNGROUNDED_AGENT_RESPONSE"
    assert response.final_response is None


def test_agent_http_rejects_invalid_request_and_remote_origin() -> None:
    with pytest.raises(ValidationError):
        AgentAPIRequest(message="", session_id="../outside")
    assert _origin_is_loopback("http://127.0.0.1:4173")
    assert _origin_is_loopback("http://localhost:4173")
    assert not _origin_is_loopback("https://example.com")


def test_dashboard_runner_loads_only_approved_env_keys(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("UNRELATED_SECRET", raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "DEEPSEEK_API_KEY='test-key-not-real'\n"
        "export DEEPSEEK_BASE_URL=https://example.invalid/v1\n"
        "UNRELATED_SECRET=must-not-load\n",
        encoding="utf-8",
    )

    load_local_deepseek_env(env_path)

    assert os.environ["DEEPSEEK_API_KEY"] == "test-key-not-real"
    assert os.environ["DEEPSEEK_BASE_URL"] == "https://example.invalid/v1"
    assert "UNRELATED_SECRET" not in os.environ
