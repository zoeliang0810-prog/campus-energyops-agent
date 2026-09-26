from pathlib import Path

import pytest

from energyops.agent_app import build_energyops_agent_loop
from energyops.agent_runtime import AgentModelStep, AgentToolCall


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("call_count", [2, 4, 5])
def test_business_loop_handles_bounded_tool_batches(tmp_path, call_count):
    """The dashboard factory must support batched queries and retain its cap."""
    class Transport:
        def complete(self, messages, tools):
            replies = [message for message in messages if message["role"] == "tool"]
            if replies:
                assert len(replies) == call_count
                return AgentModelStep(content="已查询现有场景和数据质量。")
            return AgentModelStep(tool_calls=[
                AgentToolCall(
                    call_id=f"query-{index}",
                    name=("get_available_scenarios" if index % 2 == 0
                          else "get_data_quality_report"),
                    arguments={},
                )
                for index in range(call_count)
            ])

    loop = build_energyops_agent_loop(
        project_root=PROJECT_ROOT,
        config_path=PROJECT_ROOT / "config/scenarios/hkust_gz_campus_baseline.json",
        state_dir=tmp_path,
        session_id="batch-query",
        transport=Transport(),
    )
    result = loop.run("计划和实际偏差多大？", session_id="batch-query")
    replies = [message for message in result.messages if message["role"] == "tool"]
    if call_count <= 4:
        assert result.status == "completed"
        assert len(replies) == call_count
        assert result.failure_code is None
    else:
        assert result.status == "safe_terminated"
        assert result.failure_code == "TOOL_CALL_BUDGET_EXCEEDED"
        assert not replies
