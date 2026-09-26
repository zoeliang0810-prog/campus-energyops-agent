from __future__ import annotations

import json
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from energyops.agent_runtime import (
    AgentLoop,
    AgentModelStep,
    AgentPolicy,
    AgentToolCall,
    JsonAgentSessionStore,
    ToolDefinition,
    ToolRegistry,
)
from energyops.trace import TraceWriter


class EchoArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str


class ScriptedTransport:
    def __init__(self, steps: list[AgentModelStep]) -> None:
        self.steps = list(steps)
        self.requests: list[tuple[list[dict], list[dict]]] = []

    def complete(self, messages, tools):
        self.requests.append((list(messages), list(tools)))
        if not self.steps:
            raise RuntimeError("no scripted model step")
        return self.steps.pop(0)


def echo_tool(args: EchoArgs):
    return {"echo": args.text, "simulation_only": True, "executable": False}


def registry_with_echo() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Return a bounded test value.",
            args_model=EchoArgs,
            handler=echo_tool,
        )
    )
    return registry


def test_agent_loop_executes_tool_and_returns_final_response(tmp_path: Path):
    transport = ScriptedTransport(
        [
            AgentModelStep(
                tool_calls=[
                    AgentToolCall(
                        call_id="call-1", name="echo", arguments={"text": "ok"}
                    )
                ],
                finish_reason="tool_calls",
            ),
            AgentModelStep(content="已完成受限工具调用。", model="deepseek-flash"),
        ]
    )
    trace_path = tmp_path / "agent_trace.jsonl"
    result = AgentLoop(
        transport=transport,
        registry=registry_with_echo(),
        trace=TraceWriter(trace_path, "agent-test"),
    ).run("调用测试工具", session_id="session-1")

    assert result.status == "completed"
    assert result.steps == 2
    assert result.final_response == "已完成受限工具调用。"
    tool_messages = [item for item in result.messages if item["role"] == "tool"]
    assert json.loads(tool_messages[0]["content"])["echo"] == "ok"
    assert transport.requests[0][1][0]["function"]["name"] == "echo"
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert events[-1]["event_type"] == "agent_turn_completed"


def test_agent_loop_rejects_tool_outside_policy():
    transport = ScriptedTransport(
        [
            AgentModelStep(
                tool_calls=[
                    AgentToolCall(
                        call_id="call-2",
                        name="dangerous_shell",
                        arguments={},
                    )
                ]
            )
        ]
    )
    result = AgentLoop(
        transport=transport,
        registry=registry_with_echo(),
    ).run("运行命令", session_id="session-2")
    assert result.status == "safe_terminated"
    assert result.failure_code == "TOOL_NOT_ALLOWED"


def test_agent_loop_rejects_invalid_tool_arguments():
    transport = ScriptedTransport(
        [
            AgentModelStep(
                tool_calls=[
                    AgentToolCall(
                        call_id="call-3", name="echo", arguments={"wrong": "value"}
                    )
                ]
            )
        ]
    )
    result = AgentLoop(
        transport=transport,
        registry=registry_with_echo(),
    ).run("调用工具", session_id="session-3")
    assert result.status == "safe_terminated"
    assert result.failure_code == "TOOL_ARGUMENT_VALIDATION_FAILED"


def test_agent_loop_stops_at_step_limit():
    transport = ScriptedTransport(
        [
            AgentModelStep(
                tool_calls=[
                    AgentToolCall(
                        call_id="call-4", name="echo", arguments={"text": "again"}
                    )
                ]
            )
        ]
    )
    result = AgentLoop(
        transport=transport,
        registry=registry_with_echo(),
        policy=AgentPolicy(max_steps=1),
    ).run("循环", session_id="session-4")
    assert result.status == "safe_terminated"
    assert result.failure_code == "MAX_STEPS_EXCEEDED"


def test_agent_session_persists_across_turns(tmp_path: Path):
    transport = ScriptedTransport(
        [
            AgentModelStep(content="第一轮完成。"),
            AgentModelStep(content="第二轮完成。"),
        ]
    )
    store = JsonAgentSessionStore(tmp_path / "sessions")
    loop = AgentLoop(
        transport=transport,
        registry=registry_with_echo(),
        session_store=store,
    )

    first = loop.run("第一轮", session_id="durable-session")
    second = loop.run("第二轮", session_id="durable-session")

    assert first.status == "completed"
    assert second.status == "completed"
    session = store.load("durable-session")
    assert session is not None
    assert session.turn_count == 2
    assert [item["content"] for item in session.messages if item["role"] == "user"] == [
        "第一轮",
        "第二轮",
    ]
    second_request_messages = transport.requests[1][0]
    assert any(
        item["role"] == "assistant" and item["content"] == "第一轮完成。"
        for item in second_request_messages
    )


def test_agent_loop_supports_multiple_tool_calls_in_one_step():
    transport = ScriptedTransport(
        [
            AgentModelStep(
                tool_calls=[
                    AgentToolCall(
                        call_id="multi-1", name="echo", arguments={"text": "a"}
                    ),
                    AgentToolCall(
                        call_id="multi-2", name="echo", arguments={"text": "b"}
                    ),
                ]
            ),
            AgentModelStep(content="两个工具调用均已完成。"),
        ]
    )
    result = AgentLoop(
        transport=transport,
        registry=registry_with_echo(),
    ).run("执行两个调用", session_id="session-multi")
    assert result.status == "completed"
    assert len([item for item in result.messages if item["role"] == "tool"]) == 2


def test_agent_loop_safe_terminates_on_tool_timeout():
    registry = ToolRegistry()

    def slow_tool(args: EchoArgs):
        time.sleep(0.05)
        return {"echo": args.text}

    registry.register(
        ToolDefinition(
            name="slow",
            description="A deliberately slow test tool.",
            args_model=EchoArgs,
            handler=slow_tool,
        )
    )
    transport = ScriptedTransport(
        [
            AgentModelStep(
                tool_calls=[
                    AgentToolCall(
                        call_id="slow-1", name="slow", arguments={"text": "wait"}
                    )
                ]
            )
        ]
    )
    result = AgentLoop(
        transport=transport,
        registry=registry,
        policy=AgentPolicy(tool_timeout_seconds=0.001),
    ).run("超时测试", session_id="session-timeout")
    assert result.status == "safe_terminated"
    assert result.failure_code == "TOOL_TIMEOUT"


def test_agent_loop_safe_terminates_on_model_failure():
    result = AgentLoop(
        transport=ScriptedTransport([]),
        registry=registry_with_echo(),
    ).run("模型失败", session_id="session-model-failure")
    assert result.status == "safe_terminated"
    assert result.failure_code == "MODEL_CALL_FAILED"
