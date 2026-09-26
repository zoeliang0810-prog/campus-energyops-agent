"""Minimal, policy-bounded Agent Harness driven by the DeepSeek API."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .trace import TraceWriter


class AgentRuntimeModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentToolCall(AgentRuntimeModel):
    call_id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class AgentModelStep(AgentRuntimeModel):
    content: str | None = None
    tool_calls: list[AgentToolCall] = Field(default_factory=list)
    finish_reason: str = "stop"
    model: str | None = None


class AgentPolicy(AgentRuntimeModel):
    max_steps: int = Field(default=8, ge=1, le=32)
    max_tool_calls_per_step: int = Field(default=4, ge=1, le=16)
    tool_timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    max_tool_result_chars: int = Field(default=20_000, ge=256, le=1_000_000)
    allowed_tools: set[str] | None = None


class AgentTurnResult(AgentRuntimeModel):
    session_id: str
    status: str
    steps: int
    final_response: str | None = None
    failure_code: str | None = None
    messages: list[dict[str, Any]] = Field(default_factory=list)
    simulation_only: bool = True
    executable: bool = False


class AgentSession(AgentRuntimeModel):
    session_id: str
    turn_count: int = Field(default=0, ge=0)
    messages: list[dict[str, Any]] = Field(default_factory=list)


class AgentSessionStore(Protocol):
    def load(self, session_id: str) -> AgentSession | None: ...

    def save(self, session: AgentSession) -> None: ...


class JsonAgentSessionStore:
    """Small local JSON session store with traversal-safe session identifiers."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", session_id):
            raise AgentRuntimeError("INVALID_SESSION_ID")
        return self.root / f"{session_id}.json"

    def load(self, session_id: str) -> AgentSession | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            return AgentSession.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError) as error:
            raise AgentRuntimeError("SESSION_LOAD_FAILED") from error

    def save(self, session: AgentSession) -> None:
        path = self._path(session.session_id)
        temporary = path.with_suffix(".tmp")
        try:
            temporary.write_text(
                session.model_dump_json(indent=2), encoding="utf-8"
            )
            temporary.replace(path)
        except OSError as error:
            raise AgentRuntimeError("SESSION_SAVE_FAILED") from error


class AgentModelTransport(Protocol):
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentModelStep: ...


ToolHandler = Callable[[BaseModel], Any]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: ToolHandler

    def api_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.args_model.model_json_schema(),
            },
        }


class AgentRuntimeError(RuntimeError):
    """A safe, user-visible runtime boundary failure."""


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, definition: ToolDefinition) -> None:
        if not definition.name or definition.name in self._tools:
            raise ValueError(f"duplicate or empty tool name: {definition.name}")
        self._tools[definition.name] = definition

    @property
    def names(self) -> set[str]:
        return set(self._tools)

    def api_schemas(self, allowed_tools: set[str] | None = None) -> list[dict[str, Any]]:
        selected = allowed_tools if allowed_tools is not None else self.names
        return [
            self._tools[name].api_schema()
            for name in sorted(selected)
            if name in self._tools
        ]

    def execute(
        self,
        call: AgentToolCall,
        *,
        timeout_seconds: float,
        max_result_chars: int,
    ) -> str:
        definition = self._tools.get(call.name)
        if definition is None:
            raise AgentRuntimeError("UNKNOWN_TOOL")
        try:
            args = definition.args_model.model_validate(call.arguments)
        except ValidationError as error:
            raise AgentRuntimeError("TOOL_ARGUMENT_VALIDATION_FAILED") from error

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="energyops-tool")
        future = executor.submit(definition.handler, args)
        try:
            value = future.result(timeout=timeout_seconds)
        except FutureTimeoutError as error:
            future.cancel()
            raise AgentRuntimeError("TOOL_TIMEOUT") from error
        except Exception as error:
            raise AgentRuntimeError("TOOL_EXECUTION_FAILED") from error
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        encoded = json.dumps(value, ensure_ascii=False, default=str)
        if len(encoded) > max_result_chars:
            raise AgentRuntimeError("TOOL_RESULT_TOO_LARGE")
        return encoded


class AgentLoop:
    def __init__(
        self,
        *,
        transport: AgentModelTransport,
        registry: ToolRegistry,
        policy: AgentPolicy | None = None,
        system_prompt: str | None = None,
        trace: TraceWriter | None = None,
        session_store: AgentSessionStore | None = None,
    ) -> None:
        self.transport = transport
        self.registry = registry
        self.policy = policy or AgentPolicy()
        self.system_prompt = system_prompt or (
            "You are the language and orchestration layer of a simulation-only campus "
            "EnergyOps agent. Use only the supplied tools. Never invent energy values, "
            "never bypass verification, and never describe a result as executable."
        )
        self.trace = trace
        self.session_store = session_store

    def _save_session(
        self,
        session_id: str,
        turn_count: int,
        messages: list[dict[str, Any]],
    ) -> None:
        if self.session_store is not None:
            self.session_store.save(
                AgentSession(
                    session_id=session_id,
                    turn_count=turn_count,
                    messages=messages,
                )
            )

    def _trace(
        self,
        event_type: str,
        status: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self.trace is not None:
            self.trace.append(
                event_type=event_type,
                component="energyops_agent",
                status=status,
                payload=payload or {},
            )

    def _safe_terminated(
        self,
        *,
        session_id: str,
        steps: int,
        messages: list[dict[str, Any]],
        failure_code: str,
        turn_count: int = 1,
        persist: bool = True,
    ) -> AgentTurnResult:
        self._trace(
            "agent_turn_terminated",
            "safe_terminated",
            {"failure_code": failure_code, "steps": steps},
        )
        if persist:
            try:
                self._save_session(session_id, turn_count, messages)
            except AgentRuntimeError:
                failure_code = "SESSION_SAVE_FAILED"
        return AgentTurnResult(
            session_id=session_id,
            status="safe_terminated",
            steps=steps,
            failure_code=failure_code,
            messages=messages,
        )

    def run(self, user_text: str, *, session_id: str) -> AgentTurnResult:
        try:
            session = self.session_store.load(session_id) if self.session_store else None
        except AgentRuntimeError as error:
            return self._safe_terminated(
                session_id=session_id,
                steps=0,
                messages=[],
                failure_code=str(error),
                persist=False,
            )
        if session is None:
            session = AgentSession(
                session_id=session_id,
                messages=[{"role": "system", "content": self.system_prompt}],
            )
        messages = list(session.messages)
        messages.append({"role": "user", "content": user_text})
        turn_count = session.turn_count + 1
        allowed = self.policy.allowed_tools or self.registry.names
        if not allowed.issubset(self.registry.names):
            return self._safe_terminated(
                session_id=session_id,
                steps=0,
                messages=messages,
                failure_code="POLICY_REFERENCES_UNKNOWN_TOOL",
                turn_count=turn_count,
            )
        tool_schemas = self.registry.api_schemas(allowed)
        self._trace(
            "agent_turn_started",
            "running",
            {"session_id": session_id, "allowed_tools": sorted(allowed)},
        )

        for step_index in range(1, self.policy.max_steps + 1):
            try:
                self._trace(
                    "agent_model_requested",
                    "running",
                    {"step": step_index, "message_count": len(messages)},
                )
                step = self.transport.complete(messages, tool_schemas)
            except Exception as error:
                code = str(error) if str(error).isupper() else "MODEL_CALL_FAILED"
                return self._safe_terminated(
                    session_id=session_id,
                    steps=step_index,
                    messages=messages,
                    failure_code=code,
                    turn_count=turn_count,
                )

            if len(step.tool_calls) > self.policy.max_tool_calls_per_step:
                return self._safe_terminated(
                    session_id=session_id,
                    steps=step_index,
                    messages=messages,
                    failure_code="TOOL_CALL_BUDGET_EXCEEDED",
                    turn_count=turn_count,
                )

            assistant_message: dict[str, Any] = {
                "role": "assistant",
                "content": step.content,
            }
            if step.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.call_id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(
                                call.arguments, ensure_ascii=False, default=str
                            ),
                        },
                    }
                    for call in step.tool_calls
                ]
            messages.append(assistant_message)

            if not step.tool_calls:
                if not step.content or not step.content.strip():
                    return self._safe_terminated(
                        session_id=session_id,
                        steps=step_index,
                        messages=messages,
                        failure_code="EMPTY_FINAL_RESPONSE",
                        turn_count=turn_count,
                    )
                self._trace(
                    "agent_turn_completed",
                    "completed",
                    {"steps": step_index, "model": step.model},
                )
                try:
                    self._save_session(session_id, turn_count, messages)
                except AgentRuntimeError:
                    return self._safe_terminated(
                        session_id=session_id,
                        steps=step_index,
                        messages=messages,
                        failure_code="SESSION_SAVE_FAILED",
                        turn_count=turn_count,
                        persist=False,
                    )
                return AgentTurnResult(
                    session_id=session_id,
                    status="completed",
                    steps=step_index,
                    final_response=step.content,
                    messages=messages,
                )

            for call in step.tool_calls:
                if call.name not in allowed:
                    return self._safe_terminated(
                        session_id=session_id,
                        steps=step_index,
                        messages=messages,
                        failure_code="TOOL_NOT_ALLOWED",
                        turn_count=turn_count,
                    )
                self._trace(
                    "agent_tool_started",
                    "running",
                    {"step": step_index, "tool": call.name, "call_id": call.call_id},
                )
                try:
                    result = self.registry.execute(
                        call,
                        timeout_seconds=self.policy.tool_timeout_seconds,
                        max_result_chars=self.policy.max_tool_result_chars,
                    )
                except AgentRuntimeError as error:
                    return self._safe_terminated(
                        session_id=session_id,
                        steps=step_index,
                        messages=messages,
                        failure_code=str(error),
                        turn_count=turn_count,
                    )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.call_id,
                        "content": result,
                    }
                )
                self._trace(
                    "agent_tool_completed",
                    "pass",
                    {"step": step_index, "tool": call.name, "call_id": call.call_id},
                )
                try:
                    self._save_session(session_id, turn_count, messages)
                except AgentRuntimeError:
                    return self._safe_terminated(
                        session_id=session_id,
                        steps=step_index,
                        messages=messages,
                        failure_code="SESSION_SAVE_FAILED",
                        turn_count=turn_count,
                        persist=False,
                    )

        return self._safe_terminated(
            session_id=session_id,
            steps=self.policy.max_steps,
            messages=messages,
            failure_code="MAX_STEPS_EXCEEDED",
            turn_count=turn_count,
        )
