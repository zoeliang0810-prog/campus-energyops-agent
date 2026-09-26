"""Loopback-only HTTP boundary for the EnergyOps dashboard and AgentLoop."""

from __future__ import annotations

import json
import os
import threading
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .agent_app import build_energyops_agent_loop
from .agent_runtime import AgentLoop, AgentTurnResult


MAX_REQUEST_BYTES = 16_384
SESSION_PATTERN = r"[A-Za-z0-9_-]{1,128}"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
BUSINESS_TOOLS = [
    "get_available_scenarios",
    "get_data_quality_report",
    "create_verified_dispatch",
    "compare_verified_plans",
    "explain_evidence",
]


class AgentHTTPModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AgentAPIRequest(AgentHTTPModel):
    message: str = Field(min_length=1, max_length=4_000)
    session_id: str = Field(pattern=f"^{SESSION_PATTERN}$")


class AgentAPIResponse(AgentHTTPModel):
    session_id: str
    status: str
    steps: int = 0
    final_response: str | None = None
    failure_code: str | None = None
    tool_calls: list[str] = Field(default_factory=list)
    dispatch_run_ids: list[str] = Field(default_factory=list)
    simulation_only: bool = True
    executable: bool = False


LoopFactory = Callable[[str], AgentLoop]


def _result_tool_calls(result: AgentTurnResult) -> list[str]:
    names: list[str] = []
    for message in result.messages:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            name = call.get("function", {}).get("name")
            if isinstance(name, str) and name not in names:
                names.append(name)
    return names


def _result_dispatch_ids(result: AgentTurnResult) -> list[str]:
    run_ids: list[str] = []
    for message in result.messages:
        if message.get("role") != "tool":
            continue
        try:
            payload = json.loads(message.get("content", ""))
        except (TypeError, json.JSONDecodeError):
            continue
        run_id = payload.get("run_id") if isinstance(payload, dict) else None
        if isinstance(run_id, str) and run_id not in run_ids:
            run_ids.append(run_id)
    return run_ids


class EnergyOpsAgentService:
    """Serialize turns through a safe, response-minimizing HTTP application layer."""

    def __init__(
        self,
        *,
        project_root: Path,
        config_path: Path,
        state_dir: Path,
        model: str = "deepseek-flash",
        solver_name: str | None = None,
        loop_factory: LoopFactory | None = None,
    ) -> None:
        self.project_root = project_root.resolve()
        self.config_path = config_path.resolve()
        self.state_dir = state_dir.resolve()
        self.model = model
        self.solver_name = solver_name
        self._lock = threading.Lock()
        self._uses_default_factory = loop_factory is None
        self._loop_factory = loop_factory or self._default_loop_factory

    def _default_loop_factory(self, session_id: str) -> AgentLoop:
        return build_energyops_agent_loop(
            project_root=self.project_root,
            config_path=self.config_path,
            state_dir=self.state_dir,
            session_id=session_id,
            model=self.model,
            solver_name=self.solver_name,
        )

    def health(self) -> dict[str, Any]:
        configured = bool(os.environ.get("DEEPSEEK_API_KEY"))
        return {
            "status": "available" if configured else "unavailable",
            "api_configured": configured,
            "model": self.model,
            "tools": BUSINESS_TOOLS,
            "simulation_only": True,
            "executable": False,
            "reason": None if configured else "DEEPSEEK_API_KEY_NOT_CONFIGURED",
        }

    def _persist_latest_dispatch_result(self, result: AgentTurnResult) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        result_path = self.state_dir / "result.json"
        temporary = result_path.with_suffix(".tmp")
        temporary.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        temporary.replace(result_path)

    def run(self, request: AgentAPIRequest) -> AgentAPIResponse:
        if not os.environ.get("DEEPSEEK_API_KEY") and self._uses_default_factory:
            return AgentAPIResponse(
                session_id=request.session_id,
                status="safe_terminated",
                failure_code="DEEPSEEK_API_KEY_NOT_CONFIGURED",
            )
        with self._lock:
            try:
                result = self._loop_factory(request.session_id).run(
                    request.message.strip(), session_id=request.session_id
                )
            except Exception:
                return AgentAPIResponse(
                    session_id=request.session_id,
                    status="safe_terminated",
                    failure_code="AGENT_SERVICE_FAILED",
                )
        tool_calls = _result_tool_calls(result)
        if result.status == "completed" and not tool_calls:
            return AgentAPIResponse(
                session_id=result.session_id,
                status="safe_terminated",
                steps=result.steps,
                failure_code="UNGROUNDED_AGENT_RESPONSE",
            )
        dispatch_run_ids = _result_dispatch_ids(result)
        if result.status == "completed" and dispatch_run_ids:
            try:
                self._persist_latest_dispatch_result(result)
            except OSError:
                return AgentAPIResponse(
                    session_id=result.session_id,
                    status="safe_terminated",
                    steps=result.steps,
                    failure_code="AGENT_RESULT_PERSIST_FAILED",
                )
        return AgentAPIResponse(
            session_id=result.session_id,
            status=result.status,
            steps=result.steps,
            final_response=result.final_response,
            failure_code=result.failure_code,
            tool_calls=tool_calls,
            dispatch_run_ids=dispatch_run_ids,
            simulation_only=True,
            executable=False,
        )


def _origin_is_loopback(origin: str | None) -> bool:
    if not origin:
        return True
    parsed = urlsplit(origin)
    return parsed.scheme in {"http", "https"} and parsed.hostname in LOOPBACK_HOSTS


def create_request_handler(
    service: EnergyOpsAgentService, dist_dir: Path
) -> type[SimpleHTTPRequestHandler]:
    class EnergyOpsRequestHandler(SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(dist_dir), **kwargs)

        def end_headers(self) -> None:
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Frame-Options", "DENY")
            super().end_headers()

        def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/api/health":
                self._send_json(HTTPStatus.OK, service.health())
                return
            super().do_GET()

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/api/agent":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "NOT_FOUND"})
                return
            if not _origin_is_loopback(self.headers.get("Origin")):
                self._send_json(HTTPStatus.FORBIDDEN, {"error": "ORIGIN_NOT_ALLOWED"})
                return
            if self.headers.get_content_type() != "application/json":
                self._send_json(
                    HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                    {"error": "CONTENT_TYPE_MUST_BE_JSON"},
                )
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._send_json(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    {"error": "REQUEST_SIZE_INVALID"},
                )
                return
            try:
                payload = json.loads(self.rfile.read(length))
                request = AgentAPIRequest.model_validate(payload)
            except (json.JSONDecodeError, ValidationError):
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "INVALID_REQUEST"})
                return
            response = service.run(request)
            status = (
                HTTPStatus.OK
                if response.status == "completed"
                else HTTPStatus.SERVICE_UNAVAILABLE
            )
            self._send_json(status, response.model_dump(mode="json"))

        def log_message(self, format: str, *args: Any) -> None:
            print(f"[energyops-http] {self.address_string()} {format % args}")

    return EnergyOpsRequestHandler


def serve_dashboard(
    *,
    service: EnergyOpsAgentService,
    dist_dir: Path,
    host: str = "127.0.0.1",
    port: int = 4173,
) -> None:
    if host not in LOOPBACK_HOSTS:
        raise ValueError("ENERGYOPS_HTTP_MUST_BIND_TO_LOOPBACK")
    if not (dist_dir / "index.html").is_file():
        raise FileNotFoundError("DASHBOARD_DIST_NOT_BUILT")
    handler = create_request_handler(service, dist_dir.resolve())
    server = ThreadingHTTPServer((host, port), handler)
    print(f"EnergyOps dashboard: http://{host}:{port}")
    print(f"Agent API: http://{host}:{port}/api/agent")
    print(f"DeepSeek API configured: {service.health()['api_configured']}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
