"""Replaceable language-model providers for parsing and grounded explanation."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol, TypeVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import ValidationError

from .agent_runtime import AgentModelStep, AgentToolCall
from .contracts import EvidencePack, GroundedExplanation, ParsedCampusRequest


class AgentProvider(Protocol):
    name: str

    def parse_request(self, user_text: str) -> ParsedCampusRequest: ...

    def explain_result(
        self,
        request: ParsedCampusRequest,
        evidence: EvidencePack,
    ) -> GroundedExplanation: ...


class ProviderError(RuntimeError):
    """A structured provider boundary failure; never treated as a valid result."""


class MockProvider:
    """Offline deterministic provider used for tests and pre-API integration."""

    name = "mock"

    def __init__(self, parsed_request: ParsedCampusRequest | None = None) -> None:
        self.parsed_request = parsed_request

    def parse_request(self, user_text: str) -> ParsedCampusRequest:
        if self.parsed_request is not None:
            return self.parsed_request
        return ParsedCampusRequest(
            scenario_name="港科广负荷—九江光储配置仿真",
            load_dataset_id="hkust_gz_selected_meter_load",
            pv_profile_id="jiujiang_campus_pv_2026-02-19",
            battery_config_id="jiujiang_50kw_100kwh",
        )

    def explain_result(
        self,
        request: ParsedCampusRequest,
        evidence: EvidencePack,
    ) -> GroundedExplanation:
        claims = {claim.claim_id: claim for claim in evidence.claims}
        verification = claims["verification_status"].value
        peak_reduction = claims.get("peak_reduction_kw")
        pv_ratio = claims.get("pv_self_consumption_ratio")
        battery_charge = claims.get("battery_charge_energy_kwh")
        battery_discharge = claims.get("battery_discharge_energy_kwh")
        details = [f"独立校验结果为 {verification}"]
        references = ["verification_status", "release_status"]
        if peak_reduction is not None:
            details.append(f"仿真削减峰值 {float(peak_reduction.value):.3f} kW")
            references.append("peak_reduction_kw")
        if pv_ratio is not None:
            details.append(f"光伏自用率 {float(pv_ratio.value):.2%}")
            references.append("pv_self_consumption_ratio")
        if battery_charge is not None and battery_discharge is not None:
            details.append(
                "参考分时价格下储能充电 "
                f"{float(battery_charge.value):.3f} kWh、放电 "
                f"{float(battery_discharge.value):.3f} kWh"
            )
            references.extend(
                ["battery_charge_energy_kwh", "battery_discharge_energy_kwh"]
            )
        data_warnings: list[str] = []
        for dataset_name in ("pv", "load"):
            findings = claims.get(f"{dataset_name}_quality_findings")
            if findings is not None and isinstance(findings.value, list):
                data_warnings.extend(str(item) for item in findings.value)
                references.append(f"{dataset_name}_quality_findings")
        return GroundedExplanation(
            summary="；".join(details) + "。",
            warnings=[
                "这是校园本地天气推算光伏、设备参数未核验的仿真，不能作为设备控制指令。",
                "当前价格是九江旧方案中的广东2025参考分时价格，不报告真实电费节省。",
                *data_warnings,
            ],
            evidence_claim_ids=references,
            provider=self.name,
        )


ModelT = TypeVar("ModelT", ParsedCampusRequest, GroundedExplanation)


def _json_object(text: str) -> dict[str, object]:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text[text.find("{") : text.rfind("}") + 1]
    if not candidate:
        raise ValueError("provider returned no JSON object")
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("provider JSON must be an object")
    return value


class DeepSeekAPIProvider:
    """Direct DeepSeek Chat Completions adapter for parsing and explanation."""

    name = "deepseek-api"

    def __init__(
        self,
        *,
        model: str = "deepseek-flash",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
        max_normal_calls: int = 12,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com"
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_normal_calls = max_normal_calls
        self.normal_calls = 0
        self.repair_calls = 0
        if not self.api_key:
            raise ProviderError("real provider requires DEEPSEEK_API_KEY")

    def _consume_budget(self, *, repair: bool = False) -> None:
        if repair:
            if self.repair_calls >= 2:
                raise ProviderError("provider repair budget exhausted")
            self.repair_calls += 1
        else:
            if self.normal_calls >= self.max_normal_calls:
                raise ProviderError("provider normal-call budget exhausted")
            self.normal_calls += 1

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise ProviderError(f"DeepSeek API HTTP {error.code}") from error
        except URLError as error:
            raise ProviderError("DeepSeek API connection failed") from error
        except (UnicodeDecodeError, json.JSONDecodeError, OSError) as error:
            raise ProviderError("DeepSeek API returned an invalid response") from error
        if not isinstance(response_payload, dict):
            raise ProviderError("DeepSeek API response must be an object")
        return response_payload

    def _run(self, prompt: str, *, repair: bool = False) -> str:
        self._consume_budget(repair=repair)
        payload = self._post_chat(
            {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a constrained component of a simulation-only "
                            "campus EnergyOps system. Return one JSON object only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "response_format": {"type": "json_object"},
                "stream": False,
                "thinking": {"type": "disabled"},
                "reasoning_effort": "none",
                "max_tokens": 8192,
            }
        )

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError("DeepSeek API response has no message content") from error
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("DeepSeek API returned empty message content")
        return content

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AgentModelStep:
        """Return one model step for the self-owned AgentLoop."""
        self._consume_budget()
        request_payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "thinking": {"type": "disabled"},
            "reasoning_effort": "none",
            "max_tokens": 8192,
        }
        if tools:
            request_payload["tools"] = tools
            request_payload["tool_choice"] = "auto"
        payload = self._post_chat(request_payload)
        try:
            choice = payload["choices"][0]
            message = choice["message"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError("DeepSeek API response has no assistant message") from error

        tool_calls: list[AgentToolCall] = []
        for raw_call in message.get("tool_calls") or []:
            try:
                function = raw_call["function"]
                arguments = json.loads(function.get("arguments") or "{}")
                if not isinstance(arguments, dict):
                    raise ValueError("tool arguments must be an object")
                tool_calls.append(
                    AgentToolCall(
                        call_id=str(raw_call["id"]),
                        name=str(function["name"]),
                        arguments=arguments,
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ProviderError("DeepSeek API returned an invalid tool call") from error
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            raise ProviderError("DeepSeek API assistant content must be text or null")
        return AgentModelStep(
            content=content,
            tool_calls=tool_calls,
            finish_reason=str(choice.get("finish_reason", "stop")),
            model=str(payload.get("model", self.model)),
        )

    def _validated_json(self, prompt: str, model_type: type[ModelT], *, phase: str) -> ModelT:
        raw = self._run(prompt)
        try:
            return model_type.model_validate(_json_object(raw))
        except (ValueError, json.JSONDecodeError, ValidationError) as first_error:
            repair_prompt = (
                "Repair the following output into one JSON object that exactly matches this "
                f"JSON Schema. Do not add prose.\nSCHEMA:\n{json.dumps(model_type.model_json_schema(), ensure_ascii=False)}"
                f"\nINVALID OUTPUT:\n{raw}\nERROR:\n{first_error}"
            )
            repaired = self._run(
                repair_prompt,
                repair=True,
            )
            try:
                return model_type.model_validate(_json_object(repaired))
            except (ValueError, json.JSONDecodeError, ValidationError) as error:
                raise ProviderError(f"{phase} output failed local schema validation") from error

    def parse_request(self, user_text: str) -> ParsedCampusRequest:
        prompt = (
            "You are the request parser for a simulation-only campus EnergyOps workflow. "
            "Return JSON only. Never compute power, energy, cost, SOC, or a schedule. "
            "The only allowed scenario uses 24 hourly points, a fixed non-dispatchable load "
            "profile, simulated grid charging under a reference time-of-use price, no grid "
            "export, and is never executable. Extract the user's intent into this "
            f"schema:\n{json.dumps(ParsedCampusRequest.model_json_schema(), ensure_ascii=False)}"
            f"\nUSER REQUEST:\n{user_text}"
        )
        return self._validated_json(prompt, ParsedCampusRequest, phase="parse")

    def explain_result(
        self,
        request: ParsedCampusRequest,
        evidence: EvidencePack,
    ) -> GroundedExplanation:
        allowed_ids = {claim.claim_id for claim in evidence.claims}
        prompt = (
            "Explain only the supplied verified evidence. Return JSON only. Do not calculate new "
            "numbers, do not claim real savings, and do not describe this as executable. Every "
            "numeric statement must cite its claim id in evidence_claim_ids. "
            f"Schema:\n{json.dumps(GroundedExplanation.model_json_schema(), ensure_ascii=False)}"
            f"\nREQUEST:\n{request.model_dump_json()}\nEVIDENCE:\n{evidence.model_dump_json()}"
        )
        explanation = self._validated_json(
            prompt, GroundedExplanation, phase="explain"
        )
        unknown = set(explanation.evidence_claim_ids) - allowed_ids
        if unknown or not explanation.evidence_claim_ids:
            raise ProviderError(
                "explanation contains missing or unknown evidence claim bindings"
            )
        explanation.provider = self.name
        return explanation
