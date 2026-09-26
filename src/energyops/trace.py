"""Append-only JSONL trace writer with field-level secret redaction."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


SENSITIVE_KEYS = {"api_key", "authorization", "token", "secret", "password"}


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.lower() in SENSITIVE_KEYS else _redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


class TraceWriter:
    def __init__(self, path: str | Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sequence = self._existing_sequence()

    def _existing_sequence(self) -> int:
        """Continue an append-only trace without reusing sequence numbers."""
        if not self.path.exists():
            return 0
        highest = 0
        try:
            with self.path.open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    if event.get("run_id") == self.run_id:
                        highest = max(highest, int(event.get("sequence", 0)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("TRACE_LOAD_FAILED") from error
        return highest

    def append(
        self,
        *,
        event_type: str,
        component: str,
        status: str,
        payload: dict[str, Any] | None = None,
        evidence_id: str | None = None,
    ) -> None:
        self.sequence += 1
        event = {
            "run_id": self.run_id,
            "sequence": self.sequence,
            "timestamp": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
            "event_type": event_type,
            "component": component,
            "status": status,
            "payload": _redact(payload or {}),
            "evidence_id": evidence_id,
        }
        with self.path.open("a", encoding="utf-8") as target:
            target.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
