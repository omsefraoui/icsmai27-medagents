"""Socle commun des agents : accès LLM, journal d'audit, empreintes."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from ..llm import get_llm
from ..schemas import AuditEvent


def fingerprint(obj: Any) -> str:
    raw = obj.model_dump_json() if hasattr(obj, "model_dump_json") else json.dumps(obj, default=str, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class BaseAgent:
    name: str = "agent"
    role: str = ""

    def __init__(self, llm: Any | None = None, verbose: bool = True):
        self._llm = llm
        self.verbose = verbose
        self.events: list[AuditEvent] = []

    @property
    def llm(self):
        return self._llm or get_llm()

    def log(self, action: str, status: str = "ok", output: Any = None, **details: Any) -> AuditEvent:
        ev = AuditEvent(agent=self.name, action=action, status=status, details=details,
                        output_hash=fingerprint(output) if output is not None else None)
        self.events.append(ev)
        if self.verbose:
            icon = {"ok": "✅", "warning": "⚠️ ", "blocked": "⛔", "human": "🧑‍⚕️"}.get(status, "•")
            print(f"{icon} [{self.name}] {action}")
        return ev

    def drain_events(self) -> list[dict[str, Any]]:
        out = [e.model_dump(mode="json") for e in self.events]
        self.events = []
        return out
