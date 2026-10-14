"""Couche d'accès au LLM local (Ollama).

Principes appliqués dans tout l'atelier :
1. Le LLM tourne en local : aucune donnée patient ne quitte l'infrastructure.
2. Toute sortie du LLM destinée à la machine est *structurée* : on passe le
   schéma JSON Pydantic à Ollama (`format=`) puis on revalide avec Pydantic.
3. En cas de sortie invalide, on renvoie l'erreur au LLM (auto-correction),
   avec un nombre d'essais borné.
4. Chaque appel est journalisé (prompt, réponse, durée) pour l'audit.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from . import config

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Le LLM n'a pas produit de sortie exploitable."""


@dataclass
class LLMCall:
    agent: str
    system: str
    user: str
    response: str
    seconds: float
    schema: str | None = None
    attempt: int = 1
    ok: bool = True


@dataclass
class OllamaLLM:
    model: str = config.LLM_MODEL
    host: str = config.OLLAMA_HOST
    temperature: float = config.LLM_TEMPERATURE
    num_ctx: int = config.LLM_NUM_CTX
    timeout: float = config.LLM_TIMEOUT_S
    history: list[LLMCall] = field(default_factory=list)

    def __post_init__(self) -> None:
        import ollama  # import tardif : le package reste importable sans le client

        self._client = ollama.Client(host=self.host, timeout=self.timeout)

    # ------------------------------------------------------------------ utils
    def health_check(self) -> dict[str, Any]:
        """Vérifie que le serveur répond et que le modèle est disponible."""
        try:
            listed = self._client.list()
        except Exception as exc:  # pragma: no cover - dépend de l'environnement
            return {"ok": False, "host": self.host, "error": f"Serveur Ollama injoignable : {exc}"}
        names = [m.model for m in listed.models]
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        present = wanted in names or self.model in names
        return {
            "ok": present,
            "host": self.host,
            "model": self.model,
            "models_disponibles": names,
            "error": None if present else f"Modèle absent : lancez `ollama pull {self.model}`",
        }

    def _options(self) -> dict[str, Any]:
        return {"temperature": self.temperature, "num_ctx": self.num_ctx, "seed": config.LLM_SEED}

    # ------------------------------------------------------------------- chat
    def chat(self, system: str, user: str, agent: str = "?") -> str:
        """Réponse libre (texte). À réserver aux sorties destinées à un humain."""
        t0 = time.perf_counter()
        resp = self._client.chat(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            options=self._options(),
        )
        content = resp.message.content or ""
        self.history.append(LLMCall(agent, system, user, content, time.perf_counter() - t0))
        return content

    def structured(
        self,
        system: str,
        user: str,
        schema: type[T],
        agent: str = "?",
        max_attempts: int = 3,
    ) -> T:
        """Sortie structurée validée par Pydantic, avec boucle d'auto-correction."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        json_schema = schema.model_json_schema()
        last_error = ""
        for attempt in range(1, max_attempts + 1):
            t0 = time.perf_counter()
            resp = self._client.chat(
                model=self.model, messages=messages, format=json_schema, options=self._options()
            )
            content = resp.message.content or ""
            try:
                parsed = schema.model_validate_json(content)
                self.history.append(
                    LLMCall(agent, system, user, content, time.perf_counter() - t0, schema.__name__, attempt)
                )
                return parsed
            except ValidationError as exc:
                last_error = exc.json(include_url=False)
                self.history.append(
                    LLMCall(agent, system, user, content, time.perf_counter() - t0, schema.__name__, attempt, ok=False)
                )
                messages += [
                    {"role": "assistant", "content": content},
                    {
                        "role": "user",
                        "content": "Ta réponse ne respecte pas le schéma JSON. Erreurs de validation :\n"
                        f"{last_error}\nRenvoie UNIQUEMENT un JSON valide corrigé.",
                    },
                ]
        raise LLMError(f"{agent}: sortie invalide après {max_attempts} essais : {last_error[:500]}")

    # ----------------------------------------------------------- observabilité
    def stats(self) -> dict[str, Any]:
        calls = self.history
        return {
            "appels": len(calls),
            "echecs_validation": sum(not c.ok for c in calls),
            "secondes_total": round(sum(c.seconds for c in calls), 1),
            "par_agent": {
                a: sum(1 for c in calls if c.agent == a) for a in sorted({c.agent for c in calls})
            },
        }

    def last(self, n: int = 1) -> list[dict[str, Any]]:
        return [c.__dict__ for c in self.history[-n:]]


# ---------------------------------------------------------------- singleton
_LLM: Any = None


def get_llm() -> OllamaLLM:
    """LLM partagé par tous les agents (modifiable via set_llm)."""
    global _LLM
    if _LLM is None:
        _LLM = OllamaLLM()
    return _LLM


def set_llm(llm: Any) -> None:
    """Injecte un autre client (autre modèle, autre hôte...)."""
    global _LLM
    _LLM = llm


def pretty(obj: Any) -> str:
    if isinstance(obj, BaseModel):
        return obj.model_dump_json(indent=2)
    return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
