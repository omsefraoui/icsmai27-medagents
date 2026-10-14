"""Accès au registre de modèles (cartes + artefacts vérifiés par empreinte)."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import joblib

from .. import config
from ..schemas import ModelCard
from ..tools.parsers import sha256_file


class IntegrityError(RuntimeError):
    pass


def ensure_models(verbose: bool = True) -> None:
    """Entraîne les modèles s'ils n'existent pas encore (≈ 30 s)."""
    if not (config.MODELS_DIR / "registry.json").exists():
        from .build import build

        build(verbose=verbose)


def list_cards() -> list[ModelCard]:
    ensure_models(verbose=False)
    data = json.loads((config.MODELS_DIR / "registry.json").read_text("utf-8"))
    return [ModelCard.model_validate(d) for d in data]


def get_card(model_id: str) -> ModelCard:
    for c in list_cards():
        if c.model_id == model_id:
            return c
    raise KeyError(f"Modèle inconnu : {model_id}")


@lru_cache(maxsize=8)
def load_bundle(model_id: str) -> dict[str, Any]:
    """Charge l'artefact APRÈS vérification de son empreinte SHA-256."""
    card = get_card(model_id)
    path = config.MODELS_DIR / card.artifact
    digest = sha256_file(path)
    if card.sha256 and digest != card.sha256:
        raise IntegrityError(f"Empreinte invalide pour {model_id} : artefact modifié ?")
    return joblib.load(path)
