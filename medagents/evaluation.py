"""Évaluation du pipeline contre la vérité terrain des dossiers synthétiques.

Métriques :
- fuite de PHI : part des identifiants de référence retrouvés dans les sorties dé-identifiées ;
- exactitude des features : |valeur - attendu| ≤ 2 % (ou ≤ 0,05 en absolu) ;
- exactitude des pathologies (présence ET négation) ;
- choix du modèle.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from . import config
from .tools import parsers


def load_truth() -> dict[str, Any]:
    return json.loads((config.GROUND_TRUTH_DIR / "ground_truth.json").read_text("utf-8"))


def deid_corpus(phi_report: dict) -> str:
    parts = []
    for path in phi_report["deid_files"].values():
        p = Path(path)
        if p.suffix == ".dcm":
            ds = parsers.read_dicom(p)
            parts.append("\n".join(f"{el.keyword}={el.value}" for el in ds if el.keyword != "PixelData"))
        else:
            parts.append(parsers.read_text(p))
    return "\n".join(parts)


def phi_leaks(phi_report: dict, truth: dict) -> list[str]:
    corpus = deid_corpus(phi_report).lower()
    return [s for s in truth["phi_strings"] if s.lower() in corpus]


def feature_table(features: dict, truth: dict, preprocess: dict | None = None) -> pd.DataFrame:
    rows = []
    feats = dict(features["features"])
    for k, v in ((preprocess or {}).get("output_row") or {}).items():  # variables dérivées par l'Agent 5
        if k not in feats and k in (preprocess or {}).get("derived", []):
            feats[k] = {"value": v, "source_kind": "derived (Agent 5)"}
    for name, expected in truth["features"].items():
        got = feats.get(name, {}).get("value")
        ok = got is not None and abs(float(got) - expected) <= max(0.02 * abs(expected), 0.05)
        rows.append({"feature": name, "attendu": expected, "obtenu": got, "ok": ok,
                     "source": feats.get(name, {}).get("source_kind")})
    return pd.DataFrame(rows)


def condition_table(features: dict, truth: dict) -> pd.DataFrame:
    got = {c["name"]: not c["negated"] for c in features["conditions"]}
    return pd.DataFrame([{"pathologie": k, "attendu": v, "obtenu": got.get(k), "ok": got.get(k, False) == v}
                         for k, v in truth["conditions"].items()])


def evaluate_run(pid: str, state: dict) -> dict[str, Any]:
    truth = load_truth()[pid]
    leaks = phi_leaks(state["phi_report"], truth)
    ft = feature_table(state["features"], truth, state.get("preprocess"))
    ct = condition_table(state["features"], truth)
    selected = (state.get("match") or {}).get("selected_model_id")
    return {
        "patient": pid,
        "phi_rappel": round(1 - len(leaks) / len(truth["phi_strings"]), 3),
        "phi_fuites": leaks,
        "features_exactitude": round(ft["ok"].mean(), 3),
        "pathologies_exactitude": round(ct["ok"].mean(), 3),
        "modele_attendu": truth["expected_model"],
        "modele_choisi": selected,
        "modele_ok": selected == truth["expected_model"],
        "_features": ft,
        "_conditions": ct,
    }
