"""Affichages pratiques pour les notebooks (tableaux, rapport, graphe)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _d(x: Any) -> dict:
    return x.model_dump(mode="json") if hasattr(x, "model_dump") else x


def manifest_df(manifest) -> pd.DataFrame:
    m = _d(manifest)
    return pd.DataFrame([{
        "id": f["file_id"], "fichier": f["original_name"].split("/")[-1], "type": f["file_type"],
        "détection": f["detection_method"], "confiance": f["confidence"], "valide": f["schema_valid"],
        "archive": f["from_archive"], "route": " → ".join(f["route_to"]) or "quarantaine",
        "info LLM": f["metadata"].get("document_type", ""),
    } for f in m["files"]])


def phi_df(report) -> pd.DataFrame:
    r = _d(report)
    return pd.DataFrame([{"fichier": e["file_id"], "catégorie": e["category"], "détecteur": e["detector"],
                          "remplacement": e["replacement"]} for e in r["entities"]])


def features_df(pf) -> pd.DataFrame:
    p = _d(pf)
    return pd.DataFrame([{"feature": k, "valeur": v["value"], "unité": v["unit"], "code": v["code"],
                          "source": v["source_kind"], "fichier": v["source_file"], "date": v["observed_at"],
                          "preuve": (v["evidence"] or "")[:60]} for k, v in p["features"].items()])


def conditions_df(pf) -> pd.DataFrame:
    p = _d(pf)
    return pd.DataFrame([{"pathologie": c["label"], "SNOMED-CT": c["snomed"], "CIM-10": c["icd10"],
                          "niée": c["negated"], "source": c["source_kind"], "preuve": (c["evidence"] or "")[:50]}
                         for c in p["conditions"]])


def matches_df(match) -> pd.DataFrame:
    m = _d(match)
    return pd.DataFrame([{"modèle": x["model_id"], "compatible": x["compatible"], "confiance": x["confidence"],
                          "couverture requises": x["coverage_required"], "à dériver": ", ".join(x["derivable"]),
                          "manquantes": ", ".join(x["required_missing"]), "hors domaine": "; ".join(x["out_of_range"]),
                          "exclusion": ", ".join(x["excluded_by"])} for x in m["matches"]])


def audit_df(events: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"heure": e["ts"][11:19], "acteur": e["agent"], "action": e["action"],
                          "statut": e["status"], "empreinte": e.get("output_hash")} for e in events])


def show_report(report) -> None:
    from IPython.display import Image, Markdown, display

    r = _d(report)
    display(Markdown(r["markdown"].split("![Waterfall SHAP]")[0]))
    if r.get("path"):
        png = next(Path(r["path"]).parent.glob("shap_*.png"), None)
        if png:
            display(Image(filename=str(png)))
    display(Markdown("**Lecture**" + r["markdown"].split("**Lecture**", 1)[1]))


def show_graph(graph) -> None:
    """Affiche le graphe LangGraph (Mermaid en texte si le rendu PNG est indisponible)."""
    from IPython.display import Image, Markdown, display

    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception:
        display(Markdown("```mermaid\n" + graph.get_graph().draw_mermaid() + "\n```"))
