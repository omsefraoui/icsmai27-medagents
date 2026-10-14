"""Orchestration LangGraph du pipeline multi-agents, avec Human-in-the-Loop.

    START → ingestion → anonymisation ─┬─(PASS)──────→ extraction → matching ─┬─(modèle)→ prétraitement
                                       └─(BLOCKED)→ 🧑 revue_confidentialité   └─(aucun)→ 🧑 revue_données ─→ matching
    prétraitement → inférence → 🧑 validation_clinique → finalisation → END

Chaque nœud 🧑 appelle `interrupt()` : le graphe est mis en pause, son état
est sauvegardé par le *checkpointer*, et l'exécution reprend avec
`graph.invoke(Command(resume=...), config)` une fois la décision humaine prise.
"""
from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from . import config
from .agents.anonymizer import AnonymizerAgent
from .agents.features import FeatureExtractionAgent
from .agents.inference import InferenceAgent
from .agents.ingestion import IngestionAgent
from .agents.matcher import ModelMatcherAgent
from .agents.preprocess import PreprocessingAgent
from .schemas import (AuditEvent, ClinicalFeature, ClinicalReport, IngestionManifest, MatchReport, PatientFeatures,
                      PHIReport, PreprocessResult)
from .tools.terminology import OBSERVABLES

VERBOSE = True


class PipelineState(TypedDict, total=False):
    source: str
    workdir: str
    clinical_question: str
    manifest: dict
    phi_report: dict
    features: dict
    match: dict
    preprocess: dict
    prediction: dict
    report: dict
    status: str
    audit: Annotated[list, operator.add]
    human_decisions: Annotated[list, operator.add]


def _dump(m) -> dict:
    return m.model_dump(mode="json")


def _human_event(action: str, decision: dict, status: str = "human") -> dict:
    return AuditEvent(agent="Humain", action=action, status=status, details=decision).model_dump(mode="json")


# ================================================================== nœuds
def ingest(state: PipelineState) -> dict:
    agent = IngestionAgent(verbose=VERBOSE)
    manifest = agent.run(state["source"], state["workdir"])
    return {"manifest": _dump(manifest), "audit": agent.drain_events(), "status": "INGESTED"}


def anonymize(state: PipelineState) -> dict:
    agent = AnonymizerAgent(verbose=VERBOSE)
    rep = agent.run(IngestionManifest.model_validate(state["manifest"]), state["workdir"])
    return {"phi_report": _dump(rep), "audit": agent.drain_events(),
            "status": "DEIDENTIFIED" if rep.status == "PASS" else "PRIVACY_BLOCKED"}


def privacy_review(state: PipelineState) -> dict:
    rep = state["phi_report"]
    decision = interrupt({
        "type": "privacy",
        "question": "Le contrôle résiduel a trouvé des identifiants. Abandonner, ou accepter le risque "
                    "(justification obligatoire, tracée) ?",
        "residual_phi": rep["residual_phi"],
        "options": {"action": ["abort", "accept_risk"], "justification": "texte", "reviewer": "nom"},
    })
    ev = _human_event("Revue confidentialité", decision)
    if decision.get("action") != "accept_risk" or not decision.get("justification"):
        return {"status": "ABORTED_PRIVACY", "audit": [ev], "human_decisions": [decision]}
    return {"status": "DEIDENTIFIED_WITH_OVERRIDE", "audit": [ev], "human_decisions": [decision]}


def extract(state: PipelineState) -> dict:
    agent = FeatureExtractionAgent(verbose=VERBOSE)
    pf = agent.run(IngestionManifest.model_validate(state["manifest"]), PHIReport.model_validate(state["phi_report"]))
    return {"features": _dump(pf), "audit": agent.drain_events(), "status": "FEATURES_EXTRACTED"}


def match(state: PipelineState) -> dict:
    agent = ModelMatcherAgent(verbose=VERBOSE)
    rep = agent.run(PatientFeatures.model_validate(state["features"]), state.get("clinical_question", ""))
    return {"match": _dump(rep), "audit": agent.drain_events(),
            "status": "MODEL_SELECTED" if rep.selected_model_id else "NO_COMPATIBLE_MODEL"}


def data_review(state: PipelineState) -> dict:
    m = MatchReport.model_validate(state["match"])
    pf = PatientFeatures.model_validate(state["features"])
    best = m.matches[0]
    decision = interrupt({
        "type": "data",
        "question": "Aucun modèle compatible. Compléter/corriger des valeurs (avec leur source), ou abandonner ?",
        "best_candidate": best.model_id,
        "required_missing": best.required_missing,
        "out_of_range": best.out_of_range,
        "conflicts": [c.model_dump() for c in pf.conflicts],
        "options": {"action": ["provide_values", "abort"], "values": {"feature": "valeur (unité canonique)"},
                    "source": "texte", "reviewer": "nom"},
    })
    ev = _human_event("Revue des données", decision)
    if decision.get("action") != "provide_values" or not decision.get("values"):
        return {"status": "ABORTED_NO_MODEL", "audit": [ev], "human_decisions": [decision]}
    ref = pf.reference_date
    for name, value in decision["values"].items():
        obs = OBSERVABLES.get(name)
        pf.features[name] = ClinicalFeature(
            name=name, value=float(value), unit=obs.unit if obs else None, code_system="LOINC" if obs and obs.loinc
            else None, code=obs.loinc if obs else None, display=obs.display if obs else name, observed_at=ref,
            source_file=decision.get("source", "saisie clinicien"), source_kind="human",
            evidence=f"saisi par {decision.get('reviewer', '?')} — {decision.get('source', '')}")
        pf.conflicts = [c for c in pf.conflicts if c.name != name]
    return {"features": _dump(pf), "status": "DATA_COMPLETED_BY_HUMAN", "audit": [ev], "human_decisions": [decision]}


def preprocess(state: PipelineState) -> dict:
    agent = PreprocessingAgent(verbose=VERBOSE)
    res = agent.run(PatientFeatures.model_validate(state["features"]), MatchReport.model_validate(state["match"]))
    return {"preprocess": _dump(res), "audit": agent.drain_events(), "status": "PREPROCESSED"}


def infer(state: PipelineState) -> dict:
    agent = InferenceAgent(verbose=VERBOSE)
    pf = PatientFeatures.model_validate(state["features"])
    pre = PreprocessResult.model_validate(state["preprocess"])
    outdir = Path(state["workdir"]) / "rapport"
    pred = agent.predict(pre, figdir=outdir)
    rep = agent.report(pf, MatchReport.model_validate(state["match"]), pre, pred, outdir=outdir)
    return {"prediction": _dump(pred), "report": _dump(rep), "audit": agent.drain_events(), "status": "REPORT_DRAFTED"}


def clinical_review(state: PipelineState) -> dict:
    rep = ClinicalReport.model_validate(state["report"])
    decision = interrupt({
        "type": "clinical_review",
        "question": "Valider ou rejeter le rapport avant diffusion ?",
        "risk": state["prediction"]["probability"],
        "grounding_ok": rep.grounding_ok,
        "unsupported_numbers": rep.unsupported_numbers,
        "report_path": rep.path,
        "options": {"decision": ["approve", "reject"], "comment": "texte", "reviewer": "nom"},
    })
    approved = decision.get("decision") == "approve"
    rep.status = "VALIDATED" if approved else "REJECTED"
    rep.reviewer_comment = decision.get("comment")
    return {"report": _dump(rep), "status": "VALIDATED" if approved else "REJECTED",
            "audit": [_human_event("Validation clinique", decision)], "human_decisions": [decision]}


def finalize(state: PipelineState) -> dict:
    wd = Path(state["workdir"])
    ev = AuditEvent(agent="Orchestrateur", action=f"Fin du pipeline — statut {state.get('status')}").model_dump(mode="json")
    trail = state.get("audit", []) + [ev]
    with open(wd / "audit_trail.jsonl", "w", encoding="utf-8") as fh:
        for e in trail:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    rep = state.get("report")
    if rep and rep.get("path"):
        p = Path(rep["path"])
        md = p.read_text("utf-8").replace("BROUILLON — validation clinique requise", {
            "VALIDATED": f"VALIDÉ par {state['human_decisions'][-1].get('reviewer', '?')}",
            "REJECTED": f"REJETÉ par {state['human_decisions'][-1].get('reviewer', '?')}"}.get(rep["status"], rep["status"]))
        if rep.get("reviewer_comment"):
            md += f"\n**Commentaire du relecteur** : {rep['reviewer_comment']}\n"
        p.write_text(md, "utf-8")
    return {"audit": [ev]}


# ================================================================ routage
def route_privacy(state: PipelineState) -> str:
    return "extract" if state["status"] == "DEIDENTIFIED" else "privacy_review"


def route_after_privacy_review(state: PipelineState) -> str:
    return "extract" if state["status"] == "DEIDENTIFIED_WITH_OVERRIDE" else "finalize"


def route_model(state: PipelineState) -> str:
    return "preprocess" if state["status"] == "MODEL_SELECTED" else "data_review"


def route_after_data_review(state: PipelineState) -> str:
    return "match" if state["status"] == "DATA_COMPLETED_BY_HUMAN" else "finalize"


def build_graph(checkpointer: Any | None = None, human_review: bool = True):
    g = StateGraph(PipelineState)
    for name, fn in [("ingest", ingest), ("anonymize", anonymize), ("privacy_review", privacy_review),
                     ("extract", extract), ("match", match), ("data_review", data_review),
                     ("preprocess", preprocess), ("infer", infer), ("clinical_review", clinical_review),
                     ("finalize", finalize)]:
        g.add_node(name, fn)
    g.add_edge(START, "ingest")
    g.add_edge("ingest", "anonymize")
    g.add_conditional_edges("anonymize", route_privacy, ["extract", "privacy_review"])
    g.add_conditional_edges("privacy_review", route_after_privacy_review, ["extract", "finalize"])
    g.add_edge("extract", "match")
    g.add_conditional_edges("match", route_model, ["preprocess", "data_review"])
    g.add_conditional_edges("data_review", route_after_data_review, ["match", "finalize"])
    g.add_edge("preprocess", "infer")
    if human_review:
        g.add_edge("infer", "clinical_review")
        g.add_edge("clinical_review", "finalize")
    else:
        g.add_edge("infer", "finalize")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver())


# ================================================================ helpers
def new_run(source: str | Path, clinical_question: str = "", thread_id: str | None = None) -> tuple[dict, dict]:
    """État initial + configuration (thread) pour un dossier."""
    from datetime import datetime

    source = Path(source)
    # suffixe horaire : relancer une cellule crée un nouveau fil d'exécution propre
    thread_id = f"{thread_id or source.stem}_{datetime.now():%H%M%S}"
    workdir = config.OUTPUT_DIR / thread_id
    workdir.mkdir(parents=True, exist_ok=True)
    state = {"source": str(source), "workdir": str(workdir), "clinical_question": clinical_question,
             "audit": [], "human_decisions": []}
    return state, {"configurable": {"thread_id": thread_id}}


def pending_interrupt(graph, cfg) -> dict | None:
    """Charge utile de l'interruption en attente (ou None si le graphe ne l'est pas)."""
    snap = graph.get_state(cfg)
    for task in snap.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None


def resume(graph, cfg, value: dict) -> dict:
    return graph.invoke(Command(resume=value), cfg)
