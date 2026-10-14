"""Agent 6 — Model Inference Agent with Explainability.

- Prédiction + intervalle d'incertitude (dispersion de 20 modèles bootstrap).
- Contributions SHAP exactes (LinearExplainer / TreeExplainer) + waterfall.
- Rapport clinique : les CHIFFRES sont insérés par le code à partir d'une
  fiche de faits ; le LLM rédige seulement la prose, à partir de cette fiche.
- Vérification d'ancrage (grounding) : tout nombre écrit par le LLM doit
  exister dans la fiche de faits, sinon régénération puis signalement.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..model_registry.registry import get_card, load_bundle
from ..schemas import (ClinicalReport, Contribution, MatchReport, PatientFeatures, Prediction, PreprocessResult,
                       ReportNarrativeLLM)
from ..tools import explain
from ..tools.terminology import OBSERVABLES
from .base import BaseAgent

SYSTEM_PROMPT = """Tu es un médecin rédacteur de rapports d'aide à la décision.
Tu rédiges en français, de façon factuelle et prudente, à partir UNIQUEMENT de la fiche de faits JSON.
Règles absolues :
- n'écris AUCUN chiffre qui ne figure pas dans la fiche (recopie-les à l'identique) ;
- présente le résultat comme une estimation probabiliste, jamais comme un diagnostic ;
- mentionne les variables imputées ou dérivées et les conflits de données s'il y en a ;
- n'émets aucune recommandation thérapeutique : liste ce que le clinicien doit vérifier."""

LABELS = {k: o.display for k, o in OBSERVABLES.items()} | {
    "diabetes": "Diabète de type 2", "atrial_fibrillation": "Fibrillation atriale",
    "family_history_diabetes": "Antécédent familial de diabète", "sex_female": "Sexe féminin",
}


def _numbers(text: str) -> list[str]:
    return re.findall(r"(?<![\w-])\d+(?:[.,]\d+)?", text)


def _allowed_numbers(obj: Any) -> set[float]:
    vals: set[float] = set()

    def walk(o):
        if isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            for x in (o, o * 100):
                for d in (0, 1, 2):
                    vals.add(round(float(x), d))
        elif isinstance(o, str):
            for n in _numbers(o):
                vals.add(float(n.replace(",", ".")))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(obj)
    return vals


def grounding_check(narrative: ReportNarrativeLLM, facts: dict[str, Any]) -> list[str]:
    allowed = _allowed_numbers(facts) | set(range(1, 5)) | {95.0}  # petits ordinaux, « IC 95 % »
    text = " ".join([narrative.synthese, *narrative.facteurs_explicatifs, *narrative.limites,
                     *narrative.points_a_verifier])
    bad = []
    for n in _numbers(text):
        v = float(n.replace(",", "."))
        if not any(abs(v - a) < 1e-9 for a in allowed):
            bad.append(n)
    return bad


class InferenceAgent(BaseAgent):
    name = "Agent 6 · Inference & XAI"
    role = "Prédire, expliquer, rédiger un rapport traçable"

    def __init__(self, llm=None, verbose: bool = True, use_llm: bool = True, top_k: int = 6):
        super().__init__(llm, verbose)
        self.use_llm = use_llm
        self.top_k = top_k

    # ------------------------------------------------------------ prédiction
    def predict(self, pre: PreprocessResult, figdir: Path | None = None) -> Prediction:
        card = get_card(pre.model_id)
        bundle = load_bundle(pre.model_id)
        cols = bundle["feature_names"]
        x = pd.DataFrame([pre.output_row])[cols]
        p = float(bundle["model"].predict_proba(x)[0, 1])
        boots = np.array([m.predict_proba(x)[0, 1] for m in bundle["bootstrap"]])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        cat = max((t for t in card.risk_thresholds.items() if p >= t[1]), key=lambda t: t[1])[0]

        vals, base = explain.shap_values(bundle["model"], bundle["background"][cols], x)
        contribs = sorted((Contribution(feature=c, value=float(x[c].iloc[0]), shap=round(float(s), 4))
                           for c, s in zip(cols, vals)), key=lambda c: abs(c.shap), reverse=True)
        fig = None
        if figdir is not None:
            fig = str(explain.waterfall_png(vals, base, x.iloc[0], [LABELS.get(c, c) for c in cols],
                                            Path(figdir) / f"shap_{pre.model_id}.png",
                                            f"Contributions SHAP (log-odds) — {card.name}"))
        pred = Prediction(model_id=card.model_id, model_version=card.version, probability=round(p, 4),
                          ci_low=round(float(lo), 4), ci_high=round(float(hi), 4), risk_category=cat,
                          base_value=round(base, 4), contributions=contribs, figure_path=fig)
        self.log(f"Risque {p:.1%} [{lo:.1%} ; {hi:.1%}] → {cat}", output=pred)
        return pred

    # --------------------------------------------------------------- rapport
    def facts(self, pf: PatientFeatures, match: MatchReport, pre: PreprocessResult, pred: Prediction) -> dict:
        card = get_card(pred.model_id)
        top = pred.contributions[: self.top_k]
        return {
            "modele": {"nom": card.name, "version": card.version, "question": card.clinical_question},
            "resultat": {"risque_pct": round(pred.probability * 100, 1),
                         "intervalle_incertitude_95_pct": [round(pred.ci_low * 100, 1), round(pred.ci_high * 100, 1)],
                         "categorie": pred.risk_category,
                         "seuils_pct": {k: round(v * 100) for k, v in card.risk_thresholds.items()}},
            "facteurs_principaux": [
                {"variable": LABELS.get(c.feature, c.feature), "valeur": round(c.value, 2),
                 "effet": "augmente le risque" if c.shap > 0 else "diminue le risque",
                 "contribution_log_odds": round(c.shap, 2)} for c in top],
            "qualite_donnees": {
                "imputees": [LABELS.get(f, f) for f in pre.imputed],
                "derivees": [LABELS.get(f, f) for f in pre.derived],
                "bornees": [LABELS.get(f, f) for f in pre.clipped],
                "conflits": [c.model_dump() for c in pf.conflicts],
                "extraites_par_llm": [LABELS.get(k, k) for k, f in pf.features.items()
                                      if f.source_kind == "llm_extraction"],
            },
            "limites_modele": card.limitations,
            "performance_validation": card.performance,
        }

    def report(self, pf: PatientFeatures, match: MatchReport, pre: PreprocessResult, pred: Prediction,
               outdir: Path | None = None) -> ClinicalReport:
        facts = self.facts(pf, match, pre, pred)
        narrative, bad = self._narrative(facts)
        md = self._render(pf, match, pre, pred, facts, narrative, bad)
        path = None
        if outdir is not None:
            outdir.mkdir(parents=True, exist_ok=True)
            path = outdir / f"rapport_{pf.pseudo_id}_{pred.model_id}.md"
            path.write_text(md, encoding="utf-8")
            (outdir / f"fiche_faits_{pf.pseudo_id}.json").write_text(json.dumps(facts, ensure_ascii=False, indent=2),
                                                                     "utf-8")
        rep = ClinicalReport(pseudo_id=pf.pseudo_id, model_id=pred.model_id, markdown=md, narrative=narrative,
                             grounding_ok=not bad, unsupported_numbers=bad, path=str(path) if path else None)
        self.log("Rapport rédigé — ancrage " + ("OK" if not bad else f"ÉCHEC {bad}"),
                 "ok" if not bad else "warning", output=rep)
        return rep

    def _narrative(self, facts: dict) -> tuple[ReportNarrativeLLM, list[str]]:
        if not self.use_llm:
            return self._fallback_narrative(facts), []
        prompt = f"Fiche de faits :\n{json.dumps(facts, ensure_ascii=False, indent=1)}"
        bad: list[str] = []
        narrative = None
        for _ in range(2):
            try:
                narrative = self.llm.structured(SYSTEM_PROMPT, prompt, ReportNarrativeLLM, agent=self.name)
            except Exception as exc:
                self.log(f"Rédaction LLM indisponible ({exc})", "warning")
                return self._fallback_narrative(facts), []
            bad = grounding_check(narrative, facts)
            if not bad:
                return narrative, []
            self.log(f"Chiffres non ancrés détectés {bad} → régénération", "warning")
            prompt += (f"\n\nATTENTION : ta version précédente contenait des chiffres absents de la fiche : {bad}. "
                       "Réécris sans aucun chiffre inventé.")
        return narrative, bad

    @staticmethod
    def _fallback_narrative(facts: dict) -> ReportNarrativeLLM:
        r = facts["resultat"]
        return ReportNarrativeLLM(
            synthese=f"Risque estimé {r['risque_pct']} % (catégorie {r['categorie']}).",
            facteurs_explicatifs=[f"{f['variable']} ({f['valeur']}) {f['effet']}" for f in facts["facteurs_principaux"]],
            limites=facts["limites_modele"], points_a_verifier=["Vérifier les données sources."])

    @staticmethod
    def _render(pf, match, pre, pred, facts, nar: ReportNarrativeLLM, bad: list[str]) -> str:
        card = get_card(pred.model_id)
        r = facts["resultat"]
        L = [f"# Rapport d'aide à la décision — {card.name}", "",
             f"**Patient (pseudonyme)** : `{pf.pseudo_id}` · **Date de référence** : {pf.reference_date} · "
             f"**Modèle** : `{card.model_id}` v{card.version} · **Statut** : BROUILLON — validation clinique requise", ""]
        if bad:
            L += [f"> ⚠️ **Ancrage non vérifié** : le texte rédigé contient des chiffres absents des données "
                  f"({', '.join(bad)}). À relire attentivement.", ""]
        L += ["## Résultat", "", "| Risque estimé | Incertitude (bootstrap 95 %) | Catégorie |", "|---|---|---|",
              f"| **{r['risque_pct']} %** | {r['intervalle_incertitude_95_pct'][0]} – "
              f"{r['intervalle_incertitude_95_pct'][1]} % | {r['categorie']} |", "",
              f"*Question* : {card.clinical_question}. *Seuils* : "
              + ", ".join(f"{k} ≥ {v} %" for k, v in r["seuils_pct"].items()) + ".", "",
              "## Synthèse", "", nar.synthese, "",
              "## Facteurs explicatifs (SHAP)", "", "| Variable | Valeur | Contribution (log-odds) | Effet |",
              "|---|---|---|---|"]
        for c in pred.contributions[:8]:
            arrow = "↑" if c.shap > 0 else "↓"
            L.append(f"| {LABELS.get(c.feature, c.feature)} | {c.value:g} | {c.shap:+.2f} | {arrow} |")
        L += ["", f"Valeur de base du modèle (log-odds) : {pred.base_value:+.2f}.", ""]
        if pred.figure_path:
            L += [f"![Waterfall SHAP]({Path(pred.figure_path).name})", ""]
        L += ["**Lecture** :", *[f"- {x}" for x in nar.facteurs_explicatifs], "",
              "## Qualité et provenance des données", ""]
        q = facts["qualite_donnees"]
        L += [f"- Variables dérivées : {', '.join(q['derivees']) or 'aucune'}",
              f"- Variables imputées (médiane d'entraînement) : {', '.join(q['imputees']) or 'aucune'}",
              f"- Variables bornées au domaine du modèle : {', '.join(q['bornees']) or 'aucune'}",
              f"- Extraites du texte libre par LLM (citation vérifiée) : {', '.join(q['extraites_par_llm']) or 'aucune'}",
              f"- Prétraitement exécuté : code **{pre.code_origin}** ({len(pre.attempts)} essai(s) LLM)", ""]
        if pf.conflicts:
            L += ["**Conflits entre sources** :", *[f"- {c.name} : {' / '.join(c.values)} → {c.resolution}"
                                                     for c in pf.conflicts], ""]
        L += ["| Variable | Valeur | Source | Date | Preuve |", "|---|---|---|---|---|"]
        for k, f in pf.features.items():
            ev = (f.evidence or "").replace("|", "/")[:70]
            L.append(f"| {LABELS.get(k, k)} | {f.value} {f.unit or ''} | {f.source_kind} · {f.source_file} | "
                     f"{f.observed_at or '—'} | {ev} |")
        L += ["", "## Points à vérifier par le clinicien", "", *[f"- [ ] {x}" for x in nar.points_a_verifier], "",
              "## Limites", "", *[f"- {x}" for x in dict.fromkeys(card.limitations + nar.limites)], "",
              "## Sélection du modèle", "", match.rationale, "",
              "---", f"*Performance de validation (cohorte synthétique)* : AUC {card.performance.get('auc')}, "
              f"Brier {card.performance.get('brier')}. *Empreinte de l'artefact* : `{(card.sha256 or '')[:16]}…`", "",
              "*Outil d'aide à la décision : ne remplace pas le jugement clinique. Modèle entraîné sur données "
              "synthétiques — démonstration pédagogique ICSMAI 2027.*"]
        return "\n".join(L) + "\n"
