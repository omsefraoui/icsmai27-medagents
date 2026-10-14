"""Agent 4 — Model-Data Feature Matcher Agent.

Pour chaque modèle du registre, vérifie de façon déterministe :
- critères d'inclusion / d'exclusion (pathologies) ;
- présence des variables requises, ou possibilité de les dériver ;
- plausibilité (bornes d'entraînement) et fraîcheur des mesures ;
puis calcule un score de confiance. Le LLM intervient ensuite pour choisir,
PARMI LES SEULS MODÈLES COMPATIBLES (Literal dynamique), celui qui répond le
mieux à la question clinique, et pour justifier ce choix.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, create_model

from ..model_registry.registry import list_cards
from ..schemas import MatchReport, ModelCard, ModelMatch, PatientFeatures
from .base import BaseAgent

SYSTEM_PROMPT = """Tu es un ingénieur en IA clinique. On te donne une question clinique, un résumé
(pseudonymisé) du patient et des modèles prédictifs déjà vérifiés comme techniquement compatibles.
Choisis le modèle le plus pertinent pour la question posée et justifie en 2-3 phrases en français.
Tu ne peux choisir qu'un identifiant présent dans la liste fournie."""

W_REQUIRED, W_OPTIONAL, W_FRESH = 0.6, 0.25, 0.15


class ModelMatcherAgent(BaseAgent):
    name = "Agent 4 · Model Matcher"
    role = "Associer les données du patient au(x) modèle(s) compatible(s)"

    def __init__(self, llm=None, verbose: bool = True, use_llm: bool = True):
        super().__init__(llm, verbose)
        self.use_llm = use_llm

    def evaluate(self, card: ModelCard, pf: PatientFeatures) -> ModelMatch:
        flat = pf.flat()
        present, derivable, req_missing, opt_missing, out_range, stale, reasons = [], [], [], [], [], [], []
        excluded = [c for c in card.exclusion_conditions if pf.has_condition(c)]
        not_included = [c for c in card.inclusion_conditions if not pf.has_condition(c)]
        if excluded:
            reasons.append(f"Critère d'exclusion présent : {', '.join(excluded)}")
        if not_included:
            reasons.append(f"Critère d'inclusion absent : {', '.join(not_included)}")

        for spec in card.features:
            val = flat.get(spec.name)
            if val is None:
                if spec.derivable_from and all(flat.get(d) is not None for d in spec.derivable_from):
                    derivable.append(spec.name)
                elif spec.required:
                    req_missing.append(spec.name)
                else:
                    opt_missing.append(spec.name)
                continue
            present.append(spec.name)
            if spec.dtype != "bool" and isinstance(val, (int, float)) and not isinstance(val, bool):
                if (spec.min is not None and val < spec.min) or (spec.max is not None and val > spec.max):
                    out_range.append(f"{spec.name}={val} (domaine du modèle {spec.min}–{spec.max})")
            f = pf.features.get(spec.name)
            if spec.max_age_days and f and f.observed_at and pf.reference_date:
                age = (pf.reference_date - f.observed_at).days
                if age > spec.max_age_days:
                    stale.append(f"{spec.name} ({age} j > {spec.max_age_days} j)")

        req = [s for s in card.features if s.required]
        opt = [s for s in card.features if not s.required]
        cov_req = sum(s.name in present or s.name in derivable for s in req) / max(len(req), 1)
        cov_opt = sum(s.name in present or s.name in derivable for s in opt) / max(len(opt), 1)
        fresh = 1 - len(stale) / max(len(present), 1)
        conf = W_REQUIRED * cov_req + W_OPTIONAL * cov_opt + W_FRESH * fresh
        conf -= 0.1 * len([o for o in out_range])
        compatible = not excluded and not not_included and not req_missing and not any(
            o.split("=")[0] in {s.name for s in req} for o in out_range)
        if req_missing:
            reasons.append(f"Variables requises manquantes : {', '.join(req_missing)}")
        if out_range:
            reasons.append(f"Valeurs hors domaine : {'; '.join(out_range)}")
        if derivable:
            reasons.append(f"Variables à dériver : {', '.join(derivable)}")
        if not compatible:
            conf = min(conf, 0.3) if not (excluded or not_included) else 0.0
        return ModelMatch(model_id=card.model_id, name=card.name, compatible=compatible,
                          confidence=round(max(conf, 0.0), 2), coverage_required=round(cov_req, 2),
                          coverage_optional=round(cov_opt, 2), present=present, derivable=derivable,
                          required_missing=req_missing, optional_missing=opt_missing, out_of_range=out_range,
                          stale=stale, excluded_by=excluded + [f"non-{c}" for c in not_included], reasons=reasons)

    def run(self, pf: PatientFeatures, clinical_question: str = "") -> MatchReport:
        cards = list_cards()
        matches = sorted((self.evaluate(c, pf) for c in cards), key=lambda m: (m.compatible, m.confidence),
                         reverse=True)
        compatible = [m for m in matches if m.compatible]
        for m in matches:
            self.log(f"{m.model_id}: {'compatible' if m.compatible else 'incompatible'} (confiance {m.confidence})",
                     "ok" if m.compatible else "warning", reasons=m.reasons)

        if not compatible:
            best = matches[0]
            report = MatchReport(clinical_question=clinical_question, matches=matches, selected_model_id=None,
                                 rationale=f"Aucun modèle compatible. Meilleur candidat : {best.model_id} "
                                           f"({'; '.join(best.reasons)})", needs_human_review=True)
            self.log("Aucun modèle compatible → revue humaine requise", "blocked", output=report)
            return report

        selected, rationale = compatible[0].model_id, "Seul modèle compatible (sélection déterministe)."
        if self.use_llm and (len(compatible) > 1 or clinical_question):
            ids = tuple(m.model_id for m in compatible)
            Choice = create_model("ModelChoice", selected_model_id=(Literal[ids], ...),  # type: ignore
                                  rationale=(str, Field(description="2 à 3 phrases en français")))
            summary = self._patient_summary(pf)
            cards_txt = "\n".join(f"- {m.model_id} : {c.clinical_question} | usage : {c.intended_use} | "
                                  f"confiance technique {m.confidence}"
                                  for m in compatible for c in cards if c.model_id == m.model_id)
            try:
                res = self.llm.structured(
                    SYSTEM_PROMPT,
                    f"Question clinique : {clinical_question or '(non précisée)'}\n\nPatient :\n{summary}\n\n"
                    f"Modèles compatibles :\n{cards_txt}", Choice, agent=self.name)
                selected, rationale = res.selected_model_id, res.rationale
            except Exception as exc:
                self.log(f"Sélection LLM indisponible ({exc}) → meilleur score retenu", "warning")
                rationale = "Modèle au meilleur score de confiance technique (LLM indisponible)."
        report = MatchReport(clinical_question=clinical_question, matches=matches, selected_model_id=selected,
                             rationale=rationale)
        self.log(f"Modèle retenu : {selected}", output=report)
        return report

    @staticmethod
    def _patient_summary(pf: PatientFeatures) -> str:
        feats = ", ".join(f"{k}={v.value}{(' ' + v.unit) if v.unit else ''}" for k, v in pf.features.items())
        conds = ", ".join(("NON " if c.negated else "") + c.label for c in pf.conditions)
        return f"Mesures : {feats}\nPathologies : {conds or 'aucune'}"
