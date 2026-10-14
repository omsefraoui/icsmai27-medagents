"""Agent 5 — Preprocessing Recommender & Implementor Agent.

1. RECOMMANDER (déterministe, piloté par la carte du modèle) : pour chaque
   variable attendue -> conserver, dériver (DFG CKD-EPI 2021, IMC), imputer
   (médiane d'entraînement si optionnelle), encoder, borner.
   ⚠️ Pas de standardisation ici : elle est DANS le pipeline du modèle
   (l'appliquer deux fois est une erreur classique et silencieuse).
2. IMPLEMENTOR : le LLM écrit le code `preprocess(df)` correspondant au plan.
   Le code est : analysé (AST) -> exécuté en sous-processus isolé -> soumis
   au contrat de sortie -> comparé à une implémentation de référence
   (test différentiel). En cas d'échec, l'erreur est renvoyée au LLM ; après
   N essais, on bascule sur le code de référence généré depuis le plan.
"""
from __future__ import annotations

import json

import pandas as pd

from .. import config
from ..model_registry.registry import get_card
from ..schemas import MatchReport, ModelCard, PatientFeatures, PreprocessCodeLLM, PreprocessResult, PreprocessStep
from ..tools import sandbox
from .base import BaseAgent

SYSTEM_PROMPT = """Tu es un ingénieur data en santé. Écris une fonction Python qui applique EXACTEMENT
le plan de prétraitement fourni.
Contraintes :
- signature : def preprocess(df):  (df = DataFrame pandas d'UNE ligne) ; renvoie un DataFrame ;
- imports autorisés : pandas as pd, numpy as np, math. Aucun autre import, aucun accès fichier/réseau ;
- pas de boucle while, pas d'attribut commençant par « _ » ;
- la sortie contient EXACTEMENT les colonnes demandées, dans l'ordre demandé, en float ;
- NE PAS standardiser / normaliser les variables (le modèle le fait déjà).
Formules :
- DFG CKD-EPI 2021 : kappa = 0.7 si femme sinon 0.9 ; alpha = -0.241 si femme sinon -0.302 ;
  egfr = 142 * min(creat/kappa, 1)**alpha * max(creat/kappa, 1)**(-1.200) * 0.9938**age * (1.012 si femme)
  puis arrondi à 1 décimale (creat en mg/dL, colonne creatinine ; colonne sex_female booléenne) ;
- IMC = poids (kg) / (taille (m))**2, arrondi à 1 décimale (colonne height en cm)."""


def plan_to_code(steps: list[PreprocessStep], columns: list[str]) -> str:
    """Implémentation de référence générée de façon déterministe à partir du plan."""
    lines = ["import numpy as np", "import pandas as pd", "", "", "def preprocess(df):", "    out = pd.DataFrame(index=df.index)"]
    for s in steps:
        c = s.feature
        if s.operation == "keep":
            lines.append(f"    out[{c!r}] = df[{c!r}].astype(float)")
        elif s.operation == "bool_to_int":
            lines.append(f"    out[{c!r}] = df[{c!r}].astype(bool).astype(float)")
        elif s.operation == "to_int":
            lines.append(f"    out[{c!r}] = df[{c!r}].astype(float).round()")
        elif s.operation == "impute_constant":
            lines.append(f"    out[{c!r}] = float({s.params['value']!r})")
        elif s.operation == "derive_egfr_ckdepi2021":
            lines += [
                "    female = df['sex_female'].astype(bool)",
                "    kappa = np.where(female, 0.7, 0.9)",
                "    alpha = np.where(female, -0.241, -0.302)",
                "    ratio = df['creatinine'].astype(float) / kappa",
                "    egfr = (142 * np.minimum(ratio, 1) ** alpha * np.maximum(ratio, 1) ** -1.200"
                " * 0.9938 ** df['age'].astype(float) * np.where(female, 1.012, 1.0))",
                f"    out[{c!r}] = np.round(egfr, 1)",
            ]
        elif s.operation == "derive_bmi":
            lines.append(f"    out[{c!r}] = np.round(df['weight'].astype(float) / (df['height'].astype(float) / 100) ** 2, 1)")
        elif s.operation == "clip":
            lo, hi = s.params.get("min"), s.params.get("max")
            lines.append(f"    out[{c!r}] = out[{c!r}].clip(lower={lo!r}, upper={hi!r})")
    lines.append(f"    return out[{columns!r}].astype(float)")
    return "\n".join(lines) + "\n"


class PreprocessingAgent(BaseAgent):
    name = "Agent 5 · Preprocessing"
    role = "Recommander puis implémenter (en sécurité) le prétraitement adapté au modèle"

    def __init__(self, llm=None, verbose: bool = True, use_llm: bool = True,
                 max_attempts: int = config.SANDBOX_MAX_ATTEMPTS):
        super().__init__(llm, verbose)
        self.use_llm = use_llm
        self.max_attempts = max_attempts

    # --------------------------------------------------------- recommandeur
    def recommend(self, card: ModelCard, pf: PatientFeatures) -> tuple[list[PreprocessStep], pd.DataFrame]:
        flat = pf.flat()
        steps: list[PreprocessStep] = []
        needed_inputs: set[str] = set()
        for spec in card.features:
            v = flat.get(spec.name)
            if v is None and spec.derivable_from and all(flat.get(d) is not None for d in spec.derivable_from):
                op = "derive_egfr_ckdepi2021" if spec.name == "egfr" else "derive_bmi"
                steps.append(PreprocessStep(feature=spec.name, operation=op, params={"inputs": spec.derivable_from},
                                            reason=f"absente mais dérivable de {', '.join(spec.derivable_from)}"))
                needed_inputs |= set(spec.derivable_from)
            elif v is None:
                if spec.required:
                    raise ValueError(f"Variable requise manquante et non dérivable : {spec.name}")
                steps.append(PreprocessStep(feature=spec.name, operation="impute_constant",
                                            params={"value": spec.impute_value},
                                            reason="optionnelle absente : médiane de la population d'entraînement"))
                continue
            elif spec.dtype == "bool":
                steps.append(PreprocessStep(feature=spec.name, operation="bool_to_int", reason="booléen -> 0/1"))
                needed_inputs.add(spec.name)
                continue
            elif spec.dtype == "int":
                steps.append(PreprocessStep(feature=spec.name, operation="to_int", reason="variable ordinale/entière"))
                needed_inputs.add(spec.name)
            else:
                steps.append(PreprocessStep(feature=spec.name, operation="keep", reason="présente, unité canonique"))
                needed_inputs.add(spec.name)
            if spec.min is not None or spec.max is not None:
                steps.append(PreprocessStep(feature=spec.name, operation="clip",
                                            params={"min": spec.min, "max": spec.max},
                                            reason="borner au domaine d'entraînement du modèle"))
        row = {k: flat[k] for k in sorted(needed_inputs) if k in flat}
        return steps, pd.DataFrame([row])

    # -------------------------------------------------------- implémenteur
    def run(self, pf: PatientFeatures, match: MatchReport) -> PreprocessResult:
        card = get_card(match.selected_model_id)
        steps, df_in = self.recommend(card, pf)
        cols = [f.name for f in card.features]
        bounds = {f.name: (f.min, f.max) for f in card.features}
        self.log(f"Plan de {len(steps)} étapes pour {card.model_id}",
                 output=[s.model_dump() for s in steps])

        ref_code = plan_to_code(steps, cols)
        ref = sandbox.run_sandboxed(ref_code, df_in, config.SANDBOX_TIMEOUT_S)
        if not ref.ok:
            raise RuntimeError(f"Implémentation de référence en échec : {ref.error} {ref.violations}")

        attempts: list[dict] = []
        code, origin, out = ref_code, "template", ref.output
        if self.use_llm:
            messages = (f"Plan (JSON) :\n{json.dumps([s.model_dump() for s in steps], ensure_ascii=False, indent=1)}\n\n"
                        f"Colonnes d'entrée et exemple : {df_in.to_dict(orient='records')}\n"
                        f"Colonnes de sortie (ordre exact) : {cols}")
            for i in range(1, self.max_attempts + 1):
                try:
                    gen = self.llm.structured(SYSTEM_PROMPT, messages, PreprocessCodeLLM, agent=self.name)
                except Exception as exc:
                    attempts.append({"attempt": i, "ok": False, "error": f"LLM : {exc}"})
                    break
                res = sandbox.run_sandboxed(gen.code, df_in, config.SANDBOX_TIMEOUT_S)
                issues = res.violations or ([res.error] if not res.ok else [])
                if res.ok:
                    issues += sandbox.check_output_contract(res.output, cols, bounds)
                if res.ok and not issues:
                    diff = self._diff(res.output, ref.output, cols)
                    issues += diff
                attempts.append({"attempt": i, "ok": not issues, "issues": issues, "code": gen.code})
                if not issues:
                    code, origin, out = gen.code, "llm", res.output
                    self.log(f"Code LLM validé à l'essai {i} (AST + bac à sable + contrat + test différentiel)")
                    break
                self.log(f"Essai {i} rejeté : {issues[0][:120]}", "warning")
                messages += (f"\n\nTon code précédent :\n{gen.code}\n\nIl a été REJETÉ pour : {issues}\n"
                             "Corrige-le et renvoie la version complète.")
            if origin == "template":
                self.log("Bascule sur l'implémentation de référence générée depuis le plan", "warning")

        output_row = {k: float(v) for k, v in out.iloc[0].items()}
        imputed = [s.feature for s in steps if s.operation == "impute_constant"]
        derived = [s.feature for s in steps if s.operation.startswith("derive")]
        raw = pf.flat()
        clipped = [s.feature for s in steps if s.operation == "clip" and s.feature in raw
                   and isinstance(raw[s.feature], (int, float)) and not isinstance(raw[s.feature], bool)
                   and abs(float(raw[s.feature]) - output_row[s.feature]) > 1e-9]
        result = PreprocessResult(model_id=card.model_id, steps=steps, code=code, code_origin=origin,
                                  attempts=attempts, input_row=json.loads(df_in.iloc[0].to_json()),
                                  output_row=output_row, imputed=imputed, derived=derived, clipped=clipped)
        self.log(f"Vecteur prêt ({origin}) — imputées {imputed}, dérivées {derived}, bornées {clipped}",
                 "warning" if clipped else "ok", output=result)
        return result

    @staticmethod
    def _diff(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> list[str]:
        out = []
        for c in cols:
            x, y = float(a[c].iloc[0]), float(b[c].iloc[0])
            if abs(x - y) > 0.051 + 1e-6 * abs(y):
                out.append(f"Test différentiel : {c} = {x:.3f} au lieu de {y:.3f} attendu")
        return out
