"""LLM simulé — UNIQUEMENT pour les tests automatisés (CI sans GPU).

Il imite les réponses structurées d'un modèle Ollama à partir d'heuristiques,
et injecte volontairement des erreurs (entité hallucinée, code dangereux,
chiffre inventé) pour vérifier que les garde-fous des agents fonctionnent.
Il ne remplace en aucun cas Ollama pendant l'atelier.
"""
from __future__ import annotations

import json
import re
from medagents.llm import LLMCall


class FakeLLM:
    model = "fake-llm"
    host = "local"

    def __init__(self):
        self.history: list[LLMCall] = []
        self._code_calls = 0
        self._report_calls = 0

    def health_check(self):
        return {"ok": True, "model": self.model}

    def stats(self):
        return {"appels": len(self.history)}

    def last(self, n=1):
        return [c.__dict__ for c in self.history[-n:]]

    def chat(self, system, user, agent="?"):
        return "ok"

    def structured(self, system: str, user: str, schema, agent: str = "?", max_attempts: int = 3):
        name = schema.__name__
        fn = getattr(self, f"_{name}", None)
        data = fn(user, schema) if fn else self._generic(schema.model_json_schema())
        obj = schema.model_validate(data)
        self.history.append(LLMCall(agent, system, user, json.dumps(data, ensure_ascii=False, default=str), 0.0, name))
        return obj

    def _generic(self, js):
        out = {}
        for k, p in js.get("properties", {}).items():
            if "enum" in p:
                out[k] = p["enum"][0]
            elif p.get("type") == "array":
                out[k] = []
            elif p.get("type") in ("number", "integer"):
                out[k] = 0
            elif p.get("type") == "boolean":
                out[k] = False
            else:
                out[k] = "réponse simulée"
        return out

    # ------------------------------------------------------------------
    def _NoteClassificationLLM(self, user, schema):
        t = user.lower()
        dt = "compte_rendu_hospitalisation" if "hospitalisation" in t or "discharge" in t else "compte_rendu_consultation"
        return {"document_type": dt, "specialite": "cardiologie" if "cardio" in t else "médecine générale",
                "langue": "en" if "discharge" in t else "fr", "resume": "Document clinique."}

    def _PHIFindingsLLM(self, user, schema):
        ents = []
        for m in re.finditer(r"\b(?:Dr|Mme|M\.|Mr)\s+((?:[A-ZÉ][\wéèëç'-]+\s?){1,3})", user):
            ents.append({"text": m.group(1).strip(), "category": "NOM"})
        for m in re.finditer(r"\bMUT-\d+\b", user):
            ents.append({"text": m.group(0), "category": "AUTRE_ID"})
        ents.append({"text": "Jean Dupont", "category": "NOM"})  # hallucination volontaire
        ents.append({"text": "metformine", "category": "NOM"}) if "metformine" in user else None
        return {"entities": ents}

    def _NoteExtractionLLM(self, user, schema):
        t = user
        meas, conds, meds = [], [], []

        def add(name, pat, conv=float, unit=""):
            m = re.search(pat, t, re.I)
            if m:
                meas.append({"name": name, "value": conv(m.group(1).replace(",", ".")), "unit": unit,
                             "evidence": m.group(0)})

        add("FEVG", r"(?:FEVG|LVEF)[^\d\n]{0,20}(\d+)\s*%", unit="%")
        m = re.search(r"NYHA\s+(?:class\s+)?(I{1,3}|IV)", t, re.I)
        if m:
            meas.append({"name": "classe NYHA", "value": {"I": 1, "II": 2, "III": 3, "IV": 4}[m.group(1).upper()],
                         "evidence": m.group(0)})
        m = re.search(r"(?:PA|BP)\s+(\d{2,3})/(\d{2,3})", t)
        if m:
            meas += [{"name": "PA systolique", "value": float(m.group(1)), "unit": "mmHg", "evidence": m.group(0)},
                     {"name": "PA diastolique", "value": float(m.group(2)), "unit": "mmHg", "evidence": m.group(0)}]
        add("fréquence cardiaque", r"(?:FC|HR)\s+(\d+)", unit="/min")
        add("poids", r"[Pp]oids\s+(\d+)\s*kg", unit="kg")
        add("taille", r"(\d,\d{2})\s*m\b", unit="m")
        m = re.search(r"(Deux|Two|Trois)\s+(?:prior\s+)?(?:hospitalisations|admissions)[^.\n]*12", t, re.I)
        if m:
            meas.append({"name": "hospitalisations 12 mois", "value": {"deux": 2, "two": 2, "trois": 3}[m.group(1).lower()],
                         "evidence": m.group(0)})
        meas.append({"name": "NT-proBNP", "value": 9999, "unit": "pg/mL", "evidence": "NT-proBNP 9999"})  # hallucination
        for label, pat in [("insuffisance cardiaque", r"insuffisance cardiaque|heart failure"),
                           ("fibrillation atriale", r"fibrillation atriale|atrial fibrillation"),
                           ("diabète de type 2", r"diab[eéè]t\w*(?: de type 2)?|diabetes"),
                           ("insuffisance rénale chronique", r"insuffisance rénale chronique"),
                           ("BPCO", r"BPCO"), ("hypertension", r"\bHTA\b|hypertension")]:
            for m in re.finditer(pat, t, re.I):
                line_start = max(t.rfind("\n", 0, m.start()), t.rfind(".", 0, m.start())) + 1
                before = t[line_start:m.start()]
                ev = t[line_start:m.end()].strip()
                neg = bool(re.search(r"(?:\bpas d[e'’]\s*|\bno\s+|\bnot\s+)(?:\w+\s+){0,1}$", before, re.I))
                conds.append({"label": label, "negated": neg, "evidence": ev})
        for med in ["metformine", "bisoprolol", "furosémide", "apixaban", "Sacubitril/valsartan", "furosemide"]:
            m = re.search(re.escape(med), t, re.I)
            if m:
                meds.append({"name": med, "evidence": m.group(0)})
        return {"measurements": meas, "conditions": conds, "medications": meds}

    def _LabelMappings(self, user, schema):
        labels = re.findall(r"^- (.+)$", user.split("Libellés à associer :")[1], re.M)
        table = {"pro-bnp n-terminal": "nt_probnp", "pa diastolique": "dbp"}
        return {"mappings": [{"raw": lab, "canonical": table.get(lab.lower(), "AUCUN")} for lab in labels]}

    def _ModelChoice(self, user, schema):
        prop = schema.model_json_schema()["properties"]["selected_model_id"]
        ids = prop.get("enum") or [prop["const"]]
        return {"selected_model_id": ids[0], "rationale": "Modèle le plus adapté à la question posée."}

    def _PreprocessCodeLLM(self, user, schema):
        from medagents.agents.preprocess import plan_to_code
        from medagents.schemas import PreprocessStep

        self._code_calls += 1
        if self._code_calls % 2 == 1:  # premier essai : code dangereux -> doit être rejeté
            return {"code": "import os\ndef preprocess(df):\n    os.system('echo pwned')\n    return df",
                    "explanation": "tentative"}
        plan = json.loads(user.split("Plan (JSON) :\n")[1].split("\n\nColonnes d'entrée")[0])
        cols = eval(user.split("Colonnes de sortie (ordre exact) : ")[1].split("\n")[0])  # noqa: S307 (test)
        return {"code": plan_to_code([PreprocessStep(**s) for s in plan], cols), "explanation": "plan appliqué"}

    def _ReportNarrativeLLM(self, user, schema):
        self._report_calls += 1
        facts = json.loads(user.split("Fiche de faits :\n")[1].split("\n\nATTENTION")[0])
        r = facts["resultat"]
        extra = " Ce risque est 3,7 fois supérieur à la moyenne." if self._report_calls % 2 == 1 else ""
        return {
            "synthese": f"Le risque estimé est de {r['risque_pct']} % (catégorie {r['categorie']}).{extra}",
            "facteurs_explicatifs": [f"{f['variable']} ({f['valeur']}) {f['effet']}" for f in facts["facteurs_principaux"]],
            "limites": ["Modèle entraîné sur données synthétiques."],
            "points_a_verifier": ["Vérifier les valeurs extraites du texte libre."],
        }
