"""Mini-terminologie clinique pour l'atelier.

En production on interrogerait un serveur de terminologie (FHIR
`$lookup`/`$translate`, Snowstorm pour SNOMED-CT, l'API LOINC, le CIM-10 de
l'ATIH...). Ici, un dictionnaire embarqué suffit à illustrer :
- la normalisation des noms (synonymes FR/EN, fautes, abréviations) ;
- le codage LOINC / SNOMED-CT / CIM-10 / ATC ;
- la conversion d'unités vers l'unité canonique.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Callable


def norm(text: str) -> str:
    """minuscules, sans accents, sans ponctuation superflue."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-z0-9%/ ]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def norm_unit(unit: str | None) -> str:
    if not unit:
        return ""
    u = unit.strip().replace("µ", "u").replace("μ", "u").replace(" ", "").lower()
    return {"mmol/mol": "mmol/mol", "%": "%", "umol/l": "umol/l", "µmol/l": "umol/l"}.get(u, u)


@dataclass
class Observable:
    name: str
    display: str
    loinc: str | None
    unit: str
    synonyms: list[str]
    plausible: tuple[float, float]
    conversions: dict[str, Callable[[float], float]] = field(default_factory=dict)


OBSERVABLES: dict[str, Observable] = {
    o.name: o
    for o in [
        Observable("age", "Âge", "30525-0", "a", ["age", "age ans", "âge du patient", "age years"], (0, 120)),
        Observable("bmi", "Indice de masse corporelle", "39156-5", "kg/m2",
                   ["imc", "bmi", "indice de masse corporelle", "body mass index"], (10, 80)),
        Observable("weight", "Poids", "29463-7", "kg", ["poids", "weight", "masse corporelle"], (20, 300)),
        Observable("height", "Taille", "8302-2", "cm", ["taille", "height", "stature"], (100, 230),
                   {"m": lambda v: v * 100}),
        Observable("sbp", "Pression artérielle systolique", "8480-6", "mmHg",
                   ["pas", "pression arterielle systolique", "tension systolique", "systolic blood pressure",
                    "sbp", "ta systolique", "pa systolique"], (50, 260)),
        Observable("dbp", "Pression artérielle diastolique", "8462-4", "mmHg",
                   ["pad", "pression arterielle diastolique", "diastolic blood pressure", "dbp", "pa diastolique"],
                   (30, 160)),
        Observable("heart_rate", "Fréquence cardiaque", "8867-4", "/min",
                   ["fc", "frequence cardiaque", "heart rate", "pouls", "hr"], (20, 250)),
        Observable("hba1c", "Hémoglobine glyquée", "4548-4", "%",
                   ["hba1c", "hemoglobine glyquee", "hb a1c", "glycated hemoglobin", "a1c"], (3, 20),
                   {"mmol/mol": lambda v: round(v / 10.929 + 2.15, 2)}),
        Observable("fasting_glucose", "Glycémie à jeun", "1558-6", "mg/dL",
                   ["glycemie a jeun", "glycemie", "glucose a jeun", "fasting glucose", "gaj", "glucose"],
                   (30, 700), {"mmol/l": lambda v: round(v * 18.016, 1), "g/l": lambda v: v * 100}),
        Observable("hdl", "Cholestérol HDL", "2085-9", "mg/dL",
                   ["hdl", "hdl cholesterol", "cholesterol hdl", "hdl c"], (5, 150),
                   {"mmol/l": lambda v: round(v * 38.67, 1), "g/l": lambda v: v * 100}),
        Observable("triglycerides", "Triglycérides", "2571-8", "mg/dL",
                   ["triglycerides", "tg", "trigly"], (10, 3000),
                   {"mmol/l": lambda v: round(v * 88.57, 1), "g/l": lambda v: v * 100}),
        Observable("creatinine", "Créatinine sérique", "2160-0", "mg/dL",
                   ["creatinine", "creatininemie", "creat", "serum creatinine", "creatinine serique"], (0.1, 20),
                   {"umol/l": lambda v: round(v / 88.42, 2)}),
        Observable("egfr", "DFG estimé (CKD-EPI)", "62238-1", "mL/min/1.73m2",
                   ["dfg", "dfge", "egfr", "debit de filtration glomerulaire", "ckd epi", "dfg ckd epi"], (2, 150)),
        Observable("uacr", "Rapport albuminurie/créatininurie", "9318-7", "mg/g",
                   ["rac", "uacr", "albuminurie creatininurie", "microalbuminurie", "ratio albumine creatinine"],
                   (0, 5000), {"mg/mmol": lambda v: round(v * 8.84, 1)}),
        Observable("sodium", "Natrémie", "2951-2", "mmol/L",
                   ["sodium", "natremie", "na", "na+"], (100, 180), {"meq/l": lambda v: v}),
        Observable("potassium", "Kaliémie", "2823-3", "mmol/L",
                   ["potassium", "kaliemie", "k", "k+"], (1.5, 9), {"meq/l": lambda v: v}),
        Observable("hemoglobin", "Hémoglobine", "718-7", "g/dL",
                   ["hemoglobine", "hb", "hgb", "hemoglobin"], (3, 25), {"g/l": lambda v: v / 10}),
        Observable("nt_probnp", "NT-proBNP", "33762-6", "pg/mL",
                   ["nt probnp", "ntprobnp", "nt pro bnp", "pro bnp", "nt-probnp"], (5, 70000),
                   {"ng/l": lambda v: v}),
        Observable("lvef", "Fraction d'éjection ventriculaire gauche", "10230-1", "%",
                   ["fevg", "fe vg", "fraction d ejection", "lvef", "ejection fraction", "fe"], (5, 90)),
        Observable("nyha_class", "Classe fonctionnelle NYHA", "88020-3", "classe",
                   ["nyha", "classe nyha", "stade nyha", "nyha class"], (1, 4)),
        Observable("prior_admissions_12m", "Hospitalisations dans les 12 derniers mois", None, "n",
                   ["hospitalisations 12 mois", "nb hospitalisations", "prior admissions", "hospitalisations anterieures"],
                   (0, 30)),
    ]
}

LOINC_INDEX = {o.loinc: o.name for o in OBSERVABLES.values() if o.loinc}


@dataclass
class Concept:
    name: str
    label: str
    snomed: str | None
    icd10: str | None
    synonyms: list[str]


CONDITIONS: dict[str, Concept] = {
    c.name: c
    for c in [
        Concept("diabetes", "Diabète de type 2", "44054006", "E11",
                ["diabete de type 2", "diabete type 2", "dt2", "diabete", "type 2 diabetes", "diabetes mellitus", "dnid"]),
        Concept("hypertension", "Hypertension artérielle", "38341003", "I10",
                ["hta", "hypertension arterielle", "hypertension", "high blood pressure"]),
        Concept("heart_failure", "Insuffisance cardiaque", "84114007", "I50",
                ["insuffisance cardiaque", "ic", "icfer", "heart failure", "insuffisance cardiaque chronique"]),
        Concept("atrial_fibrillation", "Fibrillation atriale", "49436004", "I48",
                ["fibrillation atriale", "fibrillation auriculaire", "fa", "acfa", "atrial fibrillation"]),
        Concept("ckd", "Maladie rénale chronique", "709044004", "N18",
                ["maladie renale chronique", "insuffisance renale chronique", "irc", "mrc", "chronic kidney disease", "ckd"]),
        Concept("copd", "BPCO", "13645005", "J44",
                ["bpco", "bronchopneumopathie chronique obstructive", "copd"]),
        Concept("obesity", "Obésité", "414916001", "E66", ["obesite", "obesity"]),
        Concept("family_history_diabetes", "Antécédent familial de diabète", "160303001", "Z83.3",
                ["antecedent familial de diabete", "atcd familial diabete", "family history of diabetes",
                 "mere diabetique", "pere diabetique"]),
    ]
}
SNOMED_INDEX = {c.snomed: c.name for c in CONDITIONS.values() if c.snomed}

MEDICATIONS: dict[str, tuple[str, str]] = {  # nom normalisé -> (ATC, libellé)
    "metformine": ("A10BA02", "Metformine"), "metformin": ("A10BA02", "Metformine"),
    "furosemide": ("C03CA01", "Furosémide"), "bisoprolol": ("C07AB07", "Bisoprolol"),
    "ramipril": ("C09AA05", "Ramipril"), "sacubitril valsartan": ("C09DX04", "Sacubitril/valsartan"),
    "entresto": ("C09DX04", "Sacubitril/valsartan"), "atorvastatine": ("C10AA05", "Atorvastatine"),
    "empagliflozine": ("A10BK03", "Empagliflozine"), "dapagliflozine": ("A10BK01", "Dapagliflozine"),
    "spironolactone": ("C03DA01", "Spironolactone"), "apixaban": ("B01AF02", "Apixaban"),
    "amlodipine": ("C08CA01", "Amlodipine"), "insuline glargine": ("A10AE04", "Insuline glargine"),
}


# ------------------------------------------------------------------ lookups
def _best(query: str, candidates: dict[str, list[str]], cutoff: float) -> tuple[str | None, float]:
    q = norm(query)
    best, best_score = None, 0.0
    for name, syns in candidates.items():
        for s in syns + [name]:
            s_n = norm(s)
            if q == s_n:
                return name, 1.0
            score = difflib.SequenceMatcher(None, q, s_n).ratio()
            if score > best_score:
                best, best_score = name, score
    return (best, best_score) if best_score >= cutoff else (None, best_score)


def map_observable(raw_name: str, cutoff: float = 0.86) -> tuple[str | None, float]:
    """Nom brut ('Créatinine (µmol/L)') -> nom canonique ('creatinine')."""
    stripped = re.sub(r"\(.*?\)|\[.*?\]", "", raw_name)  # retire l'unité entre parenthèses
    return _best(stripped, {k: o.synonyms for k, o in OBSERVABLES.items()}, cutoff)


def map_condition(label: str, cutoff: float = 0.88) -> tuple[str | None, float]:
    return _best(label, {k: c.synonyms for k, c in CONDITIONS.items()}, cutoff)


def map_medication(label: str) -> tuple[str, str] | None:
    n = norm(label)
    for key, val in MEDICATIONS.items():
        if key in n:
            return val
    return None


def unit_in_name(raw_name: str) -> str | None:
    m = re.search(r"\(([^)]+)\)|\[([^\]]+)\]", raw_name)
    return (m.group(1) or m.group(2)).strip() if m else None


class UnitError(ValueError):
    pass


def to_canonical(name: str, value: float, unit: str | None) -> tuple[float, str]:
    """Convertit `value` exprimée en `unit` vers l'unité canonique de `name`."""
    obs = OBSERVABLES[name]
    u = norm_unit(unit)
    if not u or u == norm_unit(obs.unit):
        return float(value), obs.unit
    for src, fn in obs.conversions.items():
        if u == norm_unit(src):
            return float(fn(float(value))), obs.unit
    # équivalences d'écriture courantes
    aliases = {"mg/dl": "mg/dl", "mmhg": "mmhg", "bpm": "/min", "battements/min": "/min", "kg/m²": "kg/m2"}
    if aliases.get(u) == norm_unit(obs.unit):
        return float(value), obs.unit
    raise UnitError(f"Unité '{unit}' non convertible pour {name} (attendu {obs.unit})")


def is_plausible(name: str, value: float) -> bool:
    lo, hi = OBSERVABLES[name].plausible
    return lo <= value <= hi
