"""Agent 3 — Feature Extraction Agent.

Transforme les fichiers dé-identifiés en un espace de features canonique :
- sources structurées (FHIR, HL7, CSV/Excel) : parsing déterministe ;
- comptes rendus texte : extraction par LLM *avec citation obligatoire*,
  puis vérification que la citation existe et contient la valeur ;
- normalisation terminologique (LOINC, SNOMED-CT, CIM-10, ATC) et
  conversion d'unités ; les libellés inconnus sont confiés au LLM avec une
  liste FERMÉE de codes possibles (Literal dynamique) ;
- fusion multi-sources avec traçabilité et détection de conflits.
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, create_model

from ..schemas import (ClinicalFeature, CodedConcept, FeatureConflict, FileType, IngestionManifest,
                       NoteExtractionLLM, PatientFeatures, PHIReport)
from ..tools import parsers, phi, terminology as term
from ..tools.clinical import age_at
from .base import BaseAgent

SYSTEM_PROMPT = """Tu es un agent d'abstraction clinique. À partir d'un compte rendu médical, tu extrais :
1. les MESURES chiffrées (biologie, constantes, échographie, scores) ;
2. les PATHOLOGIES mentionnées, avec negated=true si le texte dit que le patient ne l'a PAS ;
3. les MÉDICAMENTS du traitement.
Règles strictes :
- n'invente rien ; chaque élément a une citation `evidence` copiée MOT POUR MOT du texte ;
- pression artérielle « 120/80 » -> deux mesures : « PA systolique » (120) et « PA diastolique » (80) ;
- classe NYHA -> mesure « classe NYHA » avec une valeur de 1 à 4 ;
- nombre d'hospitalisations des 12 derniers mois -> mesure « hospitalisations 12 mois » ;
- FEVG / fraction d'éjection -> mesure « FEVG » en % ;
- poids, taille, âge -> mesures « poids », « taille », « âge » ;
- les doses de médicaments ne sont PAS des mesures ;
- un antécédent FAMILIAL (mère, père, fratrie) n'est PAS une pathologie du patient : libelle-le
  « antécédent familial de <pathologie> »."""

FAMILY_WORDS = re.compile(r"\b(m[eè]re|p[eè]re|fr[eè]re|s(?:oe|œ)ur|famil\w*|mother|father|sister|brother|parent)", re.I)

MAPPING_PROMPT = """Tu associes des libellés d'examens à un vocabulaire contrôlé.
Pour chaque libellé, choisis le code canonique correspondant dans la liste autorisée,
ou "AUCUN" si aucun ne correspond avec certitude."""

NUMBER_WORDS = {
    0: ["zero", "aucun", "aucune", "none", "no"], 1: ["un", "une", "one", "single", "i"],
    2: ["deux", "two", "ii", "twice"], 3: ["trois", "three", "iii"], 4: ["quatre", "four", "iv"],
    5: ["cinq", "five", "v"], 6: ["six"], 7: ["sept", "seven"], 8: ["huit", "eight"], 9: ["neuf", "nine"],
    10: ["dix", "ten"],
}
SOURCE_PRIORITY = {"human": 4, "structured": 3, "derived": 2, "llm_extraction": 1}


def value_supported(value: float, evidence: str) -> bool:
    """La valeur figure-t-elle dans la citation (chiffres, décimale FR, chiffres romains, lettres) ?"""
    ev = term.norm(evidence)
    raw = evidence.replace(",", ".")
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", raw)]
    if any(abs(n - value) < 1e-6 for n in nums):
        return True
    if float(value).is_integer() and int(value) in NUMBER_WORDS:
        return any(re.search(rf"\b{w}\b", ev) for w in NUMBER_WORDS[int(value)])
    # valeurs converties (ex. taille 1,65 m -> 165) : accepter un facteur 10/100/1000
    return any(n and abs(value / n - k) < 1e-6 for n in nums for k in (10, 100, 1000, 0.1, 0.01))


def evidence_in_text(evidence: str, text: str) -> bool:
    squash = lambda s: re.sub(r"\s+", " ", s).strip().lower()  # noqa: E731
    return len(evidence.strip()) >= 2 and squash(evidence) in squash(text)


def _date(s: Any) -> date | None:
    if not s:
        return None
    s = str(s)
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s[: len(datetime(2000, 1, 1).strftime(fmt))], fmt).date()
        except ValueError:
            continue
    parsed = phi.parse_any_date(s[:10])
    return parsed[0] if parsed else None


class FeatureExtractionAgent(BaseAgent):
    name = "Agent 3 · Feature Extraction"
    role = "Structurer, coder et normaliser les données cliniques"

    def __init__(self, llm=None, verbose: bool = True, use_llm: bool = True):
        super().__init__(llm, verbose)
        self.use_llm = use_llm

    # ================================================================ public
    def extract_text(self, text: str, source: str = "note_test.txt") -> PatientFeatures:
        """Extraction sur un texte libre isolé (pratique pour tester les garde-fous)."""
        self._reset()
        pending = self._from_note_text(text, source)
        if pending:
            self._resolve_labels(pending)
        return PatientFeatures(pseudo_id="TEST", features=self._merge(), conditions=self._dedupe(self._conditions),
                               medications=self._dedupe(self._meds), conflicts=self._conflicts,
                               unmapped=sorted(set(self._unmapped)), llm_rejected=self._rejected)

    def _reset(self) -> None:
        self._obs: dict[str, list[ClinicalFeature]] = defaultdict(list)
        self._conditions: list[CodedConcept] = []
        self._meds: list[CodedConcept] = []
        self._unmapped: list[str] = []
        self._rejected: list[str] = []
        self._conflicts: list[FeatureConflict] = []
        self._birth: date | None = None
        self._dates: list[date] = []

    def run(self, manifest: IngestionManifest, phi_report: PHIReport) -> PatientFeatures:
        self._reset()
        self._obs: dict[str, list[ClinicalFeature]] = defaultdict(list)
        self._conditions: list[CodedConcept] = []
        self._meds: list[CodedConcept] = []
        self._unmapped: list[str] = []
        self._rejected: list[str] = []
        self._conflicts: list[FeatureConflict] = []
        self._birth: date | None = None
        self._dates: list[date] = []

        types = {f.file_id: f.file_type for f in manifest.files}
        pending_labels: list[tuple[str, float, str | None, date | None, str]] = []
        for fid, path in phi_report.deid_files.items():
            ft, p = types[fid], Path(path)
            name = p.name
            try:
                if ft == FileType.FHIR:
                    self._from_fhir(p, name)
                elif ft == FileType.HL7V2:
                    self._from_hl7(p, name)
                elif ft in (FileType.CSV, FileType.EXCEL):
                    pending_labels += self._from_table(p, name)
                elif ft == FileType.CLINICAL_NOTE and self.use_llm:
                    pending_labels += self._from_note(p, name)
            except Exception as exc:
                self.log(f"Extraction impossible dans {name} : {exc}", "warning")

        if pending_labels:
            self._resolve_labels(pending_labels)

        features = self._merge()
        ref = max(self._dates) if self._dates else None
        if self._birth and ref and "age" not in {k for k, v in features.items() if v.source_kind == "structured"}:
            age = age_at(self._birth, ref)
            ev = "date de naissance (pseudonymisée) et date de référence"
            if age > 89:
                age, ev = 90, ev + " — âge agrégé à 90 (règle HIPAA > 89 ans)"
            prev = features.get("age")
            if prev and abs(float(prev.value) - age) > 1:
                self._conflicts.append(FeatureConflict(name="age", values=[str(prev.value), str(age)],
                                                       resolution="âge calculé retenu"))
            features["age"] = ClinicalFeature(name="age", value=float(age), unit="a", code_system="LOINC",
                                              code="30525-0", display="Âge", observed_at=ref,
                                              source_file="(calcul)", source_kind="derived", evidence=ev)
        for name, f in features.items():
            if isinstance(f.value, (int, float)) and not isinstance(f.value, bool) and name in term.OBSERVABLES \
                    and not term.is_plausible(name, float(f.value)):
                f.confidence = 0.1
                self._conflicts.append(FeatureConflict(
                    name=name, values=[str(f.value)],
                    resolution="valeur physiologiquement impossible — conservée pour revue humaine"))

        pf = PatientFeatures(pseudo_id=phi_report.pseudo_id, reference_date=ref, features=features,
                             conditions=self._dedupe(self._conditions), medications=self._dedupe(self._meds),
                             conflicts=self._conflicts, unmapped=sorted(set(self._unmapped)),
                             llm_rejected=self._rejected)
        self.log(f"{len(features)} features, {len(pf.conditions)} pathologies, {len(pf.medications)} traitements "
                 f"— {len(pf.conflicts)} conflit(s), {len(pf.llm_rejected)} rejet(s) LLM",
                 "warning" if pf.conflicts or pf.llm_rejected else "ok", output=pf)
        return pf

    # ============================================================ structurés
    def _add(self, name: str, value: float, unit: str | None, when: date | None, source: str, kind: str,
             evidence: str | None = None, raw_label: str | None = None) -> None:
        try:
            v, u = term.to_canonical(name, value, unit)
        except term.UnitError as exc:
            self._unmapped.append(f"{raw_label or name} : {exc}")
            return
        obs = term.OBSERVABLES[name]
        if when:
            self._dates.append(when)
        if name == "nyha_class" or name == "prior_admissions_12m":
            v = float(round(v))
        self._obs[name].append(ClinicalFeature(
            name=name, value=round(v, 2), unit=u, code_system="LOINC" if obs.loinc else None, code=obs.loinc,
            display=obs.display, observed_at=when, source_file=source, source_kind=kind, evidence=evidence,
            confidence=1.0 if kind == "structured" else 0.8))

    def _add_condition(self, cname: str, source: str, kind: str, negated=False, evidence=None, label=None):
        c = term.CONDITIONS[cname]
        self._conditions.append(CodedConcept(name=cname, label=label or c.label, snomed=c.snomed, icd10=c.icd10,
                                             negated=negated, source_file=source, source_kind=kind,
                                             evidence=evidence))

    def _from_fhir(self, p: Path, src: str) -> None:
        b = parsers.load_fhir(p)
        for pat in parsers.fhir_resources(b, "Patient"):
            self._birth = _date(pat.get("birthDate")) or self._birth
            if pat.get("gender") in ("male", "female"):
                self._obs["sex_female"].append(ClinicalFeature(
                    name="sex_female", value=pat["gender"] == "female", source_file=src, source_kind="structured",
                    display="Sexe féminin", evidence=f"Patient.gender = {pat['gender']}"))
        for o in parsers.fhir_resources(b, "Observation"):
            codings = o.get("code", {}).get("coding", [])
            loinc = next((c["code"] for c in codings if "loinc" in c.get("system", "")), None)
            name = term.LOINC_INDEX.get(loinc)
            q = o.get("valueQuantity")
            if not name or not q:
                self._unmapped.append(f"FHIR Observation {loinc or o.get('id')}")
                continue
            self._add(name, float(q["value"]), q.get("unit"), _date(o.get("effectiveDateTime")), src, "structured",
                      evidence=f"Observation LOINC {loinc}")
        for c in parsers.fhir_resources(b, "Condition"):
            for cd in c.get("code", {}).get("coding", []):
                cname = term.SNOMED_INDEX.get(cd.get("code"))
                if cname:
                    self._add_condition(cname, src, "structured", evidence=f"Condition SNOMED {cd['code']}")
        for m in parsers.fhir_resources(b, "MedicationStatement"):
            cc = m.get("medicationCodeableConcept", {})
            atc = next((c.get("code") for c in cc.get("coding", []) if "whocc" in c.get("system", "")), None)
            self._meds.append(CodedConcept(name=term.norm(cc.get("text", "")), label=cc.get("text", ""), atc=atc,
                                           source_file=src, source_kind="structured"))

    def _from_hl7(self, p: Path, src: str) -> None:
        for seg in parsers.parse_hl7(parsers.read_text(p)):
            if seg[0] == "PID":
                self._birth = _date(parsers.hl7_get(seg, 7)) or self._birth
                sex = parsers.hl7_get(seg, 8)
                if sex in ("M", "F"):
                    self._obs["sex_female"].append(ClinicalFeature(
                        name="sex_female", value=sex == "F", source_file=src, source_kind="structured",
                        display="Sexe féminin", evidence=f"PID-8 = {sex}"))
            if seg[0] == "OBX":
                loinc = parsers.hl7_get(seg, 3, 1)
                name = term.LOINC_INDEX.get(loinc) or term.map_observable(parsers.hl7_get(seg, 3, 2))[0]
                val = parsers.to_number(parsers.hl7_get(seg, 5))
                if not name or val is None:
                    self._unmapped.append(f"HL7 OBX {parsers.hl7_get(seg, 3)}")
                    continue
                self._add(name, val, parsers.hl7_get(seg, 6), _date(parsers.hl7_get(seg, 14)), src, "structured",
                          evidence=f"OBX {parsers.hl7_get(seg, 3)}")

    def _from_table(self, p: Path, src: str) -> list[tuple]:
        df = parsers.read_csv_any(p)
        cols = {c: term.norm(c) for c in df.columns}
        label_col = next((c for c, n in cols.items() if re.search(r"examen|analyse|test|parametre|libelle", n)), None)
        value_col = next((c for c, n in cols.items() if re.search(r"resultat|valeur|value|result", n)), None)
        unit_col = next((c for c, n in cols.items() if re.search(r"unite|unit", n)), None)
        date_col = next((c for c, n in cols.items() if "date" in n), None)
        pending = []
        if label_col and value_col:  # format long : une ligne = une mesure
            for _, row in df.iterrows():
                val = parsers.to_number(row[value_col])
                if val is None:
                    continue
                label = str(row[label_col])
                unit = row[unit_col] if unit_col else term.unit_in_name(label)
                when = _date(row[date_col]) if date_col else None
                name, _ = term.map_observable(label)
                if name:
                    self._add(name, val, unit, when, src, "structured", evidence=f"{label} = {row[value_col]} {unit or ''}",
                              raw_label=label)
                else:
                    pending.append((label, val, unit, when, src))
        else:  # format large : une colonne = une mesure
            for _, row in df.iterrows():
                when = _date(row[date_col]) if date_col else None
                for col in df.columns:
                    if col in (date_col, "pseudo_id"):
                        continue
                    val = parsers.to_number(row[col])
                    if val is None:
                        continue
                    name, _ = term.map_observable(col)
                    unit = term.unit_in_name(col)
                    if name:
                        self._add(name, val, unit, when, src, "structured", evidence=f"{col} = {row[col]}",
                                  raw_label=col)
                    else:
                        pending.append((col, val, unit, when, src))
        return pending

    # ================================================================ notes
    def _from_note(self, p: Path, src: str) -> list[tuple]:
        return self._from_note_text(parsers.read_text(p), src)

    def _from_note_text(self, text: str, src: str) -> list[tuple]:
        res = self.llm.structured(SYSTEM_PROMPT, f"Compte rendu :\n\"\"\"\n{text}\n\"\"\"", NoteExtractionLLM,
                                  agent=self.name)
        pending = []
        for m in res.measurements:
            if not evidence_in_text(m.evidence, text) or not value_supported(m.value, m.evidence):
                self._rejected.append(f"{m.name}={m.value} (citation « {m.evidence[:60]} » non retrouvée)")
                continue
            name, _ = term.map_observable(m.name)
            if name:
                self._add(name, m.value, m.unit or term.OBSERVABLES[name].unit, None, src, "llm_extraction",
                          evidence=m.evidence, raw_label=m.name)
            else:
                pending.append((m.name, m.value, m.unit, None, src + "|" + m.evidence))
        for c in res.conditions:
            if not evidence_in_text(c.evidence, text):
                self._rejected.append(f"pathologie {c.label} (citation non retrouvée)")
                continue
            cname, _ = term.map_condition(c.label)
            if cname and cname != "family_history_diabetes" and FAMILY_WORDS.search(c.evidence + " " + c.label):
                # garde-fou : un antécédent familial ne doit pas devenir une pathologie du patient
                cname = "family_history_diabetes" if cname == "diabetes" else None
                if cname is None:
                    self._unmapped.append(f"antécédent familial : {c.label}")
                    continue
            if cname:
                self._add_condition(cname, src, "llm_extraction", c.negated, c.evidence, c.label)
            else:
                self._unmapped.append(f"pathologie : {c.label}")
        for med in res.medications:
            mapped = term.map_medication(med.name)
            if mapped and evidence_in_text(med.evidence, text):
                self._meds.append(CodedConcept(name=term.norm(mapped[1]), label=mapped[1], atc=mapped[0],
                                               source_file=src, source_kind="llm_extraction", evidence=med.evidence))
        return pending

    # ================================================= mapping contraint LLM
    def _resolve_labels(self, pending: list[tuple]) -> None:
        labels = sorted({p[0] for p in pending})
        if not self.use_llm:
            self._unmapped += labels
            return
        allowed = tuple(sorted(term.OBSERVABLES)) + ("AUCUN",)
        Item = create_model("LabelMapping", raw=(str, ...), canonical=(Literal[allowed], ...))  # type: ignore
        Batch = create_model("LabelMappings", mappings=(list[Item], Field(default_factory=list)))  # type: ignore
        menu = "\n".join(f"- {k} : {o.display} ({o.unit})" for k, o in term.OBSERVABLES.items())
        try:
            res = self.llm.structured(MAPPING_PROMPT, f"Vocabulaire autorisé :\n{menu}\n\nLibellés à associer :\n"
                                      + "\n".join(f"- {lab}" for lab in labels), Batch, agent=self.name)
            mapping = {m.raw: m.canonical for m in res.mappings if m.raw in labels}
        except Exception as exc:
            self.log(f"Mapping LLM indisponible : {exc}", "warning")
            mapping = {}
        for label, val, unit, when, src in pending:
            canon = mapping.get(label, "AUCUN")
            source, _, evidence = src.partition("|")
            if canon == "AUCUN":
                self._unmapped.append(label)
                continue
            self._add(canon, val, unit, when, source, "llm_extraction" if evidence else "structured",
                      evidence=(evidence or f"{label} = {val}") + f" [libellé associé par LLM → {canon}]",
                      raw_label=label)

    # ================================================================ fusion
    def _merge(self) -> dict[str, ClinicalFeature]:
        out: dict[str, ClinicalFeature] = {}
        for name, obs in self._obs.items():
            ranked = sorted(obs, key=lambda f: (f.observed_at or date.min, SOURCE_PRIORITY[f.source_kind]),
                            reverse=True)
            best = ranked[0]
            if any(f.source_kind == "structured" for f in obs) and best.source_kind != "structured" \
                    and best.observed_at is None:
                best = next(f for f in ranked if f.source_kind == "structured")
            vals = {str(f.value) for f in obs}
            if len(vals) > 1 and not isinstance(best.value, bool):
                nums = [float(f.value) for f in obs]
                if max(nums) - min(nums) > 0.05 * max(abs(x) for x in nums):
                    self._conflicts.append(FeatureConflict(
                        name=name, values=[f"{f.value} ({f.source_kind}, {f.source_file}, {f.observed_at or 's.d.'})"
                                           for f in obs],
                        resolution=f"retenu : {best.value} (mesure la plus récente / source structurée prioritaire)"))
            out[name] = best
        return out

    @staticmethod
    def _dedupe(items: list[CodedConcept]) -> list[CodedConcept]:
        seen: dict[str, CodedConcept] = {}
        for c in sorted(items, key=lambda c: c.source_kind != "structured"):
            key = c.name
            if key not in seen:
                seen[key] = c
            elif seen[key].negated != c.negated and c.source_kind == "structured":
                seen[key] = c
        return list(seen.values())
