"""Agent 2 — Data Anonymizer Agent (Privacy Guard).

Pipeline :
1. Récolte des identifiants *connus* dans les sources structurées
   (FHIR Patient, HL7 PID/PV1, DICOM, colonnes identifiantes des tableaux).
2. Notes cliniques : regex + identifiants connus + NER par LLM local.
   Les entités LLM qui n'existent pas mot pour mot dans le texte sont
   rejetées (anti-hallucination) ; les autres enrichissent le dictionnaire
   d'identifiants connus (propagation inter-fichiers).
3. Pseudonymisation spécifique à chaque format (FHIR, HL7, CSV/Excel, DICOM).
4. Contrôle résiduel sur TOUTES les sorties -> statut PASS ou BLOCKED.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from .. import config
from ..schemas import FileType, IngestionManifest, PHICategory, PHIEntity, PHIFindingsLLM, PHIReport
from ..tools import parsers, phi
from ..tools.terminology import CONDITIONS, MEDICATIONS, OBSERVABLES, map_condition, norm
from .base import BaseAgent

SYSTEM_PROMPT = """Tu es un expert en protection des données de santé (RGPD, HIPAA).
Tu repères dans un texte clinique les informations qui permettent d'identifier une personne :
noms et prénoms (patient, proches, soignants), adresses, villes, numéros de téléphone, e-mails,
numéros d'identification (IPP, sécurité sociale, mutuelle, dossier), noms d'établissements précis.
Règles :
- copie chaque extrait EXACTEMENT comme dans le texte (même casse, mêmes accents) ;
- n'inclus PAS les pathologies, médicaments, résultats, unités, ni les dates ;
- si rien n'est trouvé, renvoie une liste vide."""

ID_COLUMN_HINTS = ["ipp", "nir", "patient_id", "id patient", "identifiant", "secu", "mrn", "dossier", "mutuelle"]
DROP_COLUMN_HINTS = ["nom", "prenom", "name", "telephone", "tel", "adresse", "address", "email", "mail",
                     "naissance", "ddn", "birth", "ville", "city", "code postal"]

DICOM_EMPTY_TAGS = ["PatientBirthDate", "ReferringPhysicianName", "OperatorsName", "InstitutionName",
                    "InstitutionAddress", "AccessionNumber", "DeviceSerialNumber", "PerformingPhysicianName",
                    "OtherPatientIDs", "PatientAddress", "PatientTelephoneNumbers", "StationName"]
DICOM_UID_TAGS = ["StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"]


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def _variants(value: str, cat: PHICategory) -> list[str]:
    """Variantes d'écriture d'un identifiant (dates, téléphones, NIR, noms)."""
    v = value.strip()
    out = {v}
    if cat == PHICategory.DATE:
        parsed = phi.parse_any_date(v)
        if parsed:
            d = parsed[0]
            out |= {d.strftime("%d/%m/%Y"), d.isoformat(), d.strftime("%Y%m%d"), d.strftime("%d.%m.%Y"),
                    d.strftime("%d-%m-%Y")}
    elif cat == PHICategory.PHONE:
        d = _digits(v)
        if len(d) == 10:
            pairs = [d[i:i + 2] for i in range(0, 10, 2)]
            out |= {d, " ".join(pairs), ".".join(pairs), "-".join(pairs)}
    elif cat == PHICategory.SSN:
        d = _digits(v)
        out.add(d)
        if len(d) >= 13:
            out.add(f"{d[0]} {d[1:3]} {d[3:5]} {d[5:7]} {d[7:10]} {d[10:13]}" + (f" {d[13:15]}" if len(d) > 13 else ""))
    elif cat == PHICategory.NAME:
        tokens = [t for t in re.split(r"[\s^,]+", v) if len(t) >= 3]
        out |= set(tokens)
        if len(tokens) >= 2:
            out |= {" ".join(tokens), " ".join(reversed(tokens))}
    return [o for o in out if len(o) >= 3]


def _is_medical_term(text: str) -> bool:
    n = norm(text)
    if map_condition(text, cutoff=0.92)[0]:
        return True
    if any(k in n for k in MEDICATIONS):
        return True
    return any(n == norm(s) for o in OBSERVABLES.values() for s in o.synonyms) or \
        any(n == norm(c.label) for c in CONDITIONS.values())


class AnonymizerAgent(BaseAgent):
    name = "Agent 2 · Privacy Guard"
    role = "Détecter et pseudonymiser les PHI, contrôler la conformité"

    def __init__(self, llm=None, verbose: bool = True, use_llm: bool = True, secret: str | None = None):
        super().__init__(llm, verbose)
        self.use_llm = use_llm
        self.secret = secret or config.PSEUDONYM_SECRET

    # ================================================================ public
    def run(self, manifest: IngestionManifest, workdir: str | Path) -> PHIReport:
        workdir = Path(workdir)
        out_dir = workdir / "deid"
        out_dir.mkdir(parents=True, exist_ok=True)
        files = [f for f in manifest.files if "anonymizer" in f.route_to]

        known, patient_key = self._harvest_known(files)
        pz = phi.Pseudonymizer(self.secret, patient_key, config.DATE_SHIFT_MAX_DAYS)
        self.log(f"{len(known)} identifiants connus récoltés dans les sources structurées", output=sorted(known))

        entities: list[PHIEntity] = []
        rejected: list[str] = []
        # 1) Notes en premier : le LLM enrichit le dictionnaire d'identifiants connus
        note_spans: dict[str, list[phi.Span]] = {}
        for f in [f for f in files if f.file_type == FileType.CLINICAL_NOTE]:
            text = parsers.read_text(Path(f.path))
            spans = phi.find_regex(text) + phi.find_known(text, known)
            if self.use_llm:
                llm_spans, rej = self._llm_ner(text)
                rejected += rej
                for sp in llm_spans:
                    for v in _variants(sp.text, sp.category):
                        known.setdefault(v, sp.category)
                spans += llm_spans
            note_spans[f.file_id] = spans

        # 2) Pseudonymisation fichier par fichier
        deid_files: dict[str, str] = {}
        residual: list[str] = []
        for f in files:
            src = Path(f.path)
            dst = out_dir / f"{f.file_id}{src.suffix.lower()}"  # le nom d'origine peut contenir des PHI
            try:
                if f.file_type == FileType.CLINICAL_NOTE:
                    text = parsers.read_text(src)
                    spans = note_spans[f.file_id] + phi.find_known(text, known)  # identifiants propagés
                    red, applied = phi.redact_text(text, spans, pz)
                    dst.write_text(red, encoding="utf-8")
                    entities += [self._entity(sp, rep, f.file_id) for sp, rep in applied]
                elif f.file_type == FileType.FHIR:
                    entities += self._fhir(src, dst, pz, known, f.file_id)
                elif f.file_type == FileType.HL7V2:
                    entities += self._hl7(src, dst, pz, known, f.file_id)
                elif f.file_type in (FileType.CSV, FileType.EXCEL):
                    dst = dst.with_suffix(".csv")
                    entities += self._table(src, dst, f.file_type, pz, known, f.file_id)
                elif f.file_type == FileType.DICOM:
                    entities += self._dicom(src, dst, pz, f.file_id)
                else:
                    text = parsers.read_text(src)
                    red, applied = phi.redact_text(text, phi.find_known(text, known), pz)
                    dst.write_text(red, encoding="utf-8")
                    entities += [self._entity(sp, rep, f.file_id) for sp, rep in applied]
                deid_files[f.file_id] = str(dst)
                residual += [f"{f.original_name} → {r}" for r in phi.residual_scan(self._as_text(dst, f.file_type), known)]
            except Exception as exc:
                residual.append(f"{f.original_name} → échec de pseudonymisation : {exc}")

        (workdir / "vault").mkdir(exist_ok=True)
        (workdir / "vault" / "table_correspondance.json").write_text(
            json.dumps({"pseudo_id": pz.pseudo_id, "date_shift_days": pz.shift_days, "tokens": pz.vault},
                       ensure_ascii=False, indent=2), "utf-8")

        counts: dict[str, int] = {}
        for e in entities:
            counts[e.category.value] = counts.get(e.category.value, 0) + 1
        report = PHIReport(
            bundle_id=manifest.bundle_id, pseudo_id=pz.pseudo_id, entities=entities, counts=counts,
            residual_phi=sorted(set(residual)), hipaa_checklist=self._hipaa(counts),
            gdpr_notes=self._gdpr_notes(), deid_files=deid_files, llm_rejected=rejected,
            status="PASS" if not residual else "BLOCKED",
        )
        if rejected:
            self.log(f"{len(rejected)} entités proposées par le LLM rejetées (absentes du texte)", "warning",
                     rejected=rejected)
        self.log(f"{len(entities)} PHI pseudonymisées {counts} — contrôle résiduel : {report.status}",
                 "ok" if report.status == "PASS" else "blocked", output=report, residual=report.residual_phi)
        return report

    # ======================================================= identifiants connus
    def _harvest_known(self, files) -> tuple[dict[str, PHICategory], str]:
        known: dict[str, PHICategory] = {}
        patient_key = ""

        def add(value: Any, cat: PHICategory):
            if value:
                for v in _variants(str(value), cat):
                    known.setdefault(v, cat)

        for f in files:
            p = Path(f.path)
            try:
                if f.file_type == FileType.FHIR:
                    for pat in parsers.fhir_resources(parsers.load_fhir(p), "Patient"):
                        for n in pat.get("name", []):
                            add(" ".join(n.get("given", []) + [n.get("family", "")]), PHICategory.NAME)
                        for ident in pat.get("identifier", []):
                            add(ident.get("value"), PHICategory.MRN)
                            patient_key = patient_key or ident.get("value", "")
                        add(pat.get("birthDate"), PHICategory.DATE)
                        for t in pat.get("telecom", []):
                            add(t.get("value"), PHICategory.EMAIL if t.get("system") == "email" else PHICategory.PHONE)
                        for a in pat.get("address", []):
                            for line in a.get("line", []):
                                add(line, PHICategory.ADDRESS)
                            add(a.get("city"), PHICategory.ADDRESS)
                            add(a.get("postalCode"), PHICategory.ZIP)
                        add(pat.get("id"), PHICategory.OTHER_ID)
                elif f.file_type == FileType.HL7V2:
                    for seg in parsers.parse_hl7(parsers.read_text(p)):
                        if seg[0] == "PID":
                            add(parsers.hl7_get(seg, 3, 1), PHICategory.MRN)
                            patient_key = patient_key or parsers.hl7_get(seg, 3, 1)
                            add(parsers.hl7_get(seg, 5).replace("^", " "), PHICategory.NAME)
                            add(parsers.hl7_get(seg, 7), PHICategory.DATE)
                            add(parsers.hl7_get(seg, 11, 1), PHICategory.ADDRESS)
                            add(parsers.hl7_get(seg, 11, 3), PHICategory.ADDRESS)
                            add(parsers.hl7_get(seg, 13), PHICategory.PHONE)
                            add(parsers.hl7_get(seg, 19), PHICategory.SSN)
                        if seg[0] == "PV1":
                            add(parsers.hl7_get(seg, 7).replace("^", " "), PHICategory.NAME)
                elif f.file_type == FileType.DICOM:
                    ds = parsers.read_dicom(p)
                    for tag in ("PatientName", "ReferringPhysicianName", "OperatorsName"):
                        add(str(ds.get(tag, "")).replace("^", " "), PHICategory.NAME)
                    add(str(ds.get("PatientID", "")), PHICategory.MRN)
                    add(str(ds.get("DeviceSerialNumber", "")), PHICategory.DEVICE_ID)
                elif f.file_type in (FileType.CSV, FileType.EXCEL):
                    df = parsers.read_csv_any(p) if f.file_type == FileType.CSV else parsers.read_excel_any(p)
                    for col in df.columns:
                        kind = self._column_kind(col)
                        if kind in ("id", "drop") and "date" not in norm(col):
                            cat = PHICategory.NAME if any(h in norm(col) for h in ("nom", "name", "prenom")) \
                                else PHICategory.MRN
                            for v in df[col].dropna().unique():
                                add(v, cat)
            except Exception as exc:
                self.log(f"Récolte impossible dans {f.original_name} : {exc}", "warning")
        return known, patient_key or "patient-inconnu"

    # ================================================================== LLM
    def _llm_ner(self, text: str) -> tuple[list[phi.Span], list[str]]:
        spans, rejected = [], []
        chunks = [text[i:i + 2500] for i in range(0, len(text), 2300)]
        for chunk in chunks:
            try:
                res = self.llm.structured(SYSTEM_PROMPT, f"Texte :\n\"\"\"\n{chunk}\n\"\"\"", PHIFindingsLLM,
                                          agent=self.name)
            except Exception as exc:
                self.log(f"NER LLM indisponible ({exc}) — couche règles seule", "warning")
                return spans, rejected
            for ent in res.entities:
                t = ent.text.strip()
                if len(t) < 2 or t not in text:
                    rejected.append(t)
                    continue
                if ent.category != "DATE" and _is_medical_term(t):
                    rejected.append(t)
                    continue
                cat = PHICategory(ent.category)
                spans += phi.find_exact(text, t, cat, "llm")
        return spans, rejected

    # ============================================================== formats
    @staticmethod
    def _entity(sp: phi.Span, rep: str, file_id: str) -> PHIEntity:
        return PHIEntity(category=sp.category, text=sp.text, file_id=file_id, detector=sp.detector,
                         start=sp.start, end=sp.end, replacement=rep)

    def _fhir(self, src: Path, dst: Path, pz: phi.Pseudonymizer, known, file_id) -> list[PHIEntity]:
        bundle = parsers.load_fhir(src)
        ents: list[PHIEntity] = []
        for pat in parsers.fhir_resources(bundle, "Patient"):
            for n in pat.pop("name", []):
                txt = " ".join(n.get("given", []) + [n.get("family", "")])
                ents.append(PHIEntity(category=PHICategory.NAME, text=txt, file_id=file_id, detector="structured",
                                      replacement="[supprimé]"))
            for t in pat.pop("telecom", []):
                ents.append(PHIEntity(category=PHICategory.PHONE if t.get("system") == "phone" else PHICategory.EMAIL,
                                      text=t.get("value", ""), file_id=file_id, detector="structured",
                                      replacement="[supprimé]"))
            addrs = pat.pop("address", [])
            if addrs:
                ents.append(PHIEntity(category=PHICategory.ADDRESS, text=str(addrs), file_id=file_id,
                                      detector="structured", replacement="[supprimé]"))
                pat["address"] = [{"country": a.get("country", "")} for a in addrs if a.get("country")]
            ids = pat.pop("identifier", [])
            for i in ids:
                ents.append(PHIEntity(category=PHICategory.MRN, text=i.get("value", ""), file_id=file_id,
                                      detector="structured", replacement=pz.pseudo_id))
            pat["identifier"] = [{"system": "urn:atelier:pseudo", "value": pz.pseudo_id}]
            if pat.get("birthDate"):
                shifted = pz.shift_date_str(pat["birthDate"])
                ents.append(PHIEntity(category=PHICategory.DATE, text=pat["birthDate"], file_id=file_id,
                                      detector="structured", replacement=shifted or ""))
                pat["birthDate"] = shifted
            old_id = pat.get("id")
            if old_id:
                known.setdefault(old_id, PHICategory.OTHER_ID)
        raw = json.dumps(bundle, ensure_ascii=False, indent=2)
        # remplacement des références (Patient/pat-xxx) et dates cliniques décalées
        spans = phi.find_known(raw, known) + phi.find_regex(raw, {PHICategory.DATE})
        red, applied = phi.redact_text(raw, spans, pz)
        red = re.sub(r'\[AUTRE_ID-[0-9a-f]{6}\]', pz.pseudo_id, red)
        json.loads(red)  # garantit que le JSON reste valide
        dst.write_text(red, encoding="utf-8")
        return ents + [self._entity(sp, rep, file_id) for sp, rep in applied]

    def _hl7(self, src: Path, dst: Path, pz: phi.Pseudonymizer, known, file_id) -> list[PHIEntity]:
        segs = parsers.parse_hl7(parsers.read_text(src))
        ents: list[PHIEntity] = []

        def blank(seg, idx, cat, rep=""):
            if idx < len(seg) and seg[idx]:
                ents.append(PHIEntity(category=cat, text=seg[idx], file_id=file_id, detector="structured",
                                      replacement=rep or "[vide]"))
                seg[idx] = rep

        for seg in segs:
            if seg[0] == "PID":
                blank(seg, 3, PHICategory.MRN, pz.pseudo_id)
                blank(seg, 5, PHICategory.NAME, "PSEUDO^PATIENT")
                if len(seg) > 7 and seg[7]:
                    d = phi.parse_any_date(seg[7][:8])
                    blank(seg, 7, PHICategory.DATE, pz.shift_date(d[0]).strftime("%Y%m%d") if d else "")
                for idx, cat in ((2, PHICategory.MRN), (4, PHICategory.MRN), (11, PHICategory.ADDRESS),
                                 (13, PHICategory.PHONE), (14, PHICategory.PHONE), (18, PHICategory.OTHER_ID),
                                 (19, PHICategory.SSN), (20, PHICategory.OTHER_ID)):
                    blank(seg, idx, cat)
            elif seg[0] == "PV1":
                blank(seg, 7, PHICategory.NAME, "")
            for i, field in enumerate(seg):  # horodatages AAAAMMJJ[hhmmss]
                if seg[0] == "PID" and i == 7:
                    continue
                m = re.fullmatch(r"((?:19|20)\d{6})(\d{4,6})?", field)
                if m:
                    d = phi.parse_any_date(m.group(1))
                    if d:
                        seg[i] = pz.shift_date(d[0]).strftime("%Y%m%d") + (m.group(2) or "")
        text = "\r".join("|".join(s) for s in segs) + "\r"
        strong = {PHICategory.SSN, PHICategory.PHONE, PHICategory.EMAIL}
        red, applied = phi.redact_text(text, phi.find_known(text, known) + phi.find_regex(text, strong), pz)
        dst.write_text(red, encoding="utf-8")
        return ents + [self._entity(sp, rep, file_id) for sp, rep in applied]

    @staticmethod
    def _column_kind(col: str) -> str:
        n = norm(col)
        if any(h in n for h in ID_COLUMN_HINTS):
            return "id"
        if any(re.search(rf"\b{re.escape(h)}\b", n) for h in DROP_COLUMN_HINTS):
            return "drop"
        if "date" in n:
            return "date"
        return "keep"

    def _table(self, src: Path, dst: Path, ft: FileType, pz, known, file_id) -> list[PHIEntity]:
        df = parsers.read_csv_any(src) if ft == FileType.CSV else parsers.read_excel_any(src)
        ents: list[PHIEntity] = []
        for col in list(df.columns):
            kind = self._column_kind(col)
            if kind == "id":
                ents.append(PHIEntity(category=PHICategory.MRN, text=f"colonne « {col} »", file_id=file_id,
                                      detector="structured", replacement=pz.pseudo_id))
                df[col] = pz.pseudo_id
                df = df.rename(columns={col: "pseudo_id"})
            elif kind == "drop":
                ents.append(PHIEntity(category=PHICategory.NAME, text=f"colonne « {col} »", file_id=file_id,
                                      detector="structured", replacement="[colonne supprimée]"))
                df = df.drop(columns=[col])
            elif kind == "date":
                df[col] = df[col].map(lambda v: pz.shift_date_str(str(v)) or v if pd.notna(v) else v)
        for col in df.columns:  # identifiants connus éventuellement présents dans les cellules
            df[col] = df[col].map(lambda v: phi.redact_text(str(v), phi.find_known(str(v), known), pz)[0]
                                  if isinstance(v, str) else v)
        if "pseudo_id" in df.columns and list(df.columns).count("pseudo_id") > 1:
            df = df.loc[:, ~df.columns.duplicated()]
        df.to_csv(dst, sep=";", decimal=",", index=False)
        return ents

    def _dicom(self, src: Path, dst: Path, pz: phi.Pseudonymizer, file_id: str) -> list[PHIEntity]:
        from pydicom.uid import generate_uid

        ds = parsers.read_dicom(src)
        ents: list[PHIEntity] = []
        for tag in ("PatientName", "PatientID"):
            if tag in ds:
                ents.append(PHIEntity(category=PHICategory.NAME if tag == "PatientName" else PHICategory.MRN,
                                      text=str(ds.get(tag)), file_id=file_id, detector="dicom_tag",
                                      replacement=pz.pseudo_id))
                setattr(ds, tag, pz.pseudo_id)
        for tag in DICOM_EMPTY_TAGS:
            if tag in ds and str(ds.get(tag)):
                cat = PHICategory.DEVICE_ID if "Serial" in tag else PHICategory.DATE if "Date" in tag \
                    else PHICategory.NAME
                ents.append(PHIEntity(category=cat, text=str(ds.get(tag)), file_id=file_id, detector="dicom_tag",
                                      replacement="[vide]"))
                setattr(ds, tag, "")
        for tag in ("StudyDate", "SeriesDate", "AcquisitionDate", "ContentDate"):
            if tag in ds and str(ds.get(tag)):
                d = phi.parse_any_date(str(ds.get(tag)))
                if d:
                    setattr(ds, tag, pz.shift_date(d[0]).strftime("%Y%m%d"))
        for tag in DICOM_UID_TAGS:
            if tag in ds:
                setattr(ds, tag, generate_uid(entropy_srcs=[self.secret, str(ds.get(tag))]))
        ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
        ds.remove_private_tags()
        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = "PS3.15 Basic Profile (simplified) + date shift"
        ds.save_as(str(dst), enforce_file_format=True)
        return ents

    @staticmethod
    def _as_text(path: Path, ft: FileType) -> str:
        if ft == FileType.DICOM:
            ds = parsers.read_dicom(path)
            return "\n".join(f"{el.keyword}={el.value}" for el in ds
                             if el.keyword != "PixelData" and not el.keyword.endswith("UID"))
        return parsers.read_text(path)

    # =========================================================== conformité
    @staticmethod
    def _hipaa(counts: dict[str, int]) -> dict[str, str]:
        mapping = {
            "Noms": "NOM", "Subdivisions géographiques < État (adresse, code postal)": "ADRESSE",
            "Téléphones": "TELEPHONE", "E-mails": "EMAIL", "N° sécurité sociale (NIR/SSN)": "NUM_SECU",
            "N° de dossier médical (IPP/MRN)": "IPP", "Identifiants d'appareils / n° de série": "ID_APPAREIL",
            "URL": "URL_IP", "Adresses IP": "URL_IP", "Tout autre identifiant unique": "AUTRE_ID",
        }
        out = {}
        for item in phi.HIPAA_18:
            if item.startswith("Dates"):
                out[item] = ("décalées (cohérence temporelle préservée) — pseudonymisation RGPD ; "
                             "Safe Harbor exigerait de ne garder que l'année")
            elif item in mapping:
                n = counts.get(mapping[item], 0) + (counts.get("CODE_POSTAL", 0) if mapping[item] == "ADRESSE" else 0)
                out[item] = f"traité ({n})" if n else "non détecté"
            elif item.startswith("Photos"):
                out[item] = "non applicable (pas de photographie) — vérifier le texte incrusté des images"
            else:
                out[item] = "non détecté"
        return out

    @staticmethod
    def _gdpr_notes() -> list[str]:
        return [
            "Données PSEUDONYMISÉES : elles restent des données personnelles de santé (RGPD art. 4(5) et 9).",
            "Table de correspondance stockée séparément (vault/) : accès restreint, chiffrement attendu en production.",
            "Minimisation : colonnes identifiantes supprimées ou remplacées par un pseudonyme stable (HMAC).",
            "Traitement 100 % local (LLM via Ollama) : aucun transfert de données vers un tiers ni hors UE.",
            "Base légale à documenter (art. 9(2)(h) soins ou 9(2)(j) recherche) + AIPD requise.",
        ]
