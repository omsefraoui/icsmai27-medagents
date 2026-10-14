"""Détection et pseudonymisation des informations de santé protégées (PHI).

Stratégie en couches (défense en profondeur) :
1. Identifiants *connus* : extraits des champs structurés (FHIR Patient,
   HL7 PID, DICOM PatientName...) puis recherchés dans TOUS les fichiers.
2. Expressions régulières : formats à forte structure (NIR, téléphone,
   e-mail, dates, IPP, adresses, titres + noms).
3. LLM local (dans l'agent) : entités résiduelles que les règles ratent
   (noms sans titre, établissements, lieux...). Chaque entité proposée par
   le LLM doit exister *mot pour mot* dans le texte, sinon elle est rejetée.

Pseudonymisation :
- jetons stables par HMAC-SHA256 (même entrée -> même jeton, non réversible
  sans le secret) ;
- décalage de dates cohérent par patient (préserve les intervalles) ;
- âges > 89 ans agrégés en « 90+ » (HIPAA Safe Harbor).
"""
from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from ..schemas import PHICategory

MONTHS_FR = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
             "septembre", "octobre", "novembre", "décembre"]
_MONTHS_RE = "|".join(MONTHS_FR + ["fevrier", "aout", "decembre"])

NAME_TOKEN = r"[A-ZÉÈÀÂÎÔÛÇ][A-Za-zÀ-ÖØ-öø-ÿ'\-]+"

PATTERNS: list[tuple[PHICategory, re.Pattern[str], int]] = [
    # (catégorie, motif, groupe à masquer)
    (PHICategory.EMAIL, re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), 0),
    (PHICategory.URL_IP, re.compile(r"\bhttps?://\S+|\b(?:\d{1,3}\.){3}\d{1,3}\b"), 0),
    (PHICategory.SSN, re.compile(
        r"\b[12][\s.]?\d{2}[\s.]?(?:0[1-9]|1[0-2]|[2-9]\d)[\s.]?(?:\d{2}|2A|2B)[\s.]?\d{3}[\s.]?\d{3}(?:[\s.]?\d{2})?\b"), 0),
    (PHICategory.PHONE, re.compile(r"(?<!\d)(?:\+33[\s.]?|0)[1-9](?:[\s.-]?\d{2}){4}(?!\d)"), 0),
    (PHICategory.MRN, re.compile(
        r"\b(?:IPP|NIP|MRN|N°\s*dossier|num[ée]ro\s+de\s+dossier|ID\s+patient)\s*[:#]?\s*([A-Z]{0,4}[-]?\d{5,})", re.I), 1),
    (PHICategory.DATE, re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-](?:19|20)\d{2}\b"), 0),
    (PHICategory.DATE, re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}(?!\d)"), 0),
    (PHICategory.DATE, re.compile(rf"\b\d{{1,2}}(?:er)?\s+(?:{_MONTHS_RE})\s+(?:19|20)\d{{2}}\b", re.I), 0),
    (PHICategory.ADDRESS, re.compile(
        r"\b\d{1,4}(?:\s?(?:bis|ter))?,?\s+(?:rue|avenue|av\.|boulevard|bd|place|chemin|all[ée]e|impasse|route|quai|cours)"
        r"\s+[^\n,;]{2,40}", re.I), 0),
    (PHICategory.ZIP, re.compile(rf"\b\d{{5}}\s+{NAME_TOKEN}(?:[\s-]{NAME_TOKEN})*"), 0),
    (PHICategory.NAME, re.compile(
        rf"\b(?:Dr|Docteur|Pr|Professeur|M\.|Mme|Mlle|Madame|Monsieur|Mr|Mrs|Ms)\.?[ \t]+((?:{NAME_TOKEN})(?:[ \t]+{NAME_TOKEN}){{0,2}})"), 1),
    (PHICategory.AGE_90, re.compile(r"\b(?:9\d|1[01]\d)\s*(?:ans|years?(?:\s+old)?|yo)\b", re.I), 0),
]


@dataclass(order=True)
class Span:
    start: int
    end: int
    category: PHICategory = field(compare=False)
    text: str = field(compare=False)
    detector: str = field(compare=False, default="regex")


def find_regex(text: str, categories: set[PHICategory] | None = None) -> list[Span]:
    spans: list[Span] = []
    for cat, pat, grp in PATTERNS:
        if categories and cat not in categories:
            continue
        for m in pat.finditer(text):
            s, e = m.span(grp)
            if e > s:
                spans.append(Span(s, e, cat, text[s:e], "regex"))
    return spans


def find_known(text: str, known: dict[str, PHICategory]) -> list[Span]:
    """Recherche insensible à la casse des identifiants connus (≥ 3 caractères)."""
    spans: list[Span] = []
    for value, cat in known.items():
        v = value.strip()
        if len(v) < 3:
            continue
        pat = re.compile(r"(?<![\w])" + re.escape(v) + r"(?![\w])", re.I)
        for m in pat.finditer(text):
            spans.append(Span(m.start(), m.end(), cat, m.group(0), "known_identifier"))
    return spans


def find_exact(text: str, needle: str, category: PHICategory, detector: str) -> list[Span]:
    if len(needle.strip()) < 2:
        return []
    return [Span(m.start(), m.end(), category, m.group(0), detector)
            for m in re.finditer(re.escape(needle.strip()), text)]


def merge_spans(spans: list[Span]) -> list[Span]:
    """Fusionne les chevauchements en gardant le span le plus long."""
    out: list[Span] = []
    for sp in sorted(spans, key=lambda s: (s.start, -(s.end - s.start))):
        if out and sp.start < out[-1].end:
            if sp.end > out[-1].end:
                prev = out[-1]
                out[-1] = Span(prev.start, sp.end, prev.category, prev.text + sp.text[prev.end - sp.start:],
                               prev.detector)
            continue
        out.append(sp)
    return out


# ----------------------------------------------------------- pseudonymisation
def parse_any_date(s: str) -> tuple[date, str] | None:
    s = s.strip()
    for fmt, out_fmt in (("%d/%m/%Y", "%d/%m/%Y"), ("%d-%m-%Y", "%d-%m-%Y"), ("%d.%m.%Y", "%d.%m.%Y"),
                         ("%Y-%m-%d", "%Y-%m-%d"), ("%Y%m%d", "%Y%m%d")):
        try:
            return datetime.strptime(s, fmt).date(), out_fmt
        except ValueError:
            pass
    m = re.match(rf"(\d{{1,2}})(?:er)?\s+({_MONTHS_RE})\s+(\d{{4}})", s, re.I)
    if m:
        name = m.group(2).lower().replace("fevrier", "février").replace("aout", "août").replace("decembre", "décembre")
        return date(int(m.group(3)), MONTHS_FR.index(name) + 1, int(m.group(1))), "%d/%m/%Y"
    return None


class Pseudonymizer:
    def __init__(self, secret: str, patient_key: str, max_shift_days: int = 180):
        self._secret = secret.encode()
        self.patient_key = patient_key
        digest = self._h(f"shift::{patient_key}")
        shift = int(digest[:8], 16) % (2 * max_shift_days + 1) - max_shift_days
        self.shift_days = shift if shift != 0 else 17
        self.vault: dict[str, str] = {}  # jeton -> valeur réelle (table de ré-identification)

    def _h(self, value: str) -> str:
        return hmac.new(self._secret, value.encode(), hashlib.sha256).hexdigest()

    @property
    def pseudo_id(self) -> str:
        return "PSN-" + self._h(f"patient::{self.patient_key}")[:10].upper()

    def token(self, category: PHICategory, value: str) -> str:
        tok = f"[{category.value}-{self._h(category.value + '::' + value.strip().lower())[:6]}]"
        self.vault[tok] = value
        return tok

    def shift_date(self, d: date) -> date:
        return d + timedelta(days=self.shift_days)

    def shift_date_str(self, s: str) -> str | None:
        parsed = parse_any_date(s)
        if not parsed:
            return None
        d, fmt = parsed
        return self.shift_date(d).strftime(fmt)

    def replacement(self, sp: Span) -> str:
        if sp.category == PHICategory.DATE:
            shifted = self.shift_date_str(sp.text)
            return shifted if shifted else self.token(sp.category, sp.text)
        if sp.category == PHICategory.AGE_90:
            return "90+ ans"
        return self.token(sp.category, sp.text)


def redact_text(text: str, spans: list[Span], pz: Pseudonymizer) -> tuple[str, list[tuple[Span, str]]]:
    merged = merge_spans(spans)
    out, last, applied = [], 0, []
    for sp in merged:
        rep = pz.replacement(sp)
        out.append(text[last:sp.start])
        out.append(rep)
        applied.append((sp, rep))
        last = sp.end
    out.append(text[last:])
    return "".join(out), applied


RESIDUAL_CATEGORIES = {PHICategory.EMAIL, PHICategory.SSN, PHICategory.PHONE, PHICategory.MRN,
                       PHICategory.ADDRESS, PHICategory.URL_IP, PHICategory.AGE_90}


TERMINOLOGY_URLS = ("loinc.org", "snomed.info", "hl7.org", "unitsofmeasure.org", "whocc.no", "urn:")


def residual_scan(text: str, known: dict[str, PHICategory]) -> list[str]:
    """Contrôle final : aucun identifiant fort ni identifiant connu ne doit subsister.

    Les dates ne sont pas contrôlées ici car elles sont *décalées* (pseudonymisation) ;
    les identifiants connus incluent en revanche les dates réelles (naissance...).
    """
    hits = [f"{s.category.value}: {s.text}" for s in find_regex(text, RESIDUAL_CATEGORIES)
            if not any(u in s.text for u in TERMINOLOGY_URLS)]
    hits += [f"{s.category.value}: {s.text}" for s in find_known(text, known)]
    return sorted(set(hits))


HIPAA_18 = [
    "Noms", "Subdivisions géographiques < État (adresse, code postal)", "Dates (sauf année) liées au patient",
    "Téléphones", "Fax", "E-mails", "N° sécurité sociale (NIR/SSN)", "N° de dossier médical (IPP/MRN)",
    "N° de bénéficiaire d'assurance", "N° de compte", "N° de certificat/licence", "Identifiants de véhicule",
    "Identifiants d'appareils / n° de série", "URL", "Adresses IP", "Identifiants biométriques",
    "Photos plein visage", "Tout autre identifiant unique",
]
