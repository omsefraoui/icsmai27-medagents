"""Contrats de données (Pydantic v2) échangés entre agents.

Deux familles :
- les *contrats inter-agents* (manifeste, rapport PHI, features, ...) ;
- les *schémas de sortie LLM* (suffixe « LLM »), volontairement plats et
  courts : les petits modèles locaux respectent mieux les schémas simples.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# =========================================================================
# Agent 1 — Ingestion
# =========================================================================
class FileType(str, Enum):
    CSV = "csv"
    EXCEL = "excel"
    FHIR = "fhir_json"
    HL7V2 = "hl7v2"
    DICOM = "dicom"
    CLINICAL_NOTE = "clinical_note"
    JSON = "json"
    UNKNOWN = "unknown"


class IngestedFile(BaseModel):
    file_id: str
    path: str
    original_name: str
    file_type: FileType
    detection_method: Literal["magic_bytes", "content", "extension", "llm"] = "content"
    confidence: float = Field(ge=0, le=1)
    size_bytes: int
    sha256: str
    from_archive: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_valid: bool = True
    validation_errors: list[str] = Field(default_factory=list)
    route_to: list[str] = Field(default_factory=list)


class IngestionManifest(BaseModel):
    bundle_id: str
    source: str
    created_at: str = Field(default_factory=now_iso)
    files: list[IngestedFile]
    warnings: list[str] = Field(default_factory=list)

    def by_type(self, ft: FileType) -> list[IngestedFile]:
        return [f for f in self.files if f.file_type == ft]


class NoteClassificationLLM(BaseModel):
    """Sortie LLM : nature d'un document texte libre."""

    document_type: Literal[
        "compte_rendu_consultation",
        "compte_rendu_hospitalisation",
        "compte_rendu_imagerie",
        "ordonnance",
        "courrier",
        "autre",
    ]
    specialite: str = Field(description="Spécialité médicale principale, ex. cardiologie")
    langue: Literal["fr", "en", "autre"]
    resume: str = Field(description="Une phrase, sans aucune donnée identifiante")


# =========================================================================
# Agent 2 — Anonymisation
# =========================================================================
class PHICategory(str, Enum):
    """Sous-ensemble opérationnel des 18 identifiants HIPAA Safe Harbor."""

    NAME = "NOM"
    DATE = "DATE"
    PHONE = "TELEPHONE"
    EMAIL = "EMAIL"
    SSN = "NUM_SECU"  # NIR français / SSN US
    MRN = "IPP"  # identifiant patient hospitalier
    ADDRESS = "ADRESSE"
    ZIP = "CODE_POSTAL"
    AGE_90 = "AGE_SUP_89"
    URL_IP = "URL_IP"
    DEVICE_ID = "ID_APPAREIL"
    OTHER_ID = "AUTRE_ID"


class PHIEntity(BaseModel):
    category: PHICategory
    text: str
    file_id: str
    detector: Literal["structured", "regex", "known_identifier", "llm", "dicom_tag"]
    start: int | None = None
    end: int | None = None
    replacement: str = ""


class PHIEntityLLM(BaseModel):
    text: str = Field(description="Extrait EXACT copié depuis le texte")
    category: Literal["NOM", "ADRESSE", "TELEPHONE", "EMAIL", "IPP", "NUM_SECU", "DATE", "AUTRE_ID"]


class PHIFindingsLLM(BaseModel):
    entities: list[PHIEntityLLM] = Field(default_factory=list)


class PHIReport(BaseModel):
    bundle_id: str
    pseudo_id: str
    entities: list[PHIEntity]
    counts: dict[str, int]
    residual_phi: list[str]
    hipaa_checklist: dict[str, str]
    gdpr_notes: list[str]
    deid_files: dict[str, str]  # file_id -> chemin du fichier dé-identifié
    llm_rejected: list[str] = Field(default_factory=list)  # hallucinations filtrées
    status: Literal["PASS", "BLOCKED"]


# =========================================================================
# Agent 3 — Extraction de features
# =========================================================================
class ClinicalFeature(BaseModel):
    name: str  # nom canonique (cf. tools/terminology.py)
    value: float | str | bool
    unit: str | None = None
    code_system: str | None = None  # LOINC, SNOMED-CT, ...
    code: str | None = None
    display: str | None = None
    observed_at: date | None = None
    source_file: str
    source_kind: Literal["structured", "llm_extraction", "derived", "human"]
    evidence: str | None = None
    confidence: float = 1.0


class CodedConcept(BaseModel):
    name: str
    label: str
    snomed: str | None = None
    icd10: str | None = None
    atc: str | None = None
    negated: bool = False
    source_file: str
    source_kind: Literal["structured", "llm_extraction"]
    evidence: str | None = None


class FeatureConflict(BaseModel):
    name: str
    values: list[str]
    resolution: str


class PatientFeatures(BaseModel):
    pseudo_id: str
    reference_date: date | None = None
    features: dict[str, ClinicalFeature]
    conditions: list[CodedConcept] = Field(default_factory=list)
    medications: list[CodedConcept] = Field(default_factory=list)
    conflicts: list[FeatureConflict] = Field(default_factory=list)
    unmapped: list[str] = Field(default_factory=list)
    llm_rejected: list[str] = Field(default_factory=list)

    def has_condition(self, name: str) -> bool:
        return any(c.name == name and not c.negated for c in self.conditions)

    def flat(self) -> dict[str, Any]:
        """Vue plate {feature: valeur} + conditions en booléens."""
        out: dict[str, Any] = {k: f.value for k, f in self.features.items()}
        for c in self.conditions:
            out.setdefault(c.name, not c.negated)
        return out


class MeasurementLLM(BaseModel):
    name: str = Field(description="Nom de la mesure tel qu'écrit dans le texte")
    value: float
    unit: str = Field(default="", description="Unité telle qu'écrite, vide si absente")
    evidence: str = Field(description="Courte citation EXACTE du texte contenant la valeur")


class ConditionLLM(BaseModel):
    label: str = Field(description="Pathologie telle qu'écrite")
    negated: bool = Field(description="true si le texte dit que le patient N'A PAS cette pathologie")
    evidence: str


class MedicationLLM(BaseModel):
    name: str
    evidence: str


class NoteExtractionLLM(BaseModel):
    measurements: list[MeasurementLLM] = Field(default_factory=list)
    conditions: list[ConditionLLM] = Field(default_factory=list)
    medications: list[MedicationLLM] = Field(default_factory=list)


# =========================================================================
# Agent 4 — Matching modèle / données
# =========================================================================
class FeatureSpec(BaseModel):
    name: str
    dtype: Literal["float", "int", "bool"]
    unit: str | None = None
    required: bool = True
    min: float | None = None
    max: float | None = None
    impute_value: float | None = None  # médiane d'entraînement
    derivable_from: list[str] = Field(default_factory=list)
    max_age_days: int | None = None
    description: str = ""


class ModelCard(BaseModel):
    model_id: str
    name: str
    version: str
    task: str
    clinical_question: str
    algorithm: str
    target: str
    features: list[FeatureSpec]
    exclusion_conditions: list[str] = Field(default_factory=list)
    inclusion_conditions: list[str] = Field(default_factory=list)
    risk_thresholds: dict[str, float]
    training_population: str
    performance: dict[str, float] = Field(default_factory=dict)
    intended_use: str
    limitations: list[str]
    artifact: str
    sha256: str | None = None

    def spec(self, name: str) -> FeatureSpec:
        return next(f for f in self.features if f.name == name)


class ModelMatch(BaseModel):
    model_id: str
    name: str
    compatible: bool
    confidence: float
    coverage_required: float
    coverage_optional: float
    present: list[str]
    derivable: list[str]
    required_missing: list[str]
    optional_missing: list[str]
    out_of_range: list[str]
    stale: list[str]
    excluded_by: list[str]
    reasons: list[str]


class ModelSelectionLLM(BaseModel):
    selected_model_id: str
    rationale: str = Field(description="2 à 3 phrases, en français")


class MatchReport(BaseModel):
    clinical_question: str
    matches: list[ModelMatch]
    selected_model_id: str | None
    rationale: str
    needs_human_review: bool = False


# =========================================================================
# Agent 5 — Prétraitement
# =========================================================================
class PreprocessStep(BaseModel):
    feature: str
    operation: Literal[
        "keep", "derive_egfr_ckdepi2021", "derive_bmi", "impute_constant", "clip", "bool_to_int", "to_int"
    ]
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str


class PreprocessCodeLLM(BaseModel):
    code: str = Field(description="Code Python définissant def preprocess(df) -> df")
    explanation: str


class PreprocessResult(BaseModel):
    model_id: str
    steps: list[PreprocessStep]
    code: str
    code_origin: Literal["llm", "template"]
    attempts: list[dict[str, Any]]
    input_row: dict[str, Any]
    output_row: dict[str, float]
    imputed: list[str]
    derived: list[str]
    clipped: list[str]


# =========================================================================
# Agent 6 — Inférence & explicabilité
# =========================================================================
class Contribution(BaseModel):
    feature: str
    value: float
    shap: float  # contribution en log-odds


class Prediction(BaseModel):
    model_id: str
    model_version: str
    probability: float
    ci_low: float
    ci_high: float
    risk_category: str
    base_value: float
    contributions: list[Contribution]
    figure_path: str | None = None


class ReportNarrativeLLM(BaseModel):
    synthese: str = Field(description="3 phrases maximum")
    facteurs_explicatifs: list[str] = Field(description="Un facteur par élément, du plus au moins influent")
    limites: list[str]
    points_a_verifier: list[str] = Field(description="Ce que le clinicien doit vérifier avant d'utiliser le résultat")


class ClinicalReport(BaseModel):
    pseudo_id: str
    model_id: str
    markdown: str
    narrative: ReportNarrativeLLM
    grounding_ok: bool
    unsupported_numbers: list[str]
    path: str | None = None
    status: Literal["DRAFT", "VALIDATED", "REJECTED"] = "DRAFT"
    reviewer_comment: str | None = None

    @field_validator("unsupported_numbers")
    @classmethod
    def _dedupe(cls, v: list[str]) -> list[str]:
        return sorted(set(v))


# =========================================================================
# Audit
# =========================================================================
class AuditEvent(BaseModel):
    ts: str = Field(default_factory=now_iso)
    agent: str
    action: str
    status: Literal["ok", "warning", "blocked", "human"] = "ok"
    details: dict[str, Any] = Field(default_factory=dict)
    output_hash: str | None = None
