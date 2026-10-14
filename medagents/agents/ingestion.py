"""Agent 1 — Ingestion Identifier Agent.

Rôle : recevoir un dossier brut (fichier, dossier ou ZIP, éventuellement
imbriqué), identifier chaque fichier, extraire des métadonnées non
identifiantes, valider le schéma minimal et décider de la route.

Ce qui est déterministe : octets magiques, parsing, validation.
Ce que fait le LLM : qualifier les documents texte libre (type de document,
spécialité, langue) — information utile au routage et à l'extraction.
Le LLM étant local (Ollama), lire le texte brut avant dé-identification
reste dans la zone de confiance.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from ..schemas import FileType, IngestedFile, IngestionManifest, NoteClassificationLLM
from ..tools import parsers
from .base import BaseAgent

SYSTEM_PROMPT = """Tu es un agent d'archivage médical. Tu qualifies des documents cliniques.
Réponds uniquement avec le JSON demandé. Le résumé ne doit contenir AUCUN nom, date,
adresse ni numéro identifiant le patient."""

ROUTES = {
    FileType.CSV: ["anonymizer", "feature_extraction"],
    FileType.EXCEL: ["anonymizer", "feature_extraction"],
    FileType.FHIR: ["anonymizer", "feature_extraction"],
    FileType.HL7V2: ["anonymizer", "feature_extraction"],
    FileType.CLINICAL_NOTE: ["anonymizer", "feature_extraction"],
    FileType.DICOM: ["anonymizer"],  # métadonnées seulement dans cet atelier
    FileType.JSON: ["anonymizer"],
    FileType.UNKNOWN: [],
}


class IngestionAgent(BaseAgent):
    name = "Agent 1 · Ingestion"
    role = "Identifier, valider et router les fichiers bruts"

    def __init__(self, llm=None, verbose: bool = True, use_llm: bool = True):
        super().__init__(llm, verbose)
        self.use_llm = use_llm

    # ------------------------------------------------------------------ public
    def run(self, source: str | Path, workdir: str | Path) -> IngestionManifest:
        source, workdir = Path(source), Path(workdir)
        staging = workdir / "staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        files, warnings = self._collect(source, staging, warnings=[])
        bundle_id = source.stem
        ingested: list[IngestedFile] = []
        for i, (path, archive) in enumerate(sorted(files, key=lambda t: str(t[0]))):
            ingested.append(self._identify(path, archive, f"F{i + 1:02d}", staging))

        for f in ingested:
            if f.file_type == FileType.UNKNOWN:
                warnings.append(f"{f.original_name} : format non reconnu, fichier mis en quarantaine")
            if not f.schema_valid:
                warnings.append(f"{f.original_name} : {'; '.join(f.validation_errors)}")

        manifest = IngestionManifest(bundle_id=bundle_id, source=str(source), files=ingested, warnings=warnings)
        counts = {t.value: len(manifest.by_type(t)) for t in FileType if manifest.by_type(t)}
        self.log(f"{len(ingested)} fichiers identifiés {counts}", "warning" if warnings else "ok",
                 output=manifest, counts=counts, warnings=warnings)
        return manifest

    # ---------------------------------------------------------------- interne
    def _collect(self, source: Path, staging: Path, warnings: list[str], depth: int = 0,
                 archive: str | None = None) -> tuple[list[tuple[Path, str | None]], list[str]]:
        """Parcourt récursivement fichiers, dossiers et archives ZIP (profondeur ≤ 3)."""
        out: list[tuple[Path, str | None]] = []
        if source.is_dir():
            for p in sorted(source.rglob("*")):
                if p.is_file():
                    sub, _ = self._collect(p, staging, warnings, depth, archive)
                    out += sub
            return out, warnings
        if parsers.detect_file_type(source)[0] == FileType.UNKNOWN and source.read_bytes()[:4] == b"PK\x03\x04":
            if depth >= 3:
                warnings.append(f"{source.name} : archives imbriquées trop profondes")
                return out, warnings
            dest = staging / f"{source.stem}_{depth}"
            try:
                extracted = parsers.safe_extract_zip(source, dest)
                self.log(f"Archive ouverte : {source.name} ({len(extracted)} fichiers)")
            except Exception as exc:
                warnings.append(f"{source.name} : archive refusée ({exc})")
                return out, warnings
            for p in extracted:
                sub, _ = self._collect(p, staging, warnings, depth + 1, source.name)
                out += sub
            return out, warnings
        if not str(source.resolve()).startswith(str(staging.resolve())):
            target = staging / source.name
            shutil.copy2(source, target)
            source = target
        out.append((source, archive))
        return out, warnings

    def _identify(self, path: Path, archive: str | None, file_id: str, staging: Path) -> IngestedFile:
        ft, method, conf = parsers.detect_file_type(path)
        meta, errors = parsers.validate_and_describe(path, ft)
        if ft == FileType.CLINICAL_NOTE and self.use_llm and not errors:
            text = parsers.read_text(path)[:3000]
            try:
                cls = self.llm.structured(SYSTEM_PROMPT, f"Document :\n\"\"\"\n{text}\n\"\"\"",
                                          NoteClassificationLLM, agent=self.name)
                meta.update(cls.model_dump())
                method = "llm" if method == "extension" else method
            except Exception as exc:  # le LLM ne doit jamais bloquer l'ingestion
                meta["llm_error"] = str(exc)[:200]
        rel = path.relative_to(staging) if str(path).startswith(str(staging)) else path.name
        return IngestedFile(
            file_id=file_id, path=str(path), original_name=str(rel), file_type=ft, detection_method=method,
            confidence=conf, size_bytes=path.stat().st_size, sha256=parsers.sha256_file(path),
            from_archive=archive, metadata=meta, schema_valid=not errors, validation_errors=errors,
            route_to=ROUTES[ft] if not errors else [],
        )
