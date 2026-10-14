"""Détection de format, extraction sécurisée d'archives et lecteurs de fichiers cliniques.

Formats couverts : CSV (séparateur/décimale auto), Excel, FHIR R4 (JSON),
HL7 v2 (ORU/ADT), DICOM, notes cliniques texte, archives ZIP (imbriquées).
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from ..schemas import FileType

MAX_ARCHIVE_FILES = 500
MAX_ARCHIVE_BYTES = 500 * 1024 * 1024  # garde-fou « zip bomb »


# ------------------------------------------------------------------ utils
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text(path: Path) -> str:
    raw = Path(path).read_bytes()
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# --------------------------------------------------------------- archives
class ArchiveError(RuntimeError):
    pass


def safe_extract_zip(zip_path: Path, dest: Path) -> list[Path]:
    """Extrait une archive en se protégeant du *zip-slip* et des *zip bombs*."""
    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        if len(infos) > MAX_ARCHIVE_FILES:
            raise ArchiveError(f"Trop de fichiers dans l'archive ({len(infos)})")
        if sum(i.file_size for i in infos) > MAX_ARCHIVE_BYTES:
            raise ArchiveError("Taille décompressée excessive (zip bomb ?)")
        for info in infos:
            name = Path(info.filename)
            if name.name.startswith(".") or "__MACOSX" in name.parts:
                continue
            target = (dest / name).resolve()
            if not str(target).startswith(str(dest)):
                raise ArchiveError(f"Chemin dangereux dans l'archive : {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as dst:
                dst.write(src.read())
            out.append(target)
    return out


# --------------------------------------------------------------- détection
def detect_file_type(path: Path) -> tuple[FileType, str, float]:
    """Retourne (type, méthode, confiance). Ordre : octets magiques > contenu > extension."""
    path = Path(path)
    head = path.read_bytes()[:2048]
    ext = path.suffix.lower()

    # 1) Octets magiques
    if len(head) >= 132 and head[128:132] == b"DICM":
        return FileType.DICOM, "magic_bytes", 0.99
    if head[:4] == b"PK\x03\x04":
        try:
            with zipfile.ZipFile(path) as zf:
                names = zf.namelist()
            if "[Content_Types].xml" in names and any(n.startswith("xl/") for n in names):
                return FileType.EXCEL, "magic_bytes", 0.98
        except zipfile.BadZipFile:
            pass
        return FileType.UNKNOWN, "magic_bytes", 0.5  # archive : traitée en amont

    # 2) Contenu texte
    try:
        text = head.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = head.decode("latin-1", errors="replace")
    stripped = text.lstrip()
    if stripped.startswith("MSH|"):
        return FileType.HL7V2, "content", 0.98
    if stripped.startswith("{"):
        try:
            obj = json.loads(read_text(path))
            if isinstance(obj, dict) and "resourceType" in obj:
                return FileType.FHIR, "content", 0.97
            return FileType.JSON, "content", 0.9
        except json.JSONDecodeError:
            pass
    lines = [ln for ln in text.splitlines() if ln.strip()][:5]
    if len(lines) >= 2:
        try:
            dialect = csv.Sniffer().sniff("\n".join(lines), delimiters=";,\t|")
            counts = {ln.count(dialect.delimiter) for ln in lines}
            if len(counts) == 1 and counts.pop() >= 1 and ext in {".csv", ".tsv", ".txt", ""}:
                return FileType.CSV, "content", 0.9
        except csv.Error:
            pass

    # 3) Extension (repli)
    by_ext = {".csv": FileType.CSV, ".tsv": FileType.CSV, ".xlsx": FileType.EXCEL, ".json": FileType.JSON,
              ".hl7": FileType.HL7V2, ".dcm": FileType.DICOM, ".txt": FileType.CLINICAL_NOTE,
              ".md": FileType.CLINICAL_NOTE}
    if ext in by_ext:
        return by_ext[ext], "extension", 0.6
    if text.count(" ") > 20:  # texte libre sans extension parlante
        return FileType.CLINICAL_NOTE, "content", 0.55
    return FileType.UNKNOWN, "extension", 0.2


# ------------------------------------------------------------------ lecteurs
def read_csv_any(path: Path) -> pd.DataFrame:
    """CSV « à la française » ou anglo-saxon : séparateur et décimale détectés."""
    text = read_text(path)
    sample = "\n".join(text.splitlines()[:10])
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=";,\t|").delimiter
    except csv.Error:
        sep = ","
    decimal = "," if sep == ";" else "."
    return pd.read_csv(io.StringIO(text), sep=sep, decimal=decimal, dtype=str).apply(
        lambda s: s.str.strip() if s.dtype == object else s
    )


def read_excel_any(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, dtype=str)


def to_number(x: Any) -> float | None:
    if x is None:
        return None
    s = str(x).strip().replace(" ", "").replace(" ", "").replace(",", ".")
    s = s.lstrip("<>=~")
    try:
        return float(s)
    except ValueError:
        return None


def load_fhir(path: Path) -> dict[str, Any]:
    return json.loads(read_text(path))


def fhir_resources(bundle: dict[str, Any], rtype: str) -> list[dict[str, Any]]:
    if bundle.get("resourceType") == rtype:
        return [bundle]
    return [e["resource"] for e in bundle.get("entry", []) if e.get("resource", {}).get("resourceType") == rtype]


def parse_hl7(text: str) -> list[list[str]]:
    """Segments HL7 v2 -> listes de champs (séparateur lu dans MSH-1)."""
    text = text.replace("\r\n", "\r").replace("\n", "\r")
    segs = [s for s in text.split("\r") if s.strip()]
    if not segs or not segs[0].startswith("MSH"):
        raise ValueError("Message HL7 sans segment MSH")
    sep = segs[0][3]
    return [s.split(sep) for s in segs]


def hl7_get(seg: list[str], idx: int, comp: int | None = None) -> str:
    """Champ HL7 (indexation HL7 : PID-5 = seg[5]; pour MSH, décalage d'1)."""
    i = idx - 1 if seg[0] == "MSH" else idx
    if i >= len(seg):
        return ""
    val = seg[i]
    if comp is not None:
        parts = val.split("^")
        return parts[comp - 1] if comp - 1 < len(parts) else ""
    return val


def read_dicom(path: Path):
    import pydicom

    return pydicom.dcmread(str(path))


# --------------------------------------------------------------- validation
def validate_and_describe(path: Path, ft: FileType) -> tuple[dict[str, Any], list[str]]:
    """Extrait des métadonnées *non identifiantes* et valide le schéma minimal."""
    meta: dict[str, Any] = {}
    errors: list[str] = []
    try:
        if ft in (FileType.CSV, FileType.EXCEL):
            df = read_csv_any(path) if ft == FileType.CSV else read_excel_any(path)
            meta = {"n_rows": len(df), "n_cols": df.shape[1], "columns": list(df.columns)}
            if df.empty:
                errors.append("Tableau vide")
            if df.columns.duplicated().any():
                errors.append("Colonnes dupliquées")
        elif ft == FileType.FHIR:
            b = load_fhir(path)
            types: dict[str, int] = {}
            for e in b.get("entry", []):
                rt = e.get("resource", {}).get("resourceType", "?")
                types[rt] = types.get(rt, 0) + 1
            meta = {"resourceType": b.get("resourceType"), "fhir_type": b.get("type"), "resources": types}
            if b.get("resourceType") != "Bundle":
                errors.append("Ressource racine différente de Bundle")
            for obs in fhir_resources(b, "Observation"):
                if not obs.get("code", {}).get("coding"):
                    errors.append(f"Observation {obs.get('id')} sans code")
                if "valueQuantity" not in obs and "valueCodeableConcept" not in obs:
                    errors.append(f"Observation {obs.get('id')} sans valeur")
        elif ft == FileType.HL7V2:
            segs = parse_hl7(read_text(path))
            msh = segs[0]
            meta = {"message_type": hl7_get(msh, 9), "version": hl7_get(msh, 12),
                    "segments": [s[0] for s in segs]}
            if not any(s[0] == "PID" for s in segs):
                errors.append("Segment PID manquant")
        elif ft == FileType.DICOM:
            ds = read_dicom(path)
            meta = {"Modality": str(ds.get("Modality", "")), "StudyDescription": str(ds.get("StudyDescription", "")),
                    "Rows": int(ds.get("Rows", 0)), "Columns": int(ds.get("Columns", 0)),
                    "SOPClassUID": str(ds.get("SOPClassUID", ""))}
            for tag in ("SOPClassUID", "Modality"):
                if tag not in ds:
                    errors.append(f"Tag DICOM obligatoire manquant : {tag}")
        elif ft == FileType.CLINICAL_NOTE:
            t = read_text(path)
            meta = {"n_chars": len(t), "n_lines": t.count("\n") + 1}
            if len(t.strip()) < 20:
                errors.append("Document quasi vide")
    except Exception as exc:  # fichier corrompu
        errors.append(f"Lecture impossible : {type(exc).__name__}: {exc}")
    return meta, errors
