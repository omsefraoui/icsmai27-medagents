"""Génération des dossiers patients BRUTS synthétiques de l'atelier.

Usage : python -m medagents.datagen

Trois dossiers, tous fictifs (noms, adresses, numéros inventés) :
- P001 : homme 75 ans, insuffisance cardiaque — dossier riche et hétérogène
         (ZIP contenant FHIR, HL7 v2, CSV « à la française », CR texte,
         ZIP imbriqué avec un DICOM). Fil rouge de la démonstration.
- P002 : femme 51 ans, bilan de prédiabète — unités à convertir (mmol/L,
         mmol/mol, g/L), Excel, IMC à dériver. Support des exercices.
- P003 : homme 93 ans, insuffisance cardiaque — dossier incomplet et sale
         (NT-proBNP absent, valeur aberrante, fichier corrompu, CR en
         anglais). Support de la démonstration Human-in-the-Loop.

Une vérité terrain (PHI à masquer, features attendues) est écrite dans
data/ground_truth/ pour l'évaluation du Module 4.
"""
from __future__ import annotations

import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# =============================================================================
# P001 — Lucien MARCHAND (fictif)
# =============================================================================
P001_NOTE = """CENTRE HOSPITALIER UNIVERSITAIRE — SERVICE DE CARDIOLOGIE
Unité d'insuffisance cardiaque

COMPTE RENDU D'HOSPITALISATION

Patient : M. Lucien MARCHAND, né le 14/03/1951 (75 ans)
IPP : 4471829 — N° sécurité sociale : 1 51 03 35 238 041 27
Adresse : 12 rue des Tanneurs, 35000 Rennes — Tél. : 06 12 34 56 78
Médecin traitant : Dr Yann Le Goff
Séjour du 22/08/2026 au 01/09/2026

Motif d'hospitalisation :
Décompensation cardiaque globale sur cardiopathie ischémique connue.

Antécédents :
- Insuffisance cardiaque à fraction d'éjection réduite, suivie depuis 2019.
- Deux hospitalisations pour décompensation cardiaque au cours des 12 derniers mois (mars et juin 2026).
- Fibrillation atriale permanente, anticoagulée.
- Diabète de type 2 traité par metformine.
- Insuffisance rénale chronique stade 3.
- Pas de BPCO. Ancien tabagisme sevré en 2005.

Examen clinique à l'entrée :
PA 108/64 mmHg, FC 88/min, irrégulière. SpO2 93 % en air ambiant.
Œdèmes des membres inférieurs bilatéraux, crépitants aux deux bases.
Poids 79 kg (poids sec estimé 75 kg).

Examens complémentaires :
Échocardiographie transthoracique du 25/08/2026 : FEVG estimée à 35 %, VG dilaté, insuffisance mitrale modérée.
Biologie : voir résultats du laboratoire (NT-proBNP très élevé, hyponatrémie modérée).

Évolution :
Bonne réponse au furosémide IV, perte de 4 kg. Patient en classe NYHA III à la sortie.

Traitement de sortie :
Sacubitril/valsartan 49/51 mg x2/j, bisoprolol 5 mg/j, furosémide 40 mg/j, apixaban 5 mg x2/j, metformine 1000 mg x2/j.

Suivi : consultation avec le Dr Hélène Kervella dans 15 jours. Contact : h.kervella@chu-exemple.fr

Dr Hélène Kervella, cardiologue
"""


def _fhir_bundle_p001() -> dict:
    def obs(oid, loinc, display, value, unit, when):
        return {"resource": {
            "resourceType": "Observation", "id": oid, "status": "final",
            "subject": {"reference": "Patient/pat-4471829"},
            "code": {"coding": [{"system": "http://loinc.org", "code": loinc, "display": display}], "text": display},
            "valueQuantity": {"value": value, "unit": unit, "system": "http://unitsofmeasure.org"},
            "effectiveDateTime": when}}

    def cond(cid, snomed, display):
        return {"resource": {
            "resourceType": "Condition", "id": cid,
            "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                           "code": "active"}]},
            "code": {"coding": [{"system": "http://snomed.info/sct", "code": snomed, "display": display}]},
            "subject": {"reference": "Patient/pat-4471829"}}}

    def med(mid, atc, display):
        return {"resource": {
            "resourceType": "MedicationStatement", "id": mid, "status": "active",
            "medicationCodeableConcept": {"coding": [{"system": "http://www.whocc.no/atc", "code": atc,
                                                      "display": display}], "text": display},
            "subject": {"reference": "Patient/pat-4471829"}}}

    return {
        "resourceType": "Bundle", "type": "collection", "id": "bundle-p001",
        "entry": [
            {"resource": {
                "resourceType": "Patient", "id": "pat-4471829",
                "identifier": [{"system": "urn:oid:1.2.250.1.213.1.4.8", "value": "1510335238041"},
                               {"system": "urn:chu:ipp", "value": "4471829"}],
                "name": [{"family": "Marchand", "given": ["Lucien"]}],
                "gender": "male", "birthDate": "1951-03-14",
                "telecom": [{"system": "phone", "value": "06 12 34 56 78"},
                            {"system": "email", "value": "l.marchand51@exemple.fr"}],
                "address": [{"line": ["12 rue des Tanneurs"], "city": "Rennes", "postalCode": "35000",
                             "country": "FR"}]}},
            obs("obs-ntprobnp", "33762-6", "NT-proBNP", 4850, "pg/mL", "2026-08-31T07:30:00+02:00"),
            obs("obs-na", "2951-2", "Sodium", 133, "mmol/L", "2026-08-31T07:30:00+02:00"),
            obs("obs-hb", "718-7", "Hemoglobin", 11.4, "g/dL", "2026-08-31T07:30:00+02:00"),
            obs("obs-hr", "8867-4", "Heart rate", 84, "/min", "2026-09-01T08:00:00+02:00"),
            obs("obs-sbp", "8480-6", "Systolic blood pressure", 112, "mmHg", "2026-09-01T08:00:00+02:00"),
            cond("cond-hf", "84114007", "Heart failure"),
            cond("cond-af", "49436004", "Atrial fibrillation"),
            cond("cond-t2d", "44054006", "Diabetes mellitus type 2"),
            med("med-1", "C09DX04", "Sacubitril/valsartan"), med("med-2", "C07AB07", "Bisoprolol"),
            med("med-3", "C03CA01", "Furosemide"), med("med-4", "B01AF02", "Apixaban"),
            med("med-5", "A10BA02", "Metformin"),
        ],
    }


P001_HL7 = "\r".join([
    "MSH|^~\\&|LABO|CHU_RENNES|DPI|CHU_RENNES|20260831094500||ORU^R01|MSG00042|P|2.5",
    "PID|1||4471829^^^CHU^PI||MARCHAND^LUCIEN||19510314|M|||12 RUE DES TANNEURS^^RENNES^^35000^FRA||0612345678"
    "||||||151033523804127",
    "PV1|1|I|CARDIO^U12^LIT3||||^KERVELLA^HELENE",
    "OBR|1||LAB2026-0831-77|BIOCHIMIE^Bilan biochimique|||20260831073000",
    "OBX|1|NM|2160-0^Creatinine^LN||141|umol/L|59-104|H|||F|||20260831073000",
    "OBX|2|NM|2951-2^Sodium^LN||133|mmol/L|136-145|L|||F|||20260831073000",
    "OBX|3|NM|2823-3^Potassium^LN||4.8|mmol/L|3.5-5.1|N|||F|||20260831073000",
]) + "\r"

P001_CSV = """IPP;Nom patient;Date prélèvement;Examen;Résultat;Unité
4471829;MARCHAND Lucien;31/08/2026;Hémoglobine;11,4;g/dL
4471829;MARCHAND Lucien;31/08/2026;Créatinine;141;µmol/L
4471829;MARCHAND Lucien;31/08/2026;Kaliémie;4,8;mmol/L
4471829;MARCHAND Lucien;31/08/2026;Natrémie;133;mmol/L
4471829;MARCHAND Lucien;28/08/2026;HbA1c;7,4;%
4471829;MARCHAND Lucien;28/08/2026;Pro-BNP N-terminal;4850;ng/L
"""

# =============================================================================
# P002 — Claire DUMONTEL (fictive)
# =============================================================================
P002_NOTE = """Cabinet médical des Lilas — Consultation de médecine générale du 15/09/2026

Mme Claire Dumontel, 51 ans, née le 02/11/1974, vient pour le résultat de son bilan sanguin.
Tél : 07 81 22 45 90 — Mutuelle n° MUT-558201.

Contexte : bilan demandé devant une prise de poids progressive. Mère diabétique de type 2.
Pas de diabète connu. Pas d'antécédent cardiovasculaire. Pas d'HTA traitée.
Activité physique faible (moins d'une heure par semaine). Non fumeuse.

Examen : poids 86 kg pour 1,65 m. PA 138/86 mmHg. FC 76/min.

Biologie (cf. résultats du laboratoire) : glycémie à jeun et HbA1c à la limite haute.

Conclusion : surpoids/obésité modérée avec anomalies glycémiques à surveiller.
Proposition d'évaluation du risque de diabète à 5 ans.

Dr Karim Benhaddou
"""

P002_CSV = """patient_id,sample_date,Fasting glucose (mmol/L),HbA1c (mmol/mol),HDL cholesterol (mmol/L),Triglycerides (g/L)
CD-19741102,2026-09-08,"6.2","43","1.1","1.9"
"""

# =============================================================================
# P003 — Bernard LEFORT (fictif) : dossier incomplet
# =============================================================================
P003_NOTE = """DISCHARGE SUMMARY — Department of Cardiology

Patient: Mr Bernard Lefort, 93 years old, DOB 05/06/1933. MRN: 8820461.
Phone: 02 99 41 18 73. Admitted 10/09/2026, discharged 18/09/2026.

History: chronic heart failure with reduced ejection fraction, hypertension.
Two prior admissions for heart failure in the last 12 months.
No diabetes. No atrial fibrillation on ECG.

Echocardiography: LVEF 30 %. NYHA class III at discharge.
Vitals at discharge: BP 118/70 mmHg, HR 72 bpm.
NT-proBNP: sample haemolysed, not available - to be repeated in primary care.

Discharge plan: furosemide 40 mg daily, bisoprolol 2.5 mg daily.
Follow-up with Dr Anne Quéré in 2 weeks.
"""

P003_CSV = """Date;Analyse;Valeur;Unité
17/09/2026;Créatinine;102;µmol/L
17/09/2026;Natrémie;1380;mmol/L
17/09/2026;Hémoglobine;12,1;g/dL
"""


# =============================================================================
def _dicom_bytes(name: str, pid: str, birth: str, study_date: str, physician: str) -> bytes:
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.6.1"  # Ultrasound Image Storage
    meta.MediaStorageSOPInstanceUID = generate_uid(entropy_srcs=[pid, study_date])
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid(entropy_srcs=[pid, "study"])
    ds.SeriesInstanceUID = generate_uid(entropy_srcs=[pid, "series"])
    ds.PatientName = name
    ds.PatientID = pid
    ds.PatientBirthDate = birth
    ds.PatientSex = "M"
    ds.StudyDate = study_date
    ds.StudyTime = "101500"
    ds.AccessionNumber = "ACC" + pid[-5:]
    ds.InstitutionName = "CHU EXEMPLE"
    ds.ReferringPhysicianName = physician
    ds.OperatorsName = "TECH^SONIA"
    ds.Modality = "US"
    ds.StudyDescription = "Echocardiographie transthoracique"
    ds.DeviceSerialNumber = "SN-VIVID-77231"
    rng = np.random.default_rng(int(pid[-4:]))
    img = (rng.random((64, 64)) * 255).astype(np.uint8)
    ds.Rows, ds.Columns = img.shape
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = img.tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def _excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    return buf.getvalue()


def _zip(path: Path, files: dict[str, bytes | str]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content.encode("utf-8") if isinstance(content, str) else content)


def generate(out_dir: Path | None = None) -> list[Path]:
    raw = Path(out_dir or config.RAW_DIR)
    gt_dir = config.GROUND_TRUTH_DIR
    raw.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- P001
    nested = io.BytesIO()
    with zipfile.ZipFile(nested, "w") as zf:
        zf.writestr("ETT_20260825.dcm",
                    _dicom_bytes("MARCHAND^LUCIEN", "4471829", "19510314", "20260825", "KERVELLA^HELENE"))
    _zip(raw / "dossier_P001.zip", {
        "fhir/patient_bundle.json": json.dumps(_fhir_bundle_p001(), indent=2, ensure_ascii=False),
        "hl7/ORU_labo_20260831.hl7": P001_HL7,
        "labo/bilan_biologique.csv": P001_CSV,
        "comptes_rendus/CR_hospitalisation_cardiologie.txt": P001_NOTE,
        "imagerie/export_pacs.zip": nested.getvalue(),
    })

    # ---------------- P002
    vitals = pd.DataFrame({"Nom": ["DUMONTEL Claire"], "Date": ["15/09/2026"], "Poids (kg)": [86],
                           "Taille (cm)": [165], "PA systolique (mmHg)": [138], "PA diastolique (mmHg)": [86],
                           "FC (bpm)": [76]})
    p002_fhir = {"resourceType": "Bundle", "type": "collection", "entry": [{"resource": {
        "resourceType": "Patient", "id": "pat-cd", "name": [{"family": "Dumontel", "given": ["Claire"]}],
        "gender": "female", "birthDate": "1974-11-02",
        "identifier": [{"system": "urn:cabinet:id", "value": "CD-19741102"}]}}]}
    _zip(raw / "dossier_P002.zip", {
        "patient.json": json.dumps(p002_fhir, indent=2, ensure_ascii=False),
        "consultation_15092026.txt": P002_NOTE,
        "resultats_labo.csv": P002_CSV,
        "constantes.xlsx": _excel_bytes(vitals),
    })

    # ---------------- P003
    p003_fhir = {"resourceType": "Bundle", "type": "collection", "entry": [{"resource": {
        "resourceType": "Patient", "id": "pat-8820461", "name": [{"family": "Lefort", "given": ["Bernard"]}],
        "gender": "male", "birthDate": "1933-06-05",
        "identifier": [{"system": "urn:chu:ipp", "value": "8820461"}]}}]}
    rng = np.random.default_rng(3)
    _zip(raw / "dossier_P003.zip", {
        "patient.json": json.dumps(p003_fhir, indent=2),
        "discharge_summary.txt": P003_NOTE,
        "bio_17092026.csv": P003_CSV,
        "scan_courrier_illisible.bin": bytes(rng.integers(0, 256, 3000, dtype=np.uint8)),
    })

    # ---------------- Vérité terrain
    truth = {
        "P001": {
            "phi_strings": ["Marchand", "Lucien", "4471829", "1 51 03 35 238 041 27", "151033523804127",
                            "06 12 34 56 78", "0612345678", "l.marchand51@exemple.fr", "12 rue des Tanneurs",
                            "Tanneurs", "Goff", "Kervella", "Hélène", "HELENE", "14/03/1951", "19510314",
                            "1951-03-14", "h.kervella@chu-exemple.fr", "SN-VIVID-77231"],
            "features": {"age": 75, "lvef": 35, "nt_probnp": 4850, "creatinine": 1.59, "egfr": 45.0,
                         "sodium": 133, "prior_admissions_12m": 2, "nyha_class": 3, "sbp": 112,
                         "heart_rate": 84, "hemoglobin": 11.4, "hba1c": 7.4, "potassium": 4.8},
            "conditions": {"heart_failure": True, "atrial_fibrillation": True, "diabetes": True, "ckd": True,
                           "copd": False},
            "expected_model": "hf_readmission_30d",
        },
        "P002": {
            "phi_strings": ["Dumontel", "DUMONTEL", "Claire", "02/11/1974", "1974-11-02", "07 81 22 45 90",
                            "CD-19741102", "MUT-558201", "Benhaddou"],
            "features": {"age": 51, "bmi": 31.6, "fasting_glucose": 111.7, "hba1c": 6.08, "sbp": 138,
                         "hdl": 42.5, "triglycerides": 190, "weight": 86, "height": 165},
            "conditions": {"family_history_diabetes": True, "diabetes": False},
            "expected_model": "t2d_onset_5y",
        },
        "P003": {
            "phi_strings": ["Lefort", "Bernard", "8820461", "05/06/1933", "1933-06-05", "02 99 41 18 73",
                            "Quéré", "93 years"],
            "features": {"age": 90, "lvef": 30, "nyha_class": 3, "prior_admissions_12m": 2, "sbp": 118,
                         "heart_rate": 72, "creatinine": 1.15, "hemoglobin": 12.1},
            "conditions": {"heart_failure": True, "hypertension": True, "diabetes": False,
                           "atrial_fibrillation": False},
            "expected_model": "hf_readmission_30d",
            "notes": "NT-proBNP manquant (requis) et natrémie aberrante (1380) : modèle utilisable seulement "
                     "après revue humaine. Âge réel 93 ans, agrégé à 90 (HIPAA).",
        },
    }
    (gt_dir / "ground_truth.json").write_text(json.dumps(truth, indent=2, ensure_ascii=False), "utf-8")
    return sorted(raw.glob("dossier_*.zip"))


def ensure_data() -> None:
    if not (config.RAW_DIR / "dossier_P001.zip").exists():
        generate()


if __name__ == "__main__":
    for p in generate():
        print("écrit :", p)
    shutil.rmtree(config.RAW_DIR / "__pycache__", ignore_errors=True)
