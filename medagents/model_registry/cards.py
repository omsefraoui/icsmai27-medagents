"""Catalogue des modèles « pré-entraînés » disponibles pour l'Agent 4.

⚠️ Modèles entraînés sur des cohortes SYNTHÉTIQUES simulées pour l'atelier.
Les coefficients sont plausibles mais n'ont AUCUNE validité clinique.
"""
from __future__ import annotations

from ..schemas import FeatureSpec, ModelCard

F = FeatureSpec

HF_READMISSION = ModelCard(
    model_id="hf_readmission_30d",
    name="Réhospitalisation à 30 jours — insuffisance cardiaque",
    version="1.2.0",
    task="classification binaire",
    clinical_question="Risque de réhospitalisation toutes causes à 30 jours après une hospitalisation "
    "pour insuffisance cardiaque",
    algorithm="GradientBoostingClassifier (scikit-learn), 20 ré-échantillonnages bootstrap",
    target="readmission_30d",
    features=[
        F(name="age", dtype="float", unit="a", min=18, max=105, description="Âge (années)"),
        F(name="lvef", dtype="float", unit="%", min=5, max=80, max_age_days=365, description="FEVG"),
        F(name="nt_probnp", dtype="float", unit="pg/mL", min=20, max=35000, max_age_days=90, description="NT-proBNP"),
        F(name="egfr", dtype="float", unit="mL/min/1.73m2", min=5, max=130, max_age_days=90,
          derivable_from=["creatinine", "age", "sex_female"], description="DFG CKD-EPI 2021"),
        F(name="sodium", dtype="float", unit="mmol/L", min=115, max=155, max_age_days=90, description="Natrémie"),
        F(name="prior_admissions_12m", dtype="int", unit="n", min=0, max=10,
          description="Hospitalisations dans les 12 mois précédents"),
        F(name="nyha_class", dtype="int", unit="classe", min=1, max=4, description="Classe NYHA"),
        F(name="sbp", dtype="float", unit="mmHg", required=False, min=70, max=220, description="PAS"),
        F(name="heart_rate", dtype="float", unit="/min", required=False, min=35, max=180, description="FC"),
        F(name="hemoglobin", dtype="float", unit="g/dL", required=False, min=6, max=19, description="Hémoglobine"),
        F(name="diabetes", dtype="bool", required=False, description="Diabète de type 2"),
        F(name="atrial_fibrillation", dtype="bool", required=False, description="Fibrillation atriale"),
    ],
    inclusion_conditions=["heart_failure"],
    risk_thresholds={"faible": 0.0, "intermédiaire": 0.15, "élevé": 0.30},
    training_population="Cohorte synthétique de 5 000 séjours pour insuffisance cardiaque, 40-98 ans",
    intended_use="Aide à la priorisation du suivi post-hospitalisation (télésurveillance, consultation précoce)",
    limitations=[
        "Données d'entraînement synthétiques : aucune validation clinique",
        "Non applicable hors insuffisance cardiaque établie",
        "Ne tient pas compte du contexte social ni de l'observance",
    ],
    artifact="hf_readmission_30d.joblib",
)

T2D_ONSET = ModelCard(
    model_id="t2d_onset_5y",
    name="Survenue d'un diabète de type 2 à 5 ans",
    version="2.0.1",
    task="classification binaire",
    clinical_question="Risque de développer un diabète de type 2 dans les 5 ans chez un adulte non diabétique",
    algorithm="Régression logistique standardisée (scikit-learn), 20 ré-échantillonnages bootstrap",
    target="t2d_5y",
    features=[
        F(name="age", dtype="float", unit="a", min=18, max=100),
        F(name="bmi", dtype="float", unit="kg/m2", min=15, max=60, derivable_from=["weight", "height"]),
        F(name="fasting_glucose", dtype="float", unit="mg/dL", min=60, max=125, max_age_days=365),
        F(name="hba1c", dtype="float", unit="%", min=4, max=6.4, max_age_days=365),
        F(name="sbp", dtype="float", unit="mmHg", required=False, min=80, max=220),
        F(name="hdl", dtype="float", unit="mg/dL", required=False, min=15, max=120),
        F(name="triglycerides", dtype="float", unit="mg/dL", required=False, min=30, max=1000),
        F(name="family_history_diabetes", dtype="bool", required=False),
    ],
    exclusion_conditions=["diabetes"],
    risk_thresholds={"faible": 0.0, "intermédiaire": 0.10, "élevé": 0.25},
    training_population="Cohorte synthétique de 6 000 adultes non diabétiques, 25-85 ans",
    intended_use="Dépistage ciblé et prévention (programme d'activité physique, suivi glycémique)",
    limitations=[
        "Données d'entraînement synthétiques : aucune validation clinique",
        "Contre-indiqué si diabète déjà diagnostiqué",
        "HbA1c ininterprétable en cas d'hémoglobinopathie ou d'anémie sévère",
    ],
    artifact="t2d_onset_5y.joblib",
)

CKD_PROGRESSION = ModelCard(
    model_id="ckd_progression_2y",
    name="Progression de la maladie rénale chronique à 2 ans",
    version="1.0.3",
    task="classification binaire",
    clinical_question="Risque de baisse du DFG ≥ 40 % ou d'insuffisance rénale terminale à 2 ans",
    algorithm="Régression logistique standardisée (scikit-learn), 20 ré-échantillonnages bootstrap",
    target="ckd_progression_2y",
    features=[
        F(name="age", dtype="float", unit="a", min=18, max=100),
        F(name="egfr", dtype="float", unit="mL/min/1.73m2", min=10, max=90, max_age_days=180,
          derivable_from=["creatinine", "age", "sex_female"]),
        F(name="uacr", dtype="float", unit="mg/g", min=1, max=5000, max_age_days=180,
          description="Rapport albuminurie/créatininurie"),
        F(name="sbp", dtype="float", unit="mmHg", required=False, min=80, max=220),
        F(name="hba1c", dtype="float", unit="%", required=False, min=4, max=14),
        F(name="hemoglobin", dtype="float", unit="g/dL", required=False, min=6, max=19),
        F(name="diabetes", dtype="bool", required=False),
    ],
    inclusion_conditions=["ckd"],
    risk_thresholds={"faible": 0.0, "intermédiaire": 0.10, "élevé": 0.25},
    training_population="Cohorte synthétique de 5 000 patients MRC stades 3-4",
    intended_use="Orientation néphrologique et intensification de la néphroprotection",
    limitations=[
        "Données d'entraînement synthétiques : aucune validation clinique",
        "Nécessite un RAC récent (< 6 mois)",
    ],
    artifact="ckd_progression_2y.joblib",
)

CARDS = [HF_READMISSION, T2D_ONSET, CKD_PROGRESSION]
