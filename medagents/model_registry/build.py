"""Simulation des cohortes synthétiques et entraînement des modèles du registre.

Usage :  python -m medagents.model_registry.build

Les modèles sont (ré)entraînés localement plutôt que distribués en pickle :
un pickle scikit-learn n'est pas portable entre versions et charger un
pickle inconnu revient à exécuter du code arbitraire.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .. import config
from ..tools.parsers import sha256_file
from .cards import CARDS, CKD_PROGRESSION, HF_READMISSION, T2D_ONSET

N_BOOT = 20


def _sig(z):
    return 1 / (1 + np.exp(-z))


def simulate_hf(n: int, rng: np.random.Generator) -> pd.DataFrame:
    age = np.clip(rng.normal(74, 10, n), 40, 98)
    lvef = np.clip(rng.normal(38, 12, n), 10, 75)
    nt = np.clip(np.exp(7.2 + 0.03 * (45 - lvef) + rng.normal(0, 0.9, n)), 50, 35000)
    egfr = np.clip(rng.normal(95 - 0.5 * age, 18, n), 8, 120)
    sodium = np.clip(rng.normal(138, 4, n), 120, 150)
    sbp = np.clip(rng.normal(125, 20, n), 75, 210)
    hr = np.clip(rng.normal(78, 14, n), 40, 160)
    prior = np.clip(rng.poisson(0.8, n), 0, 10)
    nyha_lat = 2.3 + 0.03 * (40 - lvef) + rng.normal(0, 0.8, n)
    nyha = np.clip(np.round(nyha_lat), 1, 4).astype(int)
    diab = rng.binomial(1, 0.35, n)
    af = rng.binomial(1, 0.38, n)
    hb = np.clip(rng.normal(12.8, 1.7, n), 7, 18)
    z = (-1.75 + 0.025 * (age - 74) - 0.02 * (lvef - 38) + 0.45 * (np.log(nt) - 7.6) - 0.015 * (egfr - 58)
         - 0.06 * (sodium - 138) + 0.35 * prior + 0.45 * (nyha - 2) + 0.25 * diab + 0.2 * af
         - 0.12 * (hb - 12.8) - 0.005 * (sbp - 125) + 0.004 * (hr - 78))
    y = rng.binomial(1, _sig(z))
    return pd.DataFrame(dict(age=age.round(), lvef=lvef.round(), nt_probnp=nt.round(), egfr=egfr.round(1),
                             sodium=sodium.round(), prior_admissions_12m=prior, nyha_class=nyha, sbp=sbp.round(),
                             heart_rate=hr.round(), hemoglobin=hb.round(1), diabetes=diab,
                             atrial_fibrillation=af, readmission_30d=y))


def simulate_t2d(n: int, rng: np.random.Generator) -> pd.DataFrame:
    age = np.clip(rng.normal(50, 12, n), 25, 85)
    bmi = np.clip(rng.normal(28, 5, n), 17, 50)
    glu = np.clip(rng.normal(92 + 0.4 * (bmi - 28), 11, n), 65, 125)
    a1c = np.clip(rng.normal(5.5 + 0.012 * (glu - 92), 0.35, n), 4.3, 6.4)
    sbp = np.clip(rng.normal(122 + 0.4 * (age - 50), 15, n), 90, 200)
    hdl = np.clip(rng.normal(56 - 0.6 * (bmi - 28), 12, n), 20, 100)
    tg = np.clip(np.exp(rng.normal(4.85 + 0.02 * (bmi - 28), 0.4, n)), 35, 900)
    fh = rng.binomial(1, 0.3, n)
    z = (-2.9 + 0.03 * (age - 50) + 0.09 * (bmi - 28) + 0.045 * (glu - 92) + 1.1 * (a1c - 5.5)
         + 0.008 * (sbp - 122) - 0.02 * (hdl - 56) + 0.003 * (tg - 130) + 0.6 * fh)
    y = rng.binomial(1, _sig(z))
    return pd.DataFrame(dict(age=age.round(), bmi=bmi.round(1), fasting_glucose=glu.round(), hba1c=a1c.round(1),
                             sbp=sbp.round(), hdl=hdl.round(), triglycerides=tg.round(),
                             family_history_diabetes=fh, t2d_5y=y))


def simulate_ckd(n: int, rng: np.random.Generator) -> pd.DataFrame:
    age = np.clip(rng.normal(66, 12, n), 25, 95)
    egfr = np.clip(rng.normal(42, 13, n), 10, 89)
    uacr = np.clip(np.exp(rng.normal(np.log(120), 1.3, n)), 2, 5000)
    sbp = np.clip(rng.normal(138, 18, n), 90, 210)
    diab = rng.binomial(1, 0.45, n)
    a1c = np.clip(rng.normal(5.7 + 1.4 * diab, 0.8, n), 4.3, 13)
    hb = np.clip(rng.normal(11.5 + 0.03 * egfr, 1.5, n), 7, 17)
    z = (-1.9 - 0.012 * (age - 66) - 0.04 * (egfr - 42) + 0.55 * (np.log(uacr) - np.log(120))
         + 0.015 * (sbp - 138) + 0.2 * (a1c - 6.3) + 0.3 * diab - 0.1 * (hb - 12.7))
    y = rng.binomial(1, _sig(z))
    return pd.DataFrame(dict(age=age.round(), egfr=egfr.round(1), uacr=uacr.round(), sbp=sbp.round(),
                             hba1c=a1c.round(1), hemoglobin=hb.round(1), diabetes=diab, ckd_progression_2y=y))


def _fit(card, df: pd.DataFrame, rng: np.random.Generator, verbose: bool = True) -> dict:
    cols = [f.name for f in card.features]
    X, y = df[cols].astype(float), df[card.target].astype(int)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=7, stratify=y)

    def make():
        if card.algorithm.startswith("Gradient"):
            return GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.05, subsample=0.8,
                                              random_state=int(rng.integers(1_000_000)))
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

    model = make().fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    perf = {"auc": round(float(roc_auc_score(yte, p)), 3), "brier": round(float(brier_score_loss(yte, p)), 3),
            "prevalence": round(float(y.mean()), 3), "n_train": int(len(Xtr)), "n_test": int(len(Xte))}
    boots = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(Xtr), len(Xtr))
        boots.append(make().fit(Xtr.iloc[idx], ytr.iloc[idx]))
    medians = Xtr.median().to_dict()
    if verbose:
        print(f"  {card.model_id:22s} AUC={perf['auc']:.3f}  Brier={perf['brier']:.3f}  prévalence={perf['prevalence']:.2f}")
    return {"model": model, "bootstrap": boots, "background": Xtr.sample(100, random_state=1),
            "feature_names": cols, "medians": medians, "performance": perf}


def build(verbose: bool = True) -> dict:
    rng = np.random.default_rng(2027)
    sims = {HF_READMISSION.model_id: simulate_hf(5000, rng), T2D_ONSET.model_id: simulate_t2d(6000, rng),
            CKD_PROGRESSION.model_id: simulate_ckd(5000, rng)}
    config.COHORTS_DIR.mkdir(parents=True, exist_ok=True)
    registry = []
    if verbose:
        print("Entraînement des modèles du registre (cohortes synthétiques) :")
    for card in CARDS:
        df = sims[card.model_id]
        df.to_csv(config.COHORTS_DIR / f"{card.model_id}_cohort.csv", index=False)
        bundle = _fit(card, df, rng, verbose)
        c = card.model_copy(deep=True)
        c.performance = bundle["performance"]
        for f in c.features:  # valeurs d'imputation = médianes d'entraînement
            f.impute_value = round(float(bundle["medians"][f.name]), 2)
        bundle["card"] = c.model_dump(mode="json")
        path = config.MODELS_DIR / c.artifact
        joblib.dump(bundle, path)
        c.sha256 = sha256_file(path)
        registry.append(c.model_dump(mode="json"))
    (config.MODELS_DIR / "registry.json").write_text(json.dumps(registry, indent=2, ensure_ascii=False), "utf-8")
    if verbose:
        print(f"Registre écrit dans {config.MODELS_DIR / 'registry.json'}")
    return {"models": [r["model_id"] for r in registry]}


if __name__ == "__main__":
    build()
