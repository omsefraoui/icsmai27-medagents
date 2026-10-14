"""Formules cliniques de référence utilisées pour dériver des variables."""
from __future__ import annotations

import math
from datetime import date


def egfr_ckd_epi_2021(creatinine_mg_dl: float, age: float, female: bool) -> float:
    """DFG estimé CKD-EPI 2021 (sans coefficient ethnique), mL/min/1,73 m².

    Inker LA et al., N Engl J Med 2021;385:1737-49.
    """
    kappa = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    ratio = creatinine_mg_dl / kappa
    egfr = 142 * min(ratio, 1) ** alpha * max(ratio, 1) ** -1.200 * 0.9938**age
    if female:
        egfr *= 1.012
    return round(egfr, 1)


def bmi(weight_kg: float, height_cm: float) -> float:
    return round(weight_kg / (height_cm / 100) ** 2, 1)


def age_at(birth: date, ref: date) -> int:
    return ref.year - birth.year - ((ref.month, ref.day) < (birth.month, birth.day))


def logit(p: float) -> float:
    return math.log(p / (1 - p))
