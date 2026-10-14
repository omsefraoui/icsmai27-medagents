"""Explicabilité : valeurs de SHAP (log-odds) et graphique waterfall."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _final_estimator(model: Any) -> tuple[Any, Any | None]:
    """(estimateur final, transformeur amont éventuel) pour un Pipeline sklearn."""
    if hasattr(model, "steps"):
        pre = model[:-1] if len(model.steps) > 1 else None
        return model.steps[-1][1], pre
    return model, None


def shap_values(model: Any, background: pd.DataFrame, x: pd.DataFrame) -> tuple[np.ndarray, float]:
    """Contributions SHAP en log-odds pour une ligne `x`.

    - Régression logistique (après StandardScaler) : LinearExplainer (exact).
    - Gradient boosting : TreeExplainer (exact pour les arbres).
    """
    import shap

    est, pre = _final_estimator(model)
    xb = pre.transform(background) if pre is not None else background
    xx = pre.transform(x) if pre is not None else x
    name = type(est).__name__
    if "Logistic" in name:
        explainer = shap.LinearExplainer(est, np.asarray(xb))
        vals = explainer.shap_values(np.asarray(xx))
        base = float(np.ravel(explainer.expected_value)[0])
    else:
        explainer = shap.TreeExplainer(est, data=np.asarray(xb), model_output="raw")
        vals = explainer.shap_values(np.asarray(xx), check_additivity=False)
        base = float(np.ravel(explainer.expected_value)[-1])
    vals = np.asarray(vals)
    if vals.ndim == 3:  # (n, features, classes)
        vals = vals[..., -1]
    return vals[0], base


def waterfall_png(values: np.ndarray, base: float, x_display: pd.Series, labels: list[str],
                  path: Path, title: str, max_display: int = 10) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    exp = shap.Explanation(values=np.asarray(values), base_values=base,
                           data=np.asarray(x_display.values, dtype=float), feature_names=labels)
    plt.figure()
    shap.plots.waterfall(exp, max_display=max_display, show=False)
    fig = plt.gcf()
    fig.suptitle(title, fontsize=11)
    fig.set_size_inches(9, 5.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def sigmoid(z: float) -> float:
    return float(1 / (1 + np.exp(-z)))
