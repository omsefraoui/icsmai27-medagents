"""Petits utilitaires pour les exercices des notebooks."""
from __future__ import annotations

import re
from typing import Any

from .schemas import PHICategory
from .tools import phi


def check(condition: Any, ok: str = "Correct !", ko: str = "Pas encore — relisez la consigne.") -> None:
    """Affiche ✅ / ❌ sans interrompre le notebook."""
    print(("✅ " + ok) if bool(condition) else ("❌ " + ko))


def register_pattern(category: PHICategory, pattern: re.Pattern[str], group: int = 0) -> None:
    """Ajoute un détecteur regex à l'Agent 2 (sans doublon si la cellule est relancée)."""
    phi.PATTERNS[:] = [p for p in phi.PATTERNS if p[1].pattern != pattern.pattern]
    phi.PATTERNS.append((category, pattern, group))
