"""Exécution contrôlée de code Python généré par un LLM.

Deux barrières :
1. **Analyse statique (AST)** : liste blanche d'imports, interdiction des
   appels/attributs dangereux, signature imposée `def preprocess(df)`.
2. **Exécution isolée** : sous-processus Python séparé (`-I`, mode isolé),
   répertoire temporaire, environnement minimal, délai maximal et, sous
   Linux, limites CPU/mémoire.

⚠️ Pédagogique : c'est une défense en profondeur, PAS un vrai bac à sable.
En production : conteneur jetable sans réseau (gVisor, Firecracker, E2B,
nsjail...), système de fichiers en lecture seule, quotas.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

ALLOWED_IMPORTS = {"pandas", "numpy", "math"}
FORBIDDEN_NAMES = {
    "exec", "eval", "compile", "open", "__import__", "globals", "locals", "vars", "input", "breakpoint",
    "getattr", "setattr", "delattr", "help", "exit", "quit", "memoryview", "dir",
}
FORBIDDEN_ATTRS = {
    "system", "popen", "remove", "unlink", "rmdir", "to_pickle", "read_pickle", "to_csv", "read_csv",
    "to_excel", "read_excel", "to_sql", "read_sql", "to_parquet", "read_parquet", "read_json", "to_json",
    "read_html", "eval", "query", "load", "save", "savez", "savetxt", "loadtxt", "fromfile", "tofile",
    "genfromtxt", "io", "lib", "os", "sys", "subprocess", "ctypes",
}


class CodeRejected(ValueError):
    pass


def validate_code(code: str, extra_forbidden_attrs: set[str] | None = None) -> list[str]:
    """Renvoie la liste des violations (vide = code accepté)."""
    problems: list[str] = []
    forbidden_attrs = FORBIDDEN_ATTRS | (extra_forbidden_attrs or set())
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [f"Erreur de syntaxe ligne {exc.lineno} : {exc.msg}"]

    has_fn = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"Import interdit : {a.name}")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"Import interdit : {node.module}")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            problems.append(f"Nom interdit : {node.id}")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                problems.append(f"Attribut privé/dunder interdit : {node.attr}")
            elif node.attr in forbidden_attrs:
                problems.append(f"Attribut interdit (E/S ou exécution) : {node.attr}")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append("global/nonlocal interdit")
        elif isinstance(node, ast.FunctionDef) and node.name == "preprocess":
            has_fn = True
            if len(node.args.args) != 1:
                problems.append("preprocess doit prendre exactement un argument (df)")
        elif isinstance(node, (ast.While,)):
            problems.append("Boucle while interdite (risque de boucle infinie)")
    if not has_fn:
        problems.append("Fonction `def preprocess(df)` absente")
    return sorted(set(problems))


_RUNNER = textwrap.dedent(
    """
    import json, sys
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 ** 3, 2 * 1024 ** 3))
    except Exception:
        pass
    import pandas as pd
    import numpy as np
    payload = json.loads(sys.stdin.read())
    df = pd.DataFrame(payload["records"], columns=payload["columns"])
    ns = {"pd": pd, "np": np}
    exec(compile(payload["code"], "<llm_code>", "exec"), ns)
    out = ns["preprocess"](df.copy())
    if not isinstance(out, pd.DataFrame):
        raise TypeError("preprocess doit renvoyer un DataFrame, reçu " + type(out).__name__)
    sys.stdout.write(json.dumps({"columns": list(map(str, out.columns)),
                                 "records": json.loads(out.to_json(orient="records"))}))
    """
)


@dataclass
class SandboxResult:
    ok: bool
    output: pd.DataFrame | None
    error: str | None
    violations: list[str]


def run_sandboxed(code: str, df: pd.DataFrame, timeout: float = 15,
                  extra_forbidden_attrs: set[str] | None = None) -> SandboxResult:
    violations = validate_code(code, extra_forbidden_attrs)
    if violations:
        return SandboxResult(False, None, "Code rejeté par l'analyse statique", violations)
    payload = json.dumps({"code": code, "columns": list(df.columns),
                          "records": json.loads(df.to_json(orient="records"))})
    env = {k: os.environ[k] for k in ("PATH", "SYSTEMROOT", "LANG") if k in os.environ}
    with tempfile.TemporaryDirectory() as tmp:
        runner = Path(tmp) / "runner.py"
        runner.write_text(_RUNNER, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(runner)], input=payload, capture_output=True, text=True,
                timeout=timeout, cwd=tmp, env=env,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(False, None, f"Délai dépassé ({timeout}s)", [])
    if proc.returncode != 0:
        err = proc.stderr.strip().splitlines()
        return SandboxResult(False, None, "\n".join(err[-6:]) or "Erreur inconnue", [])
    res = json.loads(proc.stdout)
    return SandboxResult(True, pd.DataFrame(res["records"], columns=res["columns"]), None, [])


def check_output_contract(out: pd.DataFrame, expected_columns: list[str],
                          bounds: dict[str, tuple[float | None, float | None]]) -> list[str]:
    """Contrat de sortie : colonnes exactes et ordonnées, numériques, sans NaN, bornes."""
    issues: list[str] = []
    if list(out.columns) != expected_columns:
        issues.append(f"Colonnes attendues {expected_columns}, obtenues {list(out.columns)}")
        return issues
    if len(out) != 1:
        issues.append(f"Une seule ligne attendue, {len(out)} obtenues")
    for col in expected_columns:
        s = pd.to_numeric(out[col], errors="coerce")
        if s.isna().any():
            issues.append(f"{col} : valeur manquante ou non numérique")
            continue
        lo, hi = bounds.get(col, (None, None))
        if lo is not None and (s < lo - 1e-9).any():
            issues.append(f"{col} < {lo}")
        if hi is not None and (s > hi + 1e-9).any():
            issues.append(f"{col} > {hi}")
    return issues


def describe(res: SandboxResult) -> dict[str, Any]:
    return {"ok": res.ok, "error": res.error, "violations": res.violations}
