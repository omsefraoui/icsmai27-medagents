"""Génère les notebooks « participant » (avec TODO) et « corrigé » à partir d'une source unique.

Usage : python scripts/build_notebooks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from nb import nb00, nb01, nb02, nb03, nb04  # noqa: E402

KERNEL = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python"}, "colab": {"provenance": [], "toc_visible": True},
          "accelerator": "GPU"}


def build(module, solution: bool) -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook(metadata=KERNEL)
    for cell in module.CELLS:
        kind = cell[0]
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(cell[1]))
        elif kind == "code":
            nb.cells.append(nbf.v4.new_code_cell(cell[1]))
        elif kind == "ex":
            src = cell[2] if solution else cell[1]
            header = "# ✅ CORRIGÉ\n" if solution else "# ✏️ À VOUS DE JOUER\n"
            nb.cells.append(nbf.v4.new_code_cell(header + src, metadata={"tags": ["exercise"]}))
    if solution:
        nb.cells[0].source = nb.cells[0].source.replace("\n", "  — **CORRIGÉ**\n", 1)
    return nb


def main() -> None:
    out_ex, out_sol = ROOT / "notebooks", ROOT / "notebooks" / "solutions"
    out_sol.mkdir(parents=True, exist_ok=True)
    for module in (nb00, nb01, nb02, nb03, nb04):
        has_ex = any(c[0] == "ex" for c in module.CELLS)
        nbf.write(build(module, solution=False), out_ex / f"{module.TITLE}.ipynb")
        if has_ex:
            nbf.write(build(module, solution=True), out_sol / f"{module.TITLE}_corrige.ipynb")
        print("✔", module.TITLE)


if __name__ == "__main__":
    main()
