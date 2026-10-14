"""Tests de non-régression (LLM simulé) : python -m pytest -q tests"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fake_llm import FakeLLM  # noqa: E402

from medagents import graph as G  # noqa: E402
from medagents.datagen import ensure_data  # noqa: E402
from medagents.evaluation import evaluate_run  # noqa: E402
from medagents.llm import set_llm  # noqa: E402
from medagents.model_registry.registry import ensure_models  # noqa: E402
from medagents.tools import phi, sandbox  # noqa: E402
from medagents.tools.clinical import egfr_ckd_epi_2021  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _setup():
    set_llm(FakeLLM())
    G.VERBOSE = False
    ensure_data()
    ensure_models(verbose=False)


def test_egfr_reference():
    assert egfr_ckd_epi_2021(1.59, 75, False) == 45.0


def test_phi_regex_detects_nir_and_phone():
    spans = phi.find_regex("NIR 1 51 03 35 238 041 27, tél. 06 12 34 56 78")
    cats = {s.category.value for s in spans}
    assert {"NUM_SECU", "TELEPHONE"} <= cats


def test_sandbox_blocks_os():
    assert sandbox.validate_code("import os\ndef preprocess(df):\n    return df")


@pytest.mark.parametrize("pid", ["P001", "P002", "P003"])
def test_end_to_end(pid):
    app = G.build_graph()
    st, cfg = G.new_run(f"data/raw/dossier_{pid}.zip", "", thread_id=f"test_{pid}")
    out = app.invoke(st, cfg)
    if out["status"] == "NO_COMPATIBLE_MODEL":
        out = G.resume(app, cfg, {"action": "provide_values", "values": {"nt_probnp": 2900, "sodium": 138},
                                  "source": "test", "reviewer": "pytest"})
    assert out["status"] == "REPORT_DRAFTED"
    out = G.resume(app, cfg, {"decision": "approve", "reviewer": "pytest"})
    assert out["status"] == "VALIDATED"
    r = evaluate_run(pid, out)
    assert r["phi_rappel"] == 1.0 and r["modele_ok"]
