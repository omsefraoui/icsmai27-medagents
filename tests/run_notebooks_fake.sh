#!/usr/bin/env bash
# Exécute les notebooks avec le LLM SIMULÉ (tests de non-régression, CI sans GPU).
# Usage : bash tests/run_notebooks_fake.sh [solutions|participant]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export IPYTHONDIR="$(mktemp -d)"
mkdir -p "$IPYTHONDIR/profile_default/startup"
cat > "$IPYTHONDIR/profile_default/startup/00_fake_llm.py" <<PY
import sys
sys.path[:0] = ["$ROOT", "$ROOT/tests"]
from fake_llm import FakeLLM
import medagents.llm as L
L.set_llm(FakeLLM())
print("⚠️ LLM SIMULÉ (tests) — pas d'Ollama")
PY
OUT="${OUT_DIR:-$ROOT/outputs/executed_notebooks}"; mkdir -p "$OUT"
MODE="${1:-solutions}"
if [ "$MODE" = "solutions" ]; then NBS=("$ROOT/notebooks/00_installation_ollama.ipynb" "$ROOT"/notebooks/solutions/*.ipynb); EXTRA=""
else NBS=("$ROOT"/notebooks/0*.ipynb); EXTRA="--allow-errors"; fi
for nb in "${NBS[@]}"; do
  echo "▶ $(basename "$nb")"
  (cd "$(dirname "$nb")" && jupyter nbconvert --to notebook --execute $EXTRA --ExecutePreprocessor.timeout=600 \
      --output-dir "$OUT" "$nb" 2>&1 | tail -3) || echo "   ❌ échec"
done
