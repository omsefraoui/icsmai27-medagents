"""Configuration centrale de l'atelier.

Toutes les valeurs peuvent être surchargées par variables d'environnement,
ce qui permet de pointer vers un serveur Ollama partagé (salle de formation)
sans modifier le code :

    export OLLAMA_HOST=http://192.168.1.20:11434
    export MEDAGENTS_MODEL=llama3.2:3b
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent

DATA_DIR = Path(os.getenv("MEDAGENTS_DATA_DIR", ROOT_DIR / "data"))
RAW_DIR = DATA_DIR / "raw"
COHORTS_DIR = DATA_DIR / "cohorts"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
REGISTRY_DIR = PACKAGE_DIR / "model_registry"
MODELS_DIR = Path(os.getenv("MEDAGENTS_MODELS_DIR", ROOT_DIR / "models"))
OUTPUT_DIR = Path(os.getenv("MEDAGENTS_OUTPUT_DIR", ROOT_DIR / "outputs"))

# --- LLM (Ollama, exécution 100 % locale) ---------------------------------
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL = os.getenv("MEDAGENTS_MODEL", "llama3.1:8b")
LLM_TEMPERATURE = float(os.getenv("MEDAGENTS_TEMPERATURE", "0"))
LLM_NUM_CTX = int(os.getenv("MEDAGENTS_NUM_CTX", "8192"))
LLM_TIMEOUT_S = float(os.getenv("MEDAGENTS_LLM_TIMEOUT", "180"))
LLM_SEED = 42

# --- Confidentialité -------------------------------------------------------
# Secret de pseudonymisation (HMAC). En production : coffre-fort (KMS/HSM),
# jamais dans le code ni dans le dépôt.
PSEUDONYM_SECRET = os.getenv("MEDAGENTS_SECRET", "icsmai27-atelier-NE-PAS-UTILISER-EN-PROD")
DATE_SHIFT_MAX_DAYS = 180

# --- Bac à sable d'exécution de code --------------------------------------
SANDBOX_TIMEOUT_S = 15
SANDBOX_MAX_ATTEMPTS = 3

for _d in (OUTPUT_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
