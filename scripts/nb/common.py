"""Cellules partagées par tous les notebooks."""

SETUP = '''# ⚙️ Cellule 0 — Configuration de l'environnement (à exécuter en premier)
import os, sys, subprocess
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/omsefraoui/icsmai27-medagents.git"
os.environ.setdefault("MEDAGENTS_MODEL", "llama3.1:8b")   # ou "llama3.2:3b" sur une machine modeste
# Option C — serveur Ollama partagé de la salle : décommentez et renseignez l'adresse donnée par le formateur
# os.environ["OLLAMA_HOST"] = "http://192.168.1.20:11434"

if IN_COLAB:
    if not Path("/content/icsmai27-medagents").exists():
        subprocess.run(["git", "clone", "-q", REPO_URL, "/content/icsmai27-medagents"], check=True)
    os.chdir("/content/icsmai27-medagents")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
    if "OLLAMA_HOST" not in os.environ:
        subprocess.run(["bash", "scripts/colab_ollama.sh", os.environ["MEDAGENTS_MODEL"]], check=True)

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "medagents").is_dir())
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import warnings; warnings.filterwarnings("ignore")
import pandas as pd; pd.set_option("display.max_colwidth", 80)
print("Racine du projet :", ROOT)'''

LLM_CHECK = '''# 🔌 Connexion au LLM local (Ollama)
from medagents.llm import get_llm, pretty
from medagents.datagen import ensure_data
from medagents.model_registry.registry import ensure_models

llm = get_llm()
status = llm.health_check()
print(pretty(status))
assert status["ok"], "Ollama n'est pas prêt : voir le notebook 00 ou la fiche participant (section Dépannage)."
ensure_data(); ensure_models()'''
