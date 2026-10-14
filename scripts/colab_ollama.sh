#!/usr/bin/env bash
# Installe et démarre Ollama dans Google Colab (ou toute VM Linux), puis télécharge le modèle.
# Usage : bash scripts/colab_ollama.sh [modele]   (défaut : llama3.1:8b)
set -euo pipefail
MODEL="${1:-llama3.1:8b}"
if ! command -v ollama >/dev/null 2>&1; then
  echo "▶ Installation d'Ollama…"
  (apt-get -qq update && apt-get -qq install -y zstd pciutils lshw) >/dev/null 2>&1 || true
  curl -fsSL https://ollama.com/install.sh | sh >/dev/null
fi
if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "▶ Démarrage du serveur Ollama…"
  setsid nohup ollama serve > "${OLLAMA_LOG:-/tmp/ollama.log}" 2>&1 < /dev/null &
  for i in $(seq 1 30); do curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
fi
echo "▶ Téléchargement du modèle ${MODEL} (quelques minutes la première fois)…"
ollama pull "${MODEL}" >/dev/null
ollama list
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total --format=csv || echo "⚠️ Pas de GPU détecté : exécution CPU (lente). Préférez Exécution > Modifier le type d'exécution > GPU T4."
