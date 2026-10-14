"""medagents — pipeline multi-agents médical de bout en bout (atelier ICSMAI 2027).

Du fichier brut au rapport clinique explicable :
  Agent 1 Ingestion → Agent 2 Privacy Guard → Agent 3 Feature Extraction →
  Agent 4 Model Matcher → Agent 5 Preprocessing → Agent 6 Inference & XAI
orchestrés par LangGraph, avec un LLM 100 % local (Ollama).

⚠️ Pédagogique : données et modèles synthétiques, sans validité clinique.
"""
__version__ = "1.0.0"
