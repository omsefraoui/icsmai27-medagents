from .common import LLM_CHECK, SETUP

TITLE = "02_module2_structuration_matching"
CELLS = [
("md", """# Module 2 · Structuration & matching modèle-données
**Agent 3 — Feature Extraction** · **Agent 4 — Model-Data Feature Matcher**

⏱️ *45 min : 15 min de concepts + 30 min de lab.*

À la fin de ce module, vous saurez :
- projeter des sources hétérogènes dans un **espace de features canonique** codé (LOINC, SNOMED-CT, CIM-10, ATC) ;
- faire extraire des données d'un texte libre par un LLM **sans le croire sur parole** (citation obligatoire, vérification) ;
- contraindre un LLM à une **liste fermée** de codes (Literal dynamique) ;
- vérifier automatiquement qu'un modèle pré-entraîné est **applicable** à un patient (inclusion, exclusion, couverture, domaine, fraîcheur).

```
fichiers pseudonymisés ──► [Agent 3] PatientFeatures (valeur + unité + code + source + preuve) ──► [Agent 4] MatchReport
```"""),
("code", SETUP),
("code", LLM_CHECK),
("code", '''# Rejouer le module 1 sur P001 (≈ 1 min)
from medagents import config
from medagents.agents.ingestion import IngestionAgent
from medagents.agents.anonymizer import AnonymizerAgent

WORK = config.OUTPUT_DIR / "module2_P001"
manifest = IngestionAgent(verbose=False).run(config.RAW_DIR / "dossier_P001.zip", WORK)
phi_report = AnonymizerAgent(verbose=False).run(manifest, WORK)
print(phi_report.status, phi_report.pseudo_id)'''),
("md", """## 1. Une terminologie, des unités
Le même examen arrive sous des noms et des unités différents : `Créatinine 141 µmol/L` (CSV), `2160-0^Creatinine^LN`
(HL7), `creat` dans un tableur… L'Agent 3 normalise vers **un nom canonique, un code et une unité**."""),
("code", '''from medagents.tools import terminology as term

for label in ["Créatinine (µmol/L)", "Natrémie", "HbA1c", "FEVG", "Pro-BNP N-terminal", "DFG CKD-EPI"]:
    print(f"{label:22s} →", term.map_observable(label))

print()
print("141 µmol/L  →", term.to_canonical("creatinine", 141, "µmol/L"))
print("6,2 mmol/L  →", term.to_canonical("fasting_glucose", 6.2, "mmol/L"))
print("43 mmol/mol →", term.to_canonical("hba1c", 43, "mmol/mol"))
print("FA          →", term.map_condition("FA"), "| ACFA →", term.map_condition("ACFA"))'''),
("md", """## 2. Agent 3 — extraction multi-sources
- **Structuré** (FHIR, HL7, CSV/Excel) : parsing déterministe, codes LOINC/SNOMED lus directement.
- **Texte libre** : le LLM renvoie des mesures **avec une citation**. Le code vérifie que la citation existe dans le texte
  **et** contient la valeur ; sinon, la mesure est rejetée.
- **Libellés inconnus** : le LLM choisit dans une **liste fermée** de codes (ou `AUCUN`).
- **Fusion** : mesure la plus récente, source structurée prioritaire, conflits tracés."""),
("code", '''from medagents.agents.features import FeatureExtractionAgent
from medagents.viz import features_df, conditions_df

fx = FeatureExtractionAgent()
pf = fx.run(manifest, phi_report)
features_df(pf)'''),
("code", '''display(conditions_df(pf))
print("Conflits :", [(c.name, c.values, c.resolution) for c in pf.conflicts])
print("Non associés :", pf.unmapped)
print("Rejets LLM (citation introuvable) :", pf.llm_rejected)'''),
("code", '''# Que s'est-il dit entre l'agent et le LLM ? (dernier appel d'extraction)
calls = [c for c in llm.history if c.agent.startswith("Agent 3")]
print(calls[0].response[:1500] if calls else "aucun appel")'''),
("md", """### ✏️ Exercice 2.1 — Un libellé de laboratoire inconnu (5 min)
Le CSV de P001 contient `Pro-BNP N-terminal`, que le dictionnaire ne reconnaît pas : c'est le LLM qui l'a associé
(regardez la colonne *preuve* : « libellé associé par LLM »). Deux stratégies :
- **déterministe** : ajouter un synonyme au dictionnaire (reproductible, auditable) ;
- **LLM sous contrainte** : liste fermée de codes (souple, mais à surveiller).

Ajoutez le synonyme qui permet au dictionnaire de reconnaître ce libellé."""),
("ex",
'''from medagents.exercises import check

NOUVEAU_SYNONYME = "a completer"  # TODO : forme normalisée (minuscules, sans accents ni tirets) du libellé
if NOUVEAU_SYNONYME not in term.OBSERVABLES["nt_probnp"].synonyms:
    term.OBSERVABLES["nt_probnp"].synonyms.append(NOUVEAU_SYNONYME)

check(term.map_observable("Pro-BNP N-terminal")[0] == "nt_probnp", "Libellé reconnu sans LLM.")''',
'''from medagents.exercises import check

NOUVEAU_SYNONYME = term.norm("Pro-BNP N-terminal")  # -> "pro bnp n terminal"
if NOUVEAU_SYNONYME not in term.OBSERVABLES["nt_probnp"].synonyms:
    term.OBSERVABLES["nt_probnp"].synonyms.append(NOUVEAU_SYNONYME)

check(term.map_observable("Pro-BNP N-terminal")[0] == "nt_probnp", "Libellé reconnu sans LLM.")'''),
("md", """### ✏️ Exercice 2.2 — Piéger l'extracteur (8 min)
Le texte ci-dessous contient trois pièges : une **négation**, un **antécédent familial** et une valeur **absente**
(le NT-proBNP n'est pas dosé). Lancez l'extraction et vérifiez que l'agent ne tombe dans aucun.
Puis inventez votre propre piège et observez."""),
("ex",
'''NOTE_PIEGE = """Consultation de suivi. Patiente de 67 ans. Pas de fibrillation atriale sur l'ECG de ce jour.
Père diabétique de type 2, la patiente n'est pas diabétique. FEVG à 52 %. PA 132/78 mmHg.
NT-proBNP non dosé (prélèvement hémolysé)."""

pf_test = FeatureExtractionAgent(verbose=False).extract_text(NOTE_PIEGE)
display(features_df(pf_test)); display(conditions_df(pf_test))

fa_attribuee = None        # TODO : True si pf_test.has_condition("atrial_fibrillation")
diabete_attribue = None    # TODO : idem pour "diabetes"
ntprobnp_invente = None    # TODO : True si "nt_probnp" figure dans pf_test.features

check(fa_attribuee is False and diabete_attribue is False and ntprobnp_invente is False,
      "Aucun piège n'a fonctionné.", "Complétez les TODO, ou l'agent s'est fait piéger : analysez pourquoi.")''',
'''NOTE_PIEGE = """Consultation de suivi. Patiente de 67 ans. Pas de fibrillation atriale sur l'ECG de ce jour.
Père diabétique de type 2, la patiente n'est pas diabétique. FEVG à 52 %. PA 132/78 mmHg.
NT-proBNP non dosé (prélèvement hémolysé)."""

pf_test = FeatureExtractionAgent(verbose=False).extract_text(NOTE_PIEGE)
display(features_df(pf_test)); display(conditions_df(pf_test))

fa_attribuee = pf_test.has_condition("atrial_fibrillation")
diabete_attribue = pf_test.has_condition("diabetes")
ntprobnp_invente = "nt_probnp" in pf_test.features
print("Rejets LLM :", pf_test.llm_rejected)

check(fa_attribuee is False and diabete_attribue is False and ntprobnp_invente is False,
      "Aucun piège n'a fonctionné.", "Complétez les TODO, ou l'agent s'est fait piéger : analysez pourquoi.")'''),
("md", """> 💡 Deux garde-fous travaillent ici : la **vérification de citation** (une valeur sans preuve textuelle est rejetée)
> et une **règle métier** (un antécédent familial ne devient jamais une pathologie du patient). Le prompt seul ne suffit pas.

## 3. Agent 4 — quel modèle pour ce patient ?
Le registre contient trois modèles, chacun décrit par une **carte de modèle** (variables, unités, domaine
d'entraînement, critères d'inclusion/exclusion, performances, limites)."""),
("code", '''from medagents.model_registry.registry import list_cards

for c in list_cards():
    print(f"■ {c.model_id} — {c.name}  (AUC {c.performance['auc']})")
    print("   requises :", [f.name for f in c.features if f.required])
    print("   inclusion :", c.inclusion_conditions, "| exclusion :", c.exclusion_conditions)'''),
("code", '''from medagents.agents.matcher import ModelMatcherAgent
from medagents.viz import matches_df

matcher = ModelMatcherAgent()
match = matcher.run(pf, clinical_question="Quel est le risque de réhospitalisation précoce après cette décompensation ?")
display(matches_df(match))
print("Modèle retenu :", match.selected_model_id)
print("Justification :", match.rationale)'''),
("md", """Le score de confiance combine **couverture des variables requises** (60 %), **optionnelles** (25 %) et **fraîcheur**
(15 %), pénalisé par les valeurs hors domaine. Le LLM ne choisit **que parmi les modèles déjà jugés compatibles**
par le code : il apporte la pertinence clinique, pas la vérification technique.

### ✏️ Exercice 2.3 — Une mesure périmée (7 min)
Le modèle de réhospitalisation exige un NT-proBNP de moins de 90 jours. Vieillissez artificiellement la mesure
de 200 jours et observez l'effet sur l'évaluation."""),
("ex",
'''from datetime import timedelta
from medagents.model_registry.registry import get_card

pf_vieux = pf.model_copy(deep=True)
# TODO : reculez pf_vieux.features["nt_probnp"].observed_at de 200 jours par rapport à pf_vieux.reference_date

m_vieux = matcher.evaluate(get_card("hf_readmission_30d"), pf_vieux)
print("Périmées :", m_vieux.stale, "| confiance :", m_vieux.confidence, "(avant :", match.matches[0].confidence, ")")
check(any(s.startswith("nt_probnp") for s in m_vieux.stale), "Fraîcheur contrôlée.")''',
'''from datetime import timedelta
from medagents.model_registry.registry import get_card

pf_vieux = pf.model_copy(deep=True)
pf_vieux.features["nt_probnp"].observed_at = pf_vieux.reference_date - timedelta(days=200)

m_vieux = matcher.evaluate(get_card("hf_readmission_30d"), pf_vieux)
print("Périmées :", m_vieux.stale, "| confiance :", m_vieux.confidence, "(avant :", match.matches[0].confidence, ")")
check(any(s.startswith("nt_probnp") for s in m_vieux.stale), "Fraîcheur contrôlée.")'''),
("md", """**À discuter** : une mesure périmée doit-elle rendre le modèle *incompatible* (blocage) ou seulement baisser la
confiance (avertissement) ? Qui doit en décider : l'équipe data, le clinicien, le comité de validation ?

### ✏️ Exercice 2.4 (bonus) — Le dossier P003
Rejouez les agents 1 à 4 sur P003. Pourquoi aucun modèle n'est-il compatible ? Que faudrait-il pour débloquer
la situation ? *(Nous traiterons ce cas avec un humain dans la boucle au module 4.)*"""),
("ex",
'''w3 = config.OUTPUT_DIR / "module2_P003"
# TODO : enchaînez IngestionAgent → AnonymizerAgent → FeatureExtractionAgent → ModelMatcherAgent sur P003
#        puis affichez matches_df(...) et les conflits''',
'''w3 = config.OUTPUT_DIR / "module2_P003"
m3 = IngestionAgent(verbose=False).run(config.RAW_DIR / "dossier_P003.zip", w3)
r3 = AnonymizerAgent(verbose=False).run(m3, w3)
pf3 = FeatureExtractionAgent(verbose=False).run(m3, r3)
match3 = ModelMatcherAgent(verbose=False).run(pf3)
display(matches_df(match3))
print("Conflits :", [(c.name, c.values) for c in pf3.conflicts])
print(match3.rationale)'''),
("md", """## 🧭 Synthèse du module 2
- Un espace de features **canonique et codé** est le contrat central du pipeline : chaque valeur porte son unité,
  son code, sa source et sa **preuve**.
- Le LLM est utile là où les règles échouent (texte libre, libellés exotiques), à condition d'être **contraint**
  (schéma, liste fermée) et **vérifié** (citation, règle métier).
- Le matching modèle-données est d'abord **déterministe** : inclusion/exclusion, couverture, domaine, fraîcheur.
  Le LLM n'arbitre qu'entre options valides.

➡️ Module 3 : préparer le vecteur d'entrée, exécuter le modèle et **expliquer** sa prédiction."""),
]
