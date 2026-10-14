from .common import LLM_CHECK, SETUP

TITLE = "03_module3_pretraitement_inference_xai"
CELLS = [
("md", """# Module 3 · Prétraitement, inférence & explicabilité clinique
**Agent 5 — Preprocessing Recommender & Implementor** · **Agent 6 — Model Inference with Explainability**

⏱️ *50 min : 15 min de concepts + 35 min de lab.*

À la fin de ce module, vous saurez :
- dériver un plan de prétraitement **à partir de la carte du modèle** (et éviter le piège du double *scaling*) ;
- laisser un LLM **écrire du code** sans lui faire confiance : analyse statique, exécution isolée, contrat de sortie,
  **test différentiel** ;
- produire une prédiction avec **incertitude**, des **valeurs de SHAP** et un rapport dont **chaque chiffre est vérifié**.

```
PatientFeatures + MatchReport ──► [Agent 5] vecteur modèle (plan + code vérifié) ──► [Agent 6] risque, IC, SHAP, rapport ancré
```"""),
("code", SETUP),
("code", LLM_CHECK),
("code", '''# Rejouer les modules 1-2 sur P001 (≈ 1-2 min)
from medagents import config
from medagents.agents.ingestion import IngestionAgent
from medagents.agents.anonymizer import AnonymizerAgent
from medagents.agents.features import FeatureExtractionAgent
from medagents.agents.matcher import ModelMatcherAgent

WORK = config.OUTPUT_DIR / "module3_P001"
manifest = IngestionAgent(verbose=False).run(config.RAW_DIR / "dossier_P001.zip", WORK)
phi_report = AnonymizerAgent(verbose=False).run(manifest, WORK)
pf = FeatureExtractionAgent(verbose=False).run(manifest, phi_report)
match = ModelMatcherAgent(verbose=False).run(pf, "Risque de réhospitalisation à 30 jours ?")
print(match.selected_model_id)'''),
("md", """## 1. Agent 5 — le plan de prétraitement
Le plan est **déduit de la carte du modèle**, pas improvisé : conserver, dériver, imputer (médiane d'entraînement,
uniquement pour les variables *optionnelles*), encoder, borner au domaine d'entraînement.

⚠️ **Pas de standardisation** : le `StandardScaler` est déjà dans le pipeline du modèle. Standardiser deux fois ne
produit aucune erreur… seulement des prédictions fausses."""),
("code", '''from medagents.agents.preprocess import PreprocessingAgent
from medagents.model_registry.registry import get_card

card = get_card(match.selected_model_id)
pre_agent = PreprocessingAgent()
steps, df_in = pre_agent.recommend(card, pf)
display(pd.DataFrame([s.model_dump() for s in steps]))
df_in'''),
("md", """### ✏️ Exercice 3.1 — Implémenter le DFG CKD-EPI 2021 (7 min)
Le DFG n'est pas dans le dossier : il est **dérivé** de la créatinine, de l'âge et du sexe (équation CKD-EPI 2021,
sans coefficient ethnique) :

$$\\text{DFG} = 142 \\times \\min(S_{cr}/\\kappa, 1)^{\\alpha} \\times \\max(S_{cr}/\\kappa, 1)^{-1{,}200} \\times 0{,}9938^{\\text{âge}} \\times 1{,}012\\,[\\text{si femme}]$$

avec $\\kappa = 0{,}7$ (F) / $0{,}9$ (H), $\\alpha = -0{,}241$ (F) / $-0{,}302$ (H), $S_{cr}$ en mg/dL. Arrondir à 0,1."""),
("ex",
'''from medagents.exercises import check
from medagents.tools.clinical import egfr_ckd_epi_2021  # référence

def mon_dfg(creatinine_mg_dl: float, age: float, femme: bool) -> float:
    # TODO
    return 0.0

cas = [(1.59, 75, False), (0.8, 51, True), (1.15, 90, False), (0.6, 30, True)]
for c in cas:
    print(c, "→", mon_dfg(*c), "| attendu", egfr_ckd_epi_2021(*c))
check(all(abs(mon_dfg(*c) - egfr_ckd_epi_2021(*c)) < 0.11 for c in cas), "Formule exacte.")''',
'''from medagents.exercises import check
from medagents.tools.clinical import egfr_ckd_epi_2021  # référence

def mon_dfg(creatinine_mg_dl: float, age: float, femme: bool) -> float:
    kappa, alpha = (0.7, -0.241) if femme else (0.9, -0.302)
    r = creatinine_mg_dl / kappa
    dfg = 142 * min(r, 1) ** alpha * max(r, 1) ** -1.200 * 0.9938 ** age * (1.012 if femme else 1.0)
    return round(dfg, 1)

cas = [(1.59, 75, False), (0.8, 51, True), (1.15, 90, False), (0.6, 30, True)]
for c in cas:
    print(c, "→", mon_dfg(*c), "| attendu", egfr_ckd_epi_2021(*c))
check(all(abs(mon_dfg(*c) - egfr_ckd_epi_2021(*c)) < 0.11 for c in cas), "Formule exacte.")'''),
("md", """## 2. Exécuter du code écrit par un LLM… prudemment
Quatre barrières successives :
1. **analyse statique (AST)** : imports en liste blanche (`pandas`, `numpy`, `math`), pas d'E/S, pas de `eval`, pas d'attribut `_…` ;
2. **exécution isolée** : sous-processus Python en mode isolé, délai maximal, limites CPU/mémoire ;
3. **contrat de sortie** : colonnes exactes et ordonnées, numériques, sans valeur manquante, dans le domaine ;
4. **test différentiel** : le résultat doit égaler celui d'une implémentation de référence générée depuis le plan.

> ⚠️ Les barrières 1-2 sont pédagogiques. En production : conteneur jetable sans réseau (gVisor, Firecracker…)."""),
("code", '''from medagents.tools import sandbox

sain = "import numpy as np\\ndef preprocess(df):\\n    return df.assign(x=np.log(df['age']))"
print("Code sain     :", sandbox.validate_code(sain) or "accepté")
print("Code dangereux:", sandbox.validate_code("import os\\ndef preprocess(df):\\n    os.system('rm -rf /')\\n    return df"))'''),
("md", """### ✏️ Exercice 3.2 — *Red team* du bac à sable (8 min)
Voici des tentatives d'évasion. Ajoutez-en **deux de votre invention**, puis exécutez : chacune doit être bloquée,
soit par l'analyse statique, soit à l'exécution (délai, erreur)."""),
("ex",
'''ATTAQUES = {
    "import interdit": "import subprocess\\ndef preprocess(df):\\n    return df",
    "lecture de fichier": "import pandas as pd\\ndef preprocess(df):\\n    return pd.read_csv('/etc/passwd')",
    "évasion par dunder": "def preprocess(df):\\n    return ().__class__.__base__.__subclasses__()",
    "boucle infinie déguisée": "def preprocess(df):\\n    for i in range(10**12):\\n        pass\\n    return df",
    # TODO : ajoutez deux attaques
}
bloquees = {}
for nom, code in ATTAQUES.items():
    res = sandbox.run_sandboxed(code, df_in, timeout=5)
    bloquees[nom] = not res.ok
    print(f"{'🛡️' if not res.ok else '💥'} {nom:28s} → {res.violations or res.error}")
check(len(ATTAQUES) >= 6 and all(bloquees.values()), "Toutes les attaques sont bloquées.",
      "Ajoutez vos attaques, ou une est passée : laquelle, et quelle barrière ajouter ?")''',
'''ATTAQUES = {
    "import interdit": "import subprocess\\ndef preprocess(df):\\n    return df",
    "lecture de fichier": "import pandas as pd\\ndef preprocess(df):\\n    return pd.read_csv('/etc/passwd')",
    "évasion par dunder": "def preprocess(df):\\n    return ().__class__.__base__.__subclasses__()",
    "boucle infinie déguisée": "def preprocess(df):\\n    for i in range(10**12):\\n        pass\\n    return df",
    "eval caché": "def preprocess(df):\\n    return eval('df')",
    "écriture disque": "def preprocess(df):\\n    df.to_csv('/tmp/fuite.csv')\\n    return df",
}
bloquees = {}
for nom, code in ATTAQUES.items():
    res = sandbox.run_sandboxed(code, df_in, timeout=5)
    bloquees[nom] = not res.ok
    print(f"{'🛡️' if not res.ok else '💥'} {nom:28s} → {res.violations or res.error}")
check(len(ATTAQUES) >= 6 and all(bloquees.values()), "Toutes les attaques sont bloquées.",
      "Ajoutez vos attaques, ou une est passée : laquelle, et quelle barrière ajouter ?")'''),
("code", '''# L'agent complet : le LLM écrit le code, les barrières le vérifient, l'erreur lui est renvoyée si besoin
pre = pre_agent.run(pf, match)
for a in pre.attempts:
    print(f"Essai {a['attempt']} : {'✅' if a['ok'] else '❌'} {a.get('issues') or ''}")
print("\\nOrigine du code retenu :", pre.code_origin)
print(pre.code)
pd.Series(pre.output_row, name="vecteur d'entrée du modèle").to_frame()'''),
("md", """## 3. Agent 6 — prédire, quantifier l'incertitude, expliquer
- **Incertitude** : 20 modèles ré-entraînés par *bootstrap* → intervalle 2,5 %–97,5 % des prédictions
  (incertitude *du modèle*, pas un intervalle de confiance au sens épidémiologique).
- **SHAP** : contribution de chaque variable au log-odds, par rapport à la valeur de base. Exact pour les modèles
  linéaires (`LinearExplainer`) et les arbres (`TreeExplainer`).
- ⚠️ SHAP explique **le modèle**, pas **la biologie** : ce n'est pas une relation de cause à effet."""),
("code", '''from medagents.agents.inference import InferenceAgent

inf = InferenceAgent()
pred = inf.predict(pre, figdir=WORK / "rapport")
print(f"Risque : {pred.probability:.1%}  [{pred.ci_low:.1%} ; {pred.ci_high:.1%}]  → {pred.risk_category}")
display(pd.DataFrame([c.model_dump() for c in pred.contributions]).head(8))
from IPython.display import Image
Image(filename=pred.figure_path)'''),
("md", """### ✏️ Exercice 3.3 — Scénario « et si… ? » (7 min)
Imaginez que le traitement ramène le NT-proBNP à 900 pg/mL et la natrémie à 140 mmol/L.
Modifiez le vecteur d'entrée, relancez la prédiction et comparez risques et contributions SHAP.
*Que peut-on — et que ne peut-on pas — conclure de cette comparaison ?*"""),
("ex",
'''row = dict(pre.output_row)
# TODO : modifiez row["nt_probnp"] et row["sodium"]

pred2 = inf.predict(pre.model_copy(update={"output_row": row}))
print(f"Avant : {pred.probability:.1%} → après : {pred2.probability:.1%}")
avant = {c.feature: c.shap for c in pred.contributions}
display(pd.DataFrame([{"variable": c.feature, "SHAP avant": round(avant[c.feature], 3), "SHAP après": round(c.shap, 3)}
                      for c in pred2.contributions]).head(6))
check(pred2.probability < pred.probability, "Le risque estimé baisse.")''',
'''row = dict(pre.output_row)
row["nt_probnp"] = 900.0
row["sodium"] = 140.0

pred2 = inf.predict(pre.model_copy(update={"output_row": row}))
print(f"Avant : {pred.probability:.1%} → après : {pred2.probability:.1%}")
avant = {c.feature: c.shap for c in pred.contributions}
display(pd.DataFrame([{"variable": c.feature, "SHAP avant": round(avant[c.feature], 3), "SHAP après": round(c.shap, 3)}
                      for c in pred2.contributions]).head(6))
check(pred2.probability < pred.probability, "Le risque estimé baisse.")'''),
("md", """## 4. Le rapport clinique et la vérification d'ancrage
Les **chiffres** du rapport sont insérés par le code depuis une *fiche de faits* ; le LLM ne rédige que la prose,
**à partir de cette fiche**. Ensuite, tout nombre écrit par le LLM doit exister dans la fiche — sinon régénération,
puis signalement au relecteur."""),
("code", '''report = inf.report(pf, match, pre, pred, outdir=WORK / "rapport")
print("Ancrage OK :", report.grounding_ok, report.unsupported_numbers)
from medagents.viz import show_report
show_report(report)'''),
("code", '''# La vérification d'ancrage en action
from medagents.agents.inference import grounding_check
from medagents.schemas import ReportNarrativeLLM

facts = inf.facts(pf, match, pre, pred)
faux = ReportNarrativeLLM(synthese=f"Risque de {facts['resultat']['risque_pct']} %, soit 5 fois la moyenne nationale de 17,77 %.",
                          facteurs_explicatifs=[], limites=[], points_a_verifier=[])
print("Chiffres non ancrés :", grounding_check(faux, facts))'''),
("md", """### ✏️ Exercice 3.4 (bonus) — Un contrôle de style réglementaire
Un outil d'aide à la décision ne doit ni poser de **diagnostic**, ni **prescrire**, ni afficher de **certitude**.
Écrivez `style_check(narrative)` qui renvoie la liste des expressions interdites trouvées, puis testez-la."""),
("ex",
'''INTERDITS = ["diagnostic", "prescrire", "augmenter la dose", "certainement", "il faut traiter"]

def style_check(narrative) -> list[str]:
    # TODO : concaténer synthese + facteurs + limites + points, chercher (insensible à la casse) chaque expression
    return []

ok_txt = ReportNarrativeLLM(synthese="Risque estimé élevé.", facteurs_explicatifs=[], limites=[], points_a_verifier=[])
ko_txt = ReportNarrativeLLM(synthese="Diagnostic : réhospitalisation certainement à venir.", facteurs_explicatifs=[],
                            limites=[], points_a_verifier=["Prescrire un diurétique."])
print(style_check(ok_txt), style_check(ko_txt))
check(style_check(ok_txt) == [] and {"diagnostic", "certainement", "prescrire"} <= set(style_check(ko_txt)),
      "Contrôle de style opérationnel.")''',
'''INTERDITS = ["diagnostic", "prescrire", "augmenter la dose", "certainement", "il faut traiter"]

def style_check(narrative) -> list[str]:
    texte = " ".join([narrative.synthese, *narrative.facteurs_explicatifs, *narrative.limites,
                      *narrative.points_a_verifier]).lower()
    return [mot for mot in INTERDITS if mot in texte]

ok_txt = ReportNarrativeLLM(synthese="Risque estimé élevé.", facteurs_explicatifs=[], limites=[], points_a_verifier=[])
ko_txt = ReportNarrativeLLM(synthese="Diagnostic : réhospitalisation certainement à venir.", facteurs_explicatifs=[],
                            limites=[], points_a_verifier=["Prescrire un diurétique."])
print(style_check(ok_txt), style_check(ko_txt))
check(style_check(ok_txt) == [] and {"diagnostic", "certainement", "prescrire"} <= set(style_check(ko_txt)),
      "Contrôle de style opérationnel.")'''),
("md", """## 🧭 Synthèse du module 3
- Le **plan** vient de la carte du modèle ; le LLM n'en écrit que l'**implémentation**, vérifiée par quatre barrières.
  Le test différentiel est la plus forte : le code généré doit égaler une référence exécutable.
- Une prédiction clinique se livre avec son **incertitude**, son **explication** et la **provenance** de chaque entrée.
- Séparer **chiffres (code)** et **prose (LLM)**, puis vérifier l'ancrage : c'est ainsi qu'on évite les rapports
  « plausibles mais faux ».

➡️ Module 4 : orchestrer les six agents, insérer l'humain dans la boucle, évaluer, déployer."""),
]
