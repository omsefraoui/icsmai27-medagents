from .common import LLM_CHECK, SETUP

TITLE = "04_module4_bout_en_bout_hitl"
CELLS = [
("md", """# Module 4 · Synthèse, évaluation & restitution
**Pipeline complet orchestré par LangGraph · Human-in-the-Loop · Évaluation · Déploiement**

⏱️ *50 min : 10 min de concepts + 30 min de lab + 10 min de discussion.*

À la fin de ce module, vous saurez :
- assembler les six agents en un **graphe à états** avec routage conditionnel et **mémoire persistante** (checkpointer) ;
- insérer des **points de contrôle humains** (`interrupt`) : revue de confidentialité, complétion de données, validation clinique ;
- produire une **piste d'audit** complète et **évaluer** le pipeline contre une vérité terrain ;
- raisonner sur le **déploiement hospitalier** (intégration, sécurité, réglementation)."""),
("code", SETUP),
("code", LLM_CHECK),
("md", """## 1. Le graphe
Chaque nœud lit et écrit un **état partagé** (`PipelineState`). Les arêtes conditionnelles routent selon le statut
(`PASS`/`BLOCKED`, modèle trouvé ou non). Les nœuds 🧑 appellent `interrupt()` : le graphe se met en pause,
l'état est sauvegardé, et l'exécution reprend avec `Command(resume=...)`."""),
("code", '''from medagents import config, graph as G
from medagents.viz import audit_df, matches_df, show_graph, show_report

app = G.build_graph()
show_graph(app)'''),
("md", """## 2. Démonstration — P001, du fichier brut au rapport validé"""),
("code", '''state, cfg = G.new_run(config.RAW_DIR / "dossier_P001.zip",
                       "Quel est le risque de réhospitalisation à 30 jours après cette décompensation ?",
                       thread_id="demo_P001")
out = app.invoke(state, cfg)
print("\\nStatut :", out["status"])
print("En attente d'une décision humaine :", G.pending_interrupt(app, cfg)["type"])'''),
("code", '''# Le relecteur lit le brouillon…
show_report(out["report"])'''),
("code", '''# … et le valide (ou le rejette) : la décision est tracée
out = G.resume(app, cfg, {"decision": "approve", "reviewer": "Dr A. Formateur",
                          "comment": "Cohérent avec la clinique ; télésurveillance proposée."})
print("Statut final :", out["status"])
audit_df(out["audit"])'''),
("code", '''# Artefacts produits : fichiers pseudonymisés, table de correspondance (à protéger), rapport, figure, piste d'audit
for p in sorted(Path(state["workdir"]).rglob("*")):
    if p.is_file() and "staging" not in p.parts:
        print(p.relative_to(state["workdir"]))'''),
("md", """## 3. Quand le pipeline doit s'arrêter — P003
P003 n'a pas de NT-proBNP (prélèvement hémolysé) et sa natrémie vaut **1380 mmol/L** (erreur de saisie).
Aucun modèle n'est applicable : le graphe **s'interrompt** et demande l'avis d'un humain."""),
("code", '''state3, cfg3 = G.new_run(config.RAW_DIR / "dossier_P003.zip", "", thread_id="demo_P003")
out3 = app.invoke(state3, cfg3)
print("\\nStatut :", out3["status"])
display(matches_df(out3["match"]))
G.pending_interrupt(app, cfg3)'''),
("md", """### ✏️ Exercice 4.1 — Jouer le clinicien (8 min)
Le médecin traitant vous transmet un **bilan de ville** : NT-proBNP **2 900 pg/mL**, natrémie contrôlée **138 mmol/L**.
1. Complétez la décision et reprenez le graphe. *(Une erreur ? Relancez la cellule « P003 » ci-dessus : un nouveau fil d'exécution est créé.)*
2. Lisez le rapport : où apparaît la trace de votre intervention ?
3. Validez ou rejetez le rapport, avec un commentaire."""),
("ex",
'''decision = {
    "action": "A_COMPLETER",       # TODO : "provide_values" ou "abort"
    "values": {},                  # TODO : {"nt_probnp": ..., "sodium": ...} en unités canoniques
    "source": "bilan de ville du 20/09/2026",
    "reviewer": "votre nom",
}
out3 = G.resume(app, cfg3, decision)
print("Statut :", out3["status"])

from medagents.exercises import check
check(out3["status"] == "REPORT_DRAFTED", "Le pipeline a repris jusqu'au rapport.")''',
'''decision = {
    "action": "provide_values",
    "values": {"nt_probnp": 2900, "sodium": 138},
    "source": "bilan de ville du 20/09/2026",
    "reviewer": "Dr B. Participant",
}
out3 = G.resume(app, cfg3, decision)
print("Statut :", out3["status"])

from medagents.exercises import check
check(out3["status"] == "REPORT_DRAFTED", "Le pipeline a repris jusqu'au rapport.")'''),
("ex",
'''show_report(out3["report"])
# TODO : relisez puis décidez
out3 = G.resume(app, cfg3, {"decision": "A_COMPLETER", "reviewer": "votre nom", "comment": "..."})
print("Statut final :", out3["status"])
audit_df(out3["audit"])[lambda d: d["acteur"] == "Humain"]''',
'''show_report(out3["report"])
out3 = G.resume(app, cfg3, {"decision": "approve", "reviewer": "Dr B. Participant",
                            "comment": "Patient de plus de 90 ans : âge agrégé, prudence sur l'extrapolation."})
print("Statut final :", out3["status"])
audit_df(out3["audit"])[lambda d: d["acteur"] == "Humain"]'''),
("md", """## 4. Évaluer le pipeline
Sur les trois dossiers, comparons les sorties à la vérité terrain : **rappel PHI**, **exactitude des features**,
**exactitude des pathologies** (présence et négation) et **choix du modèle**.

### ✏️ Exercice 4.2 — Le tableau de bord d'évaluation (8 min)"""),
("ex",
'''from medagents.evaluation import evaluate_run

etats = {"P001": out, "P003": out3}
# TODO : exécutez P002 avec la question « Risque de diabète à 5 ans ? », validez le rapport,
#        et ajoutez l'état final dans etats["P002"]

lignes = [{k: v for k, v in evaluate_run(pid, st).items() if not k.startswith("_")} for pid, st in etats.items()]
pd.DataFrame(lignes)''',
'''from medagents.evaluation import evaluate_run

etats = {"P001": out, "P003": out3}
s2, c2 = G.new_run(config.RAW_DIR / "dossier_P002.zip", "Risque de diabète à 5 ans ?", thread_id="demo_P002")
o2 = app.invoke(s2, c2)
etats["P002"] = G.resume(app, c2, {"decision": "approve", "reviewer": "Dr C. Évaluateur", "comment": "RAS"})

lignes = [{k: v for k, v in evaluate_run(pid, st).items() if not k.startswith("_")} for pid, st in etats.items()]
pd.DataFrame(lignes)'''),
("code", '''# Où le pipeline se trompe-t-il ? (détail des features pour chaque patient)
for pid, st in etats.items():
    r = evaluate_run(pid, st)
    erreurs = r["_features"][~r["_features"]["ok"]]
    print(pid, "— features en erreur :", "aucune" if erreurs.empty else erreurs.to_dict("records"))

# Coût en appels LLM
print(llm.stats())'''),
("md", """### ✏️ Exercice 4.3 — Un garde-fou de double validation (7 min)
Règle proposée par le comité qualité : *si le risque est **élevé** et qu'au moins une des **3 variables les plus
influentes** (SHAP) provient du **texte libre** (LLM) ou d'une **saisie humaine**, une seconde validation est exigée.*
Écrivez `needs_second_review(state)` et appliquez-la aux trois dossiers."""),
("ex",
'''def needs_second_review(state: dict) -> bool:
    pred = state["prediction"]
    feats = state["features"]["features"]
    top3 = [c["feature"] for c in pred["contributions"][:3]]
    # TODO : True si pred["risk_category"] == "élevé" ET une variable de top3 a source_kind "llm_extraction" ou "human"
    return False

for pid, st in etats.items():
    print(pid, st["prediction"]["risk_category"], [c["feature"] for c in st["prediction"]["contributions"][:3]],
          "→ double validation" if needs_second_review(st) else "→ validation simple")''',
'''def needs_second_review(state: dict) -> bool:
    pred = state["prediction"]
    feats = state["features"]["features"]
    top3 = [c["feature"] for c in pred["contributions"][:3]]
    fragile = any(feats.get(f, {}).get("source_kind") in ("llm_extraction", "human") for f in top3)
    return pred["risk_category"] == "élevé" and fragile

for pid, st in etats.items():
    print(pid, st["prediction"]["risk_category"], [c["feature"] for c in st["prediction"]["contributions"][:3]],
          "→ double validation" if needs_second_review(st) else "→ validation simple")'''),
("md", """**Pour aller plus loin** : transformez cette règle en **arête conditionnelle** du graphe
(`infer → second_review → clinical_review`) en vous inspirant de `route_model` dans `medagents/graph.py`.

## 5. Du notebook à l'hôpital — discussion (10 min)

| Sujet | Questions à se poser | Pistes |
|---|---|---|
| **Intégration** | Comment recevoir les données ? Où renvoyer le rapport ? | Serveur FHIR (`Bundle`, `DiagnosticReport`), flux HL7 v2 via moteur d'interfaces, PACS/DICOMweb |
| **Hébergement** | Où tournent le LLM et les modèles ? | GPU on-premise ou hébergeur certifié HDS ; LLM local (Ollama, vLLM) ; aucun flux sortant |
| **Sécurité** | Qui accède à quoi ? | Table de correspondance dans un coffre (KMS/HSM), RBAC, chiffrement, exécution de code en conteneur jetable |
| **Traçabilité** | Peut-on rejouer une décision ? | Piste d'audit horodatée et empreintes, versions des modèles et des prompts, registre de modèles (MLflow) |
| **Qualité** | Le système reste-t-il bon ? | Jeu de test annoté, suivi de dérive (données, LLM, modèles), revue des rejets et des interventions humaines |
| **Réglementation** | Quel statut ? | AI Act (système à haut risque), MDR (dispositif médical logiciel), RGPD (AIPD, base légale art. 9), HIPAA aux États-Unis |

## 🧭 Synthèse de l'atelier
1. **Déterministe d'abord, LLM ensuite** : parsing, règles, calculs et vérifications sont du code ; le LLM traite
   l'ambiguïté (texte libre, libellés, rédaction), toujours **contraint** et **vérifié**.
2. **Chaque agent a un contrat** (Pydantic) : c'est ce qui rend le pipeline testable, traçable et remplaçable brique par brique.
3. **L'humain est un nœud du graphe**, pas une option : ses décisions sont attendues, structurées et auditées.
4. **On évalue** chaque étage (PHI, extraction, choix du modèle, ancrage) contre une vérité terrain avant de parler de déploiement."""),
]
