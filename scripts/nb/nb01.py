from .common import LLM_CHECK, SETUP

TITLE = "01_module1_ingestion_confidentialite"
CELLS = [
("md", """# Module 1 · Ingestion & sécurisation des données
**Agent 1 — Ingestion Identifier** · **Agent 2 — Data Anonymizer (Privacy Guard)**

⏱️ *45 min : 15 min de concepts (slides) + 30 min de lab.*

À la fin de ce module, vous saurez :
- identifier et valider automatiquement des fichiers cliniques hétérogènes (DICOM, FHIR, HL7 v2, CSV, Excel, texte, ZIP) ;
- construire une **défense en profondeur** contre les fuites de PHI : identifiants connus → règles → LLM local → contrôle résiduel ;
- distinguer **pseudonymisation** (RGPD) et **anonymisation**, et lire une check-list HIPAA Safe Harbor.

```
dossier brut ──► [Agent 1] manifeste (type, validité, route) ──► [Agent 2] fichiers pseudonymisés + rapport PHI ──► PASS / BLOCKED
```"""),
("code", SETUP),
("code", LLM_CHECK),
("md", """## 1. Ce que reçoit le pipeline
Regardons le compte rendu **brut** du patient P001 (fictif). Repérez à l'œil toutes les informations identifiantes."""),
("code", '''import zipfile
from medagents import config

with zipfile.ZipFile(config.RAW_DIR / "dossier_P001.zip") as z:
    print(z.read("comptes_rendus/CR_hospitalisation_cardiologie.txt").decode("utf-8"))'''),
("md", """## 2. Agent 1 — identifier, valider, router
Ordre de détection : **octets magiques** (DICOM `DICM` à l'octet 128, ZIP `PK\\x03\\x04`) → **contenu** (`MSH|` pour HL7,
`resourceType` pour FHIR, dialecte CSV) → **extension** en dernier recours.
Les archives sont ouvertes récursivement avec protection *zip-slip* et *zip bomb*.
Le LLM local qualifie ensuite les textes libres (type de document, spécialité, langue)."""),
("code", '''from medagents.agents.ingestion import IngestionAgent
from medagents.viz import manifest_df

WORK = config.OUTPUT_DIR / "module1_P001"
manifest = IngestionAgent().run(config.RAW_DIR / "dossier_P001.zip", WORK)
manifest_df(manifest)'''),
("code", '''# Métadonnées extraites (non identifiantes) pour chaque fichier
for f in manifest.files:
    print(f"{f.file_id} {f.file_type.value:14s}", {k: v for k, v in f.metadata.items() if k != "columns"})'''),
("md", """### ✏️ Exercice 1.1 — La quarantaine (5 min)
Lancez l'Agent 1 sur le dossier **P003** et construisez la liste des fichiers **mis en quarantaine**
(aucune route). Pourquoi ce fichier a-t-il été écarté ? Avec quelle méthode de détection ?"""),
("ex",
'''manifest_p003 = IngestionAgent().run(config.RAW_DIR / "dossier_P003.zip", config.OUTPUT_DIR / "module1_P003")

quarantaine = []  # TODO : noms (f.original_name.split("/")[-1]) des fichiers dont f.route_to est vide

from medagents.exercises import check
check(quarantaine == ["scan_courrier_illisible.bin"], "Fichier corrompu correctement isolé.")''',
'''manifest_p003 = IngestionAgent().run(config.RAW_DIR / "dossier_P003.zip", config.OUTPUT_DIR / "module1_P003")

quarantaine = [f.original_name.split("/")[-1] for f in manifest_p003.files if not f.route_to]
for f in manifest_p003.files:
    if not f.route_to:
        print(f.original_name, f.file_type.value, f.detection_method, f.confidence)

from medagents.exercises import check
check(quarantaine == ["scan_courrier_illisible.bin"], "Fichier corrompu correctement isolé.")'''),
("md", """## 3. Agent 2 — Privacy Guard
Quatre couches successives :
1. **identifiants connus** récoltés dans les champs structurés (FHIR `Patient`, HL7 `PID`/`PV1`, DICOM) puis recherchés partout ;
2. **expressions régulières** (NIR, téléphone, e-mail, dates, IPP, adresses, « Dr/M./Mme + Nom ») ;
3. **NER par LLM local** sur le texte libre — toute entité absente *mot pour mot* du texte est **rejetée** (hallucination) ;
4. **contrôle résiduel** sur toutes les sorties : s'il reste un identifiant fort → statut `BLOCKED`.

Pseudonymisation : jetons HMAC stables, **décalage de dates cohérent** par patient, âges > 89 ans agrégés.

Commençons par les **règles seules** (`use_llm=False`)."""),
("code", '''from medagents.agents.anonymizer import AnonymizerAgent
from medagents.viz import phi_df

rep_rules = AnonymizerAgent(use_llm=False).run(manifest, WORK)
print("Statut :", rep_rules.status, "| résiduel :", rep_rules.residual_phi)
phi_df(rep_rules).groupby(["catégorie", "détecteur"]).size().unstack(fill_value=0)'''),
("code", '''# Puis avec la couche LLM
rep = AnonymizerAgent().run(manifest, WORK)
print("Statut :", rep.status)
print("Entités proposées par le LLM puis REJETÉES (absentes du texte ou termes médicaux) :", rep.llm_rejected)
phi_df(rep)[lambda d: d["détecteur"] == "llm"]'''),
("code", '''# Avant / après sur le compte rendu
note_id = next(f.file_id for f in manifest.files if f.file_type.value == "clinical_note")
print(open(rep.deid_files[note_id], encoding="utf-8").read()[:1200])'''),
("code", '''# Le décalage de dates préserve les intervalles (durée de séjour, délais) : indispensable pour la suite
import json
from datetime import date
vault = json.loads((WORK / "vault" / "table_correspondance.json").read_text())
print("Pseudonyme :", rep.pseudo_id, "| décalage secret :", vault["date_shift_days"], "jours")
print("Durée de séjour réelle :", (date(2026, 9, 1) - date(2026, 8, 22)).days, "jours — identique après décalage.")'''),
("code", '''# Check-list HIPAA Safe Harbor (18 identifiants) et notes RGPD
display(pd.Series(rep.hipaa_checklist, name="statut").to_frame())
for n in rep.gdpr_notes:
    print("•", n)'''),
("md", """### ✏️ Exercice 1.2 — Un détecteur pour les numéros de mutuelle (8 min)
Le dossier **P002** contient « Mutuelle n° MUT-558201 ». Sans le LLM, ce numéro fuit.
1. Écrivez une expression régulière qui reconnaît `MUT-558201` **et** `N° adhérent : 99887766`, sans attraper `HbA1c 43 mmol/mol`.
2. Enregistrez-la dans l'Agent 2 avec `register_pattern`, relancez l'agent **sans LLM** et vérifiez qu'il n'y a plus de fuite."""),
("ex",
'''import re
from medagents.exercises import check, register_pattern
from medagents.schemas import PHICategory

MUTUELLE_RE = re.compile(r"A_COMPLETER")  # TODO

tests_positifs = ["Mutuelle n° MUT-558201", "N° adhérent : 99887766"]
tests_negatifs = ["HbA1c 43 mmol/mol", "MUT seul", "PA 138/86 mmHg"]
check(all(MUTUELLE_RE.search(t) for t in tests_positifs) and not any(MUTUELLE_RE.search(t) for t in tests_negatifs),
      "La regex fait le travail.", "La regex rate un positif ou attrape un négatif.")''',
'''import re
from medagents.exercises import check, register_pattern
from medagents.schemas import PHICategory

MUTUELLE_RE = re.compile(r"\\bMUT-\\d{5,}\\b|\\badh[ée]rent\\s*:?\\s*\\d{6,}", re.I)

tests_positifs = ["Mutuelle n° MUT-558201", "N° adhérent : 99887766"]
tests_negatifs = ["HbA1c 43 mmol/mol", "MUT seul", "PA 138/86 mmHg"]
check(all(MUTUELLE_RE.search(t) for t in tests_positifs) and not any(MUTUELLE_RE.search(t) for t in tests_negatifs),
      "La regex fait le travail.", "La regex rate un positif ou attrape un négatif.")'''),
("code", '''from medagents.evaluation import load_truth, phi_leaks

truth = load_truth()
w2 = config.OUTPUT_DIR / "module1_P002"
m2 = IngestionAgent(use_llm=False, verbose=False).run(config.RAW_DIR / "dossier_P002.zip", w2)

print("Avant :", phi_leaks(AnonymizerAgent(use_llm=False, verbose=False).run(m2, w2).model_dump(), truth["P002"]))
register_pattern(PHICategory.OTHER_ID, MUTUELLE_RE)
print("Après :", phi_leaks(AnonymizerAgent(use_llm=False, verbose=False).run(m2, w2).model_dump(), truth["P002"]))'''),
("md", """### ✏️ Exercice 1.3 — Règles seules vs règles + LLM : mesurer (8 min)
Grâce à la vérité terrain (`data/ground_truth`), calculez le **rappel PHI** (part des identifiants de référence
absents des sorties) pour P001, P002 et P003, avec et sans LLM. Remplissez le tableau `resultats`."""),
("ex",
'''resultats = []
for pid in ["P001", "P002", "P003"]:
    wd = config.OUTPUT_DIR / f"module1_eval_{pid}"
    m = IngestionAgent(verbose=False).run(config.RAW_DIR / f"dossier_{pid}.zip", wd)
    for use_llm in (False, True):
        r = AnonymizerAgent(use_llm=use_llm, verbose=False).run(m, wd)
        fuites = phi_leaks(r.model_dump(), truth[pid])
        rappel = None  # TODO : 1 - (nombre de fuites / nombre d'identifiants de référence truth[pid]["phi_strings"])
        resultats.append({"patient": pid, "LLM": use_llm, "rappel": rappel, "fuites": fuites,
                          "rejets LLM": len(r.llm_rejected)})
pd.DataFrame(resultats)''',
'''resultats = []
for pid in ["P001", "P002", "P003"]:
    wd = config.OUTPUT_DIR / f"module1_eval_{pid}"
    m = IngestionAgent(verbose=False).run(config.RAW_DIR / f"dossier_{pid}.zip", wd)
    for use_llm in (False, True):
        r = AnonymizerAgent(use_llm=use_llm, verbose=False).run(m, wd)
        fuites = phi_leaks(r.model_dump(), truth[pid])
        rappel = round(1 - len(fuites) / len(truth[pid]["phi_strings"]), 3)
        resultats.append({"patient": pid, "LLM": use_llm, "rappel": rappel, "fuites": fuites,
                          "rejets LLM": len(r.llm_rejected)})
pd.DataFrame(resultats)'''),
("md", """**À discuter** : le rappel mesuré ici est-il suffisant pour publier un jeu de données ? Que mesure-t-on mal avec
cette méthode (précision, sur-masquage, identifiants indirects, quasi-identifiants combinés) ?"""),
("md", """### ✏️ Exercice 1.4 (bonus) — Attaque *zip-slip*
Fabriquez une archive dont une entrée s'appelle `../../evil.txt` et vérifiez que `safe_extract_zip` la refuse."""),
("ex",
'''import io
from medagents.tools.parsers import ArchiveError, safe_extract_zip

piege = config.OUTPUT_DIR / "piege.zip"
with zipfile.ZipFile(piege, "w") as z:
    z.writestr("rapport.txt", "contenu anodin")
    # TODO : ajoutez une entrée nommée "../../evil.txt"

try:
    safe_extract_zip(piege, config.OUTPUT_DIR / "piege_extrait")
    refus = False
except ArchiveError as exc:
    print("Refusé :", exc)
    refus = True
check(refus, "Attaque bloquée.", "L'archive a été extraite : avez-vous ajouté l'entrée piégée ?")''',
'''import io
from medagents.tools.parsers import ArchiveError, safe_extract_zip

piege = config.OUTPUT_DIR / "piege.zip"
with zipfile.ZipFile(piege, "w") as z:
    z.writestr("rapport.txt", "contenu anodin")
    z.writestr("../../evil.txt", "je veux sortir du répertoire")

try:
    safe_extract_zip(piege, config.OUTPUT_DIR / "piege_extrait")
    refus = False
except ArchiveError as exc:
    print("Refusé :", exc)
    refus = True
check(refus, "Attaque bloquée.", "L'archive a été extraite : avez-vous ajouté l'entrée piégée ?")'''),
("md", """## 🧭 Synthèse du module 1
- Le **LLM n'est qu'une couche** parmi d'autres : les règles et les identifiants connus font l'essentiel ; le LLM
  rattrape ce qui échappe aux règles, et ses propositions sont **vérifiées** avant usage.
- Le **contrôle résiduel** transforme une bonne pratique en **garde-fou bloquant**.
- Dates décalées + jetons stables = **pseudonymisation** : les données restent personnelles au sens du RGPD
  (AIPD, base légale, table de correspondance protégée). Safe Harbor HIPAA exigerait de ne garder que l'année.
- Faire tourner le LLM **en local** est ce qui autorise l'agent à lire le texte brut avant dé-identification.

➡️ Module 2 : transformer ces fichiers pseudonymisés en un **espace de features canonique**."""),
]
