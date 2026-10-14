from .common import SETUP

TITLE = "00_installation_ollama"
CELLS = [
("md", """# 00 · Installation et prise en main du LLM local
**Atelier ICSMAI 2027 — Concevoir un pipeline multi-agents médical de bout en bout**

⏱️ *À faire avant l'atelier (15 min) ou pendant l'accueil.*

Objectifs :
1. disposer d'un **LLM local** (Ollama) opérationnel — aucune donnée patient ne quitte la machine ;
2. comprendre les **sorties structurées** (JSON contraint par un schéma Pydantic), brique de base de tous nos agents ;
3. générer les **dossiers patients synthétiques** et entraîner les **modèles du registre**.

### Trois façons d'exécuter Ollama
| Option | Pour qui | Mise en place |
|---|---|---|
| **A. Portable** | 16 Go de RAM (GPU conseillé) | installer depuis https://ollama.com puis `ollama pull llama3.1:8b` (≈ 4,9 Go) — ou `llama3.2:3b` (≈ 2 Go) sur une machine modeste |
| **B. Google Colab** | pas d'installation locale | *Exécution › Modifier le type d'exécution › GPU T4*, puis lancer la cellule 0 (installe Ollama automatiquement) |
| **C. Serveur de la salle** | réseau de l'atelier | décommenter `OLLAMA_HOST` dans la cellule 0 avec l'adresse donnée par le formateur |
"""),
("code", SETUP),
("md", """## 1. Vérifier la connexion au LLM
`health_check()` interroge le serveur Ollama et vérifie que le modèle est téléchargé."""),
("code", '''from medagents import config
from medagents.llm import get_llm, pretty

print("Serveur :", config.OLLAMA_HOST, "| Modèle :", config.LLM_MODEL)
llm = get_llm()
print(pretty(llm.health_check()))'''),
("md", """## 2. Les sorties structurées : la brique de base d'un agent fiable
Un agent n'échange pas du texte libre avec les autres agents : il produit un **objet validé**.
On décrit l'objet attendu avec **Pydantic v2**, on transmet son schéma JSON à Ollama (`format=`), puis on
**revalide** la réponse. Si elle est invalide, l'erreur est renvoyée au LLM (*boucle d'auto-correction*).

L'exemple ci-dessous classe une vignette clinique fictive en niveau d'urgence."""),
("code", '''from typing import Literal
from pydantic import BaseModel, Field

class Triage(BaseModel):
    urgence: Literal["immédiate", "rapide", "différée"]
    specialite: str = Field(description="Spécialité d'orientation")
    justification: str = Field(description="Une phrase")

vignette = ("Homme de 68 ans, douleur thoracique constrictive depuis 40 minutes, irradiant au bras gauche, "
            "sueurs, antécédent de diabète.")
res = llm.structured("Tu es un médecin régulateur. Réponds en JSON.", vignette, Triage, agent="démo")
print(type(res).__name__, "→", pretty(res))'''),
("code", '''# Ce que le LLM a réellement renvoyé (brut) et le temps de réponse
last = llm.last()[0]
print(f"{last['seconds']:.1f} s — essai n°{last['attempt']}")
print(last["response"])'''),
("md", """> 💡 **À retenir.** Le type `Literal[...]` *ferme* l'espace des réponses possibles : le LLM ne peut pas inventer
> un niveau d'urgence. Nous utiliserons ce principe pour contraindre le choix de codes terminologiques (Agent 3)
> et de modèles (Agent 4)."""),
("md", "## 3. Générer les données synthétiques et entraîner le registre de modèles"),
("code", '''import time, zipfile
from medagents.datagen import generate
from medagents.model_registry.build import build

t0 = time.time()
dossiers = generate()
build()
print(f"Terminé en {time.time() - t0:.0f} s")
for d in dossiers:
    with zipfile.ZipFile(d) as z:
        print(f"\\n📦 {d.name}")
        for n in z.namelist():
            print("   ", n)'''),
("md", """Les trois dossiers sont **fictifs** mais volontairement réalistes :

| Dossier | Profil | Difficultés |
|---|---|---|
| `P001` | homme, 75 ans, insuffisance cardiaque | 5 formats (FHIR, HL7 v2, CSV « à la française », compte rendu, DICOM dans un ZIP imbriqué) |
| `P002` | femme, 51 ans, bilan glycémique | unités à convertir (mmol/L, mmol/mol, g/L), Excel, IMC à dériver |
| `P003` | homme, 93 ans, insuffisance cardiaque | donnée requise manquante, valeur aberrante, fichier corrompu, compte rendu en anglais |

✅ **Si toutes les cellules se sont exécutées, vous êtes prêt·e pour l'atelier.**"""),
]
