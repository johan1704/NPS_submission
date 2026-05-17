# NPS Prediction Challenge — Artefact

Systeme de prediction du NPS (Net Promoter Score) pour un operateur
telecom pan-africain. Le modele predit la categorie NPS
(Detracteur / Passif / Promoteur) des 85% de clients silencieux
pour prioriser les actions de retention.

---

## Structure du projet

```
nps/
├── app/
│   └── app.py                        ← interface Streamlit (section 4.8)
│
├── artifacts/
│   ├── raw/                          ← fichiers Excel IBM Telco v11.1.3+
│   │   ├── Telco_customer_churn_demographics.xlsx
│   │   ├── Telco_customer_churn_location.xlsx
│   │   ├── Telco_customer_churn_services.xlsx
│   │   ├── Telco_customer_churn_status.xlsx
│   │   └── Telco_customer_churn.xlsx
│   └── processed/                    ← (vide - fichiers generes dans notebook/)
│
├── monitoring/
│   ├── monitor.py                    ← detection de drift (section 4.9)
│   └── readme.md                     ← feedback loop documente
│
├── notebook/
│   ├── nps_4_1_to_4_7full.ipynb     ← pipeline complet sections 4.1 a 4.7
│   ├── nps_model_final.pkl           ← modele LightGBM entraine
│   ├── split_data.pkl                ← train/test split + encoders
│   ├── telco_nps_master.csv          ← dataset NPS derive (produit par notebook)
│   ├── telco_nps_verbatims.csv       ← verbatims synthetiques (section 4.4)
│   ├── verbatim_prompt.txt           ← prompt Llama3 utilise (reproductibilite)
│   ├── 04_5_calibration.png
│   ├── 04_5_lift_curve.png
│   ├── 04_5_model_comparison.png
│   ├── 04_6_shap_global.png
│   ├── 04_6_shap_segments.png
│   └── 04_7_fairness_audit.png
│
├── screenshot_interface/             ← screenshots de l'interface Streamlit
│   ├── screen1.png
│   ├── screen2.png
│   └── screen3.png
│
├── .gitignore
├── requirements.txt
├── README.md                         ← ce fichier
├── writeup_english.md                ← note technique (EN)
└── writeup_french.md                 ← note technique (FR)
```

---

## Documents

| Fichier | Contenu |
|---|---|
| `writeup_english.md` | Note technique complète : approche, label, modélisation, fairness, limites |
| `writeup_french.md` | Même contenu en français |
| `monitoring/readme.md` | Feedback loop et stratégie de retraining documentés |

---

## Installation

**Prerequis :** Python >= 3.10
**Testé avec :** Python 3.10.12

```bash
# Creer un environnement virtuel
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# Installer les dependances
pip install -r requirements.txt
```

**Pour regenerer les verbatims (optionnel — non necessaire) :**
```bash
# Installer Ollama : https://ollama.ai
ollama pull llama3
ollama serve
# Puis decommenter la section 4.4 dans le notebook
```

---

## Reproduire le pipeline complet

**Etape 1 — Verifier les donnees brutes**

Les fichiers Excel IBM Telco v11.1.3+ sont fournis dans `artifacts/raw/`.
Si absents, les telecharger depuis :
- Kaggle : https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- IBM Cognos : chercher "Telco customer churn (11.1.3+)"

**Etape 2 — Executer le notebook**

```bash
cd notebook/
jupyter notebook nps_4_1_to_4_7full.ipynb
# Kernel > Restart & Run All
```

> **Note importante — Section 4.4 (Verbatims) :**
> La generation des verbatims via Llama3 est **commentee** dans le notebook.
> Raison : Llama3 n'est pas deterministe — deux executions donnent des textes differents,
> ce qui rendrait les resultats de la comparaison avec/sans sentiment non reproductibles.
>
> Le fichier `notebook/telco_nps_verbatims.csv` est fourni dans le repo.
> **Restart & Run All fonctionne sans Ollama** — la section 4.4 est skippee.
>
> Pour regenerer les verbatims : decommenter la section 4.4,
> s'assurer qu'Ollama tourne (`ollama serve`), et relancer uniquement cette section.

Le notebook produit dans `notebook/` :
- `telco_nps_master.csv` — dataset NPS derive
- `nps_model_final.pkl` — modele entraine
- `split_data.pkl` — train/test split
- `*.png` — figures de validation et evaluation

**Etape 3 — Lancer l'interface Streamlit**

```bash
cd app/
streamlit run app.py
# Ouvre http://localhost:8501
```

**Etape 4 — Lancer le monitoring (optionnel)**

```bash
cd monitoring/
python monitor.py
```

---

## Reproductibilite

Tous les seeds sont fixes :

```python
RANDOM_SEED = 42   # train/test split, LightGBM, sampling
NOISE_RATE  = 0.10 # bruit dans la construction du label NPS
```

Fichiers commites pour garantir la reproductibilite :

| Fichier | Raison |
|---|---|
| `notebook/telco_nps_verbatims.csv` | Llama3 non deterministe — on commite le resultat |
| `notebook/nps_model_final.pkl` | Modele pre-entraine avec numpy 1.26.4 |
| `notebook/split_data.pkl` | Split pre-calcule (RANDOM_SEED=42) |
| `notebook/telco_nps_master.csv` | Dataset NPS derive |

---

## Chemins utilises dans les scripts

Les chemins sont relatifs au dossier de lancement :

| Script | Lancer depuis | Chemins attendus |
|---|---|---|
| `app.py` | `app/` | `../notebook/` pour pkl et csv, `../artifacts/raw/` pour demographics |
| `monitor.py` | `monitoring/` | `../artifacts/raw/` pour xlsx, `../notebook/` pour pkl |
| `notebook` | `notebook/` | `../artifacts/raw/` pour xlsx |

---

## Utilisation de l'IA

Conformement aux instructions du challenge (section 7) :

- **Claude (Anthropic)** : scaffolding du code, structuration du notebook, documentation
- **Llama3 via Ollama** : generation des verbatims synthetiques (section 4.4)

Les decisions de modelisation, choix analytiques et conclusions
sont de la responsabilite du candidat.

---

## Resultats cles

| Metrique | Valeur |
|---|---|
| Recall Detracteur (modele final) | 88.5% |
| Macro F1 | 52.5% |
| Quadratic Weighted Kappa | 39.3% |
| Alerte Fairness | Clients maries : ecart 13.8 pts |

---

## Limitations principales

1. Label construit depuis un proxy (Satisfaction Score 1-5), pas un vrai NPS 0-10
2. Train set de 1 056 clients seulement (15% du dataset)
3. Verbatims synthetiques — pas de vrais appels support
4. Monitoring simule sur train vs test — pas de vraies donnees production
5. Ecart fairness clients maries non corrige — a investiguer avant production