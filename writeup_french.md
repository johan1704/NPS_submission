# Prédiction du NPS pour un Opérateur Télécom Pan-Africain
## Note Technique — Artefact Data Science Challenge

---

## 1. Contexte et Problème Business

Un opérateur télécom pan-africain réalise des enquêtes NPS régulières pour suivre la fidélité de ses clients. Le problème est structurel : seulement 15% des clients répondent. L'équipe retention ne peut pas agir efficacement quand elle ne connaît le niveau de satisfaction que d'un client sur sept.

L'objectif de ce projet est de construire un système de machine learning qui prédit la catégorie NPS — Détracteur, Passif ou Promoteur — pour les 85% de clients silencieux, en utilisant les données de compte et de comportement. Les prédictions alimentent un workflow de rétention qui priorise les Détracteurs pour une action proactive avant qu'ils churnent.

Ce n'est pas un exercice de prédiction pure. Le système doit être utilisable par un retention manager sur le terrain, suffisamment interprétable pour justifier des actions individuelles, et honnête sur ce qu'il peut et ne peut pas nous dire.

---

## 2. Construction du Label NPS

### Du Satisfaction Score au Label NPS

Le dataset IBM Telco fournit un Satisfaction Score de 1 à 5 — un signal réel fourni par des humains, pas un label fabriqué. Le passage aux catégories NPS n'est pas trivial.

Le mapping baseline recommandé par le challenge attribue les scores 1-3 aux Détracteurs, 4 aux Passifs et 5 aux Promoteurs. Nous avons utilisé cela comme point de départ, puis nous l'avons challengé.

**L'ambiguïté du score 3.** Le score 3 représente 2 665 clients — 38% du dataset. Traiter tous ces clients comme Détracteurs est une approximation grossière. Un client insatisfait depuis trois mois n'est pas le même qu'un client resté quatre ans malgré une satisfaction modérée. Nous avons utilisé le tenure comme signal de discrimination :

- Score 3 + tenure < 12 mois → **Détracteur** (nouveau et insatisfait — risque de churn élevé)
- Score 3 + tenure ≥ 12 mois → **Passif** (reste malgré l'insatisfaction — moins urgent)

Nous avons également utilisé le comportement de parrainage pour affiner le score 4 : un client qui a activement recommandé l'opérateur malgré un score neutre exhibe un comportement Promoteur et est reclassé en conséquence.

Enfin, nous avons introduit 10% de bruit contrôlé sur les clients score 3 pour refléter l'incertitude inhérente aux réponses aux sondages et éviter que le modèle soit trop confiant sur un label construit par des règles.

**Validation sans vérité terrain.** Comme nous n'avons pas de vraies notes NPS, nous avons validé le label indirectement en le croisant avec le Churn Label (non utilisé dans la construction). Le résultat confirme la cohérence :

| Segment | Taux de churn |
|---|---|
| Détracteur | 80.7% |
| Passif | 4.9% |
| Promoteur | 0.0% |

Le gradient est clair et dans la direction attendue.

### Data Leakage dans la Construction du Label

Le dataset contient des variables qui doivent être exclues à la fois du label et des features du modèle :

- **Churn Value** (0/1) : c'est l'outcome churn lui-même. L'utiliser dans le label reviendrait à prédire le churn déguisé en NPS — gonfler les métriques et rendre le système inutile opérationnellement.
- **Churn Score** et **CLTV** : calculés depuis l'outcome churn par IBM, non disponibles au moment de la prédiction.
- **Churn Reason / Category** : connus uniquement après qu'un client est déjà parti.

Exclure ces variables est non négociable. Un modèle qui intègre des informations post-churn ne peut pas être déployé sur des clients actifs.

---

## 3. Préparation des Données et Feature Engineering

### Dataset et Split

Le dataset IBM Telco (v11.1.3+) couvre 7 043 clients sur six fichiers joints par Customer ID. Après construction du label, le dataset analytique contient 40 features.

Le split train/test reflète le vrai problème business : le modèle sera appliqué aux clients qui n'ont pas répondu au sondage. Nous avons utilisé un split 15/85 — 15% comme ensemble d'entraînement (simulant les répondants au sondage) et 85% comme ensemble de test (simulant les clients silencieux). C'est plus conservateur qu'un split standard 80/20 et fournit une estimation réaliste des performances de généralisation.

Le déséquilibre des classes est significatif :

| Classe | Effectif (train) | Part |
|---|---|---|
| Détracteur | ~615 | 58% |
| Passif | ~268 | 25% |
| Promoteur | ~173 | 16% |

L'accuracy standard serait trompeuse ici. Un modèle naïf prédisant toujours "Détracteur" atteindrait 58% d'accuracy sans capturer aucun Passif ou Promoteur.

### Feature Engineering

Au-delà des 31 variables originales, nous avons construit 9 features avec une justification business explicite :

| Feature | Hypothèse business |
|---|---|
| `tenure_x_contract` | Tenure × type de contrat capture mieux la profondeur d'engagement que chaque variable seule |
| `charge_per_service` | Un coût élevé par service souscrit signale une mauvaise valeur perçue |
| `refund_rate` | Les remboursements indiquent des incidents ou litiges — proxy d'insatisfaction |
| `monthly_charge_ratio` | La charge mensuelle relative au tenure capture la pression financière récente |
| `has_security` | L'adoption de services de sécurité signale la confiance envers l'opérateur |
| `has_streaming` | L'usage streaming indique l'engagement entertainment |
| `digital_engagement` | Facturation électronique + paiement automatique signale l'autonomie digitale |
| `has_referred` | Le comportement de parrainage est le signal Promoteur le plus direct disponible |
| `nb_services` | La largeur du bundle est un proxy de l'engagement client |

Les features géographiques (Zip Code, Latitude, Longitude) ont été conservées car dans un contexte télécom, la géographie est un proxy légitime de la qualité réseau — pas uniquement un marqueur socioéconomique. Cette décision est revisitée dans la section fairness.

---


---

## 4. Verbatims Synthetiques (Section 4.4)

### Motivation

Dans un vrai operateur telecom, les notes du call center, les transcripts de chat et les avis sur les applications sont parmi les sources de signal NPS les plus riches — et les moins exploitees. Les donnees tabulaires structurees nous disent ce qu'un client fait. Le texte nous dit comment il le ressent.

Le dataset IBM Telco ne contient pas de donnees textuelles. Nous avons genere un verbatim synthetique par client — une note courte de 1 a 3 phrases simulant le dernier contact du client avec le support — en utilisant Llama3 (via Ollama, en local) conditionne sur un sous-ensemble du profil de chaque client.

### Approche de Generation

Chaque verbatim a ete genere avec le template de prompt suivant, stocke dans `notebook/verbatim_prompt.txt` pour la reproductibilite :

```
You are a customer of an African telecom operator.
Customer profile :
  - Satisfaction level : {nps_label}
  - Tenure : {tenure} months
  - Contract type : {contract}
  - Number of services : {nb_services}
  - Monthly charge : {monthly_charge} USD
Expected tone : {tone}
Write 1 to 3 short sentences about what this customer said
during their last contact with customer support.
Be realistic. Do not mention any NPS score.
Reply only with the verbatim, no introduction.
```

Le ton etait mappe au label NPS : frustre pour les Detracteurs, neutre pour les Passifs, enthousiaste pour les Promoteurs. Pour refleter le bruit du monde reel, 15% des clients ont recu un ton counter-intuitif — un Detracteur qui semble satisfait, un Promoteur qui se plaint d'un detail mineur.

**Perimetre de generation.** Generer 7 043 verbatims via un LLM local prend environ 14 heures sur CPU. Nous avons genere les verbatims sur un echantillon stratifie de 499 clients (~7% du dataset), en preservant la distribution des classes NPS. Le fichier genere (`notebook/telco_nps_verbatims.csv`) est commite dans le repository pour eviter de relancer la generation.

### Extraction du Signal

Nous avons extrait les scores de sentiment avec VADER (Valence Aware Dictionary and sEntiment Reasoner), un analyseur de sentiment base sur des regles, bien adapte aux courts retours clients en anglais. VADER retourne un score compound entre -1 (tres negatif) et +1 (tres positif).

Score sentiment moyen par segment NPS, conforme aux attentes :

| Segment | Score VADER moyen |
|---|---|
| Detracteur | Negatif |
| Passif | Proche de zero |
| Promoteur | Positif |

Le bruit de 15% introduit des cas counter-intuitifs qui empeche le modele de traiter le sentiment du verbatim comme un label deterministe.

### Ce que le Texte Apporte — Evaluation Honnete

Nous avons compare les performances de LightGBM avec et sans `sentiment_score` comme feature supplementaire, sur le sous-ensemble de 499 clients avec verbatims :

- L'amelioration du Recall Detracteur etait inferieure a 2 points de pourcentage.
- La complexite ajoutee du pipeline textuel — generation, stockage, extraction de sentiment — n'est pas justifiee par ce gain marginal sur des donnees synthetiques.

**Conclusion.** Sur ce dataset, les verbatims synthetiques n'ameliorent pas significativement la prediction au-dela de la baseline tabulaire. C'est attendu : les verbatims ont ete generes depuis les memes features tabulaires que le modele voit deja, ils portent donc un signal redondant.

En production avec de vrais transcripts du call center, la conclusion serait probablement differente. Le texte capture les plaintes, les intentions de resiliation et le ton emotionnel que les donnees structurees ne peuvent pas encoder. Ce pipeline est concu pour etre pret pour de vrais verbatims — les etapes de generation et d'extraction sont identiques, seule la source de donnees change.

## 5. Modélisation et Évaluation

### Sélection des Modèles

Le NPS est une cible ordinale : Détracteur < Passif < Promoteur. Un modèle qui traite ces classes comme indépendantes ignore cet ordre — prédire Promoteur quand la vérité est Détracteur est une erreur plus grave que prédire Passif. Nous avons testé quatre approches par ordre de complexité croissante :

**Dummy Classifier** — prédit la classe majoritaire. Établit le plancher de performance. Tout modèle sérieux doit le dépasser.

**Régression Logistique** — baseline linéaire avec `class_weight='balanced'`. Rapide, interprétable, utile comme point de référence avant d'introduire de la complexité non-linéaire.

**Régression Ordinale (mord)** — respecte la nature ordonnée de la cible. Mathématiquement plus appropriée pour le NPS. En pratique, elle délivre un meilleur QWK que LightGBM en pénalisant les erreurs ordinales proportionnellement, mais reste en deçà sur le Recall Détracteur — notre métrique business principale.

**LightGBM** — gradient boosting avec `class_weight='balanced'`. Retenu comme modèle final pour sa combinaison de performance, d'interprétabilité SHAP et de scalabilité.

### Métriques d'Évaluation

| Métrique | Justification |
|---|---|
| **Recall Détracteur** | Métrique business principale : parmi tous les vrais Détracteurs, combien capture-t-on ? Un Détracteur manqué est un client qui churne sans intervention. |
| **Macro F1** | Performance équilibrée sur les trois classes — garantit que les Passifs et Promoteurs ne sont pas ignorés. |
| **Quadratic Weighted Kappa** | Pénalise les erreurs proportionnellement à leur distance ordinale. Prédire Promoteur quand la vérité est Détracteur est plus pénalisé que prédire Passif. |
| **Calibration** | Garantit que les probabilités prédites sont fiables — critique pour le threshold tuning. |

L'accuracy est explicitement exclue comme métrique principale.

### Threshold Tuning

Par défaut, le modèle classifie un client comme Détracteur quand P(Détracteur) > 0.5. D'un point de vue business, un faux négatif (rater un Détracteur) est plus coûteux qu'un faux positif (contacter un Passif inutilement). Nous avons optimisé le seuil de classification pour maximiser le Recall Détracteur :

| Modèle | Recall Détracteur | Macro F1 | QWK |
|---|---|---|---|
| Dummy | 0.000 | 0.193 | 0.000 |
| Régression Logistique | 0.809 | 0.515 | 0.391 |
| Régression Ordinale | 0.586 | 0.542 | 0.462 |
| LightGBM | 0.702 | 0.579 | 0.428 |
| **LightGBM + Threshold (0.20)** | **0.885** | **0.525** | **0.393** |

Le modèle final capture 88.5% des vrais Détracteurs dans le test set. Le compromis : le Macro F1 diminue légèrement par rapport au LightGBM par défaut, car certains Passifs sont maintenant classifiés comme Détracteurs. C'est un compromis acceptable compte tenu de l'objectif business.

**Note sur TabICL.** Nous avons testé TabICL, un modèle de fondation pour données tabulaires (in-context learning, sans tuning d'hyperparamètres). Les résultats sont comparables à LightGBM sur cette taille de dataset. LightGBM a été retenu pour la production en raison du support natif SHAP et d'une latence d'inférence plus faible. TabICL reste un candidat solide si le dataset croît significativement.

---

## 6. Drivers de Détraction

### Drivers Globaux

Les features les plus importantes pour prédire la Détraction, par magnitude des valeurs SHAP :

1. `tenure_x_contract` — le signal dominant. Les clients en contrat mensuel avec un tenure court sont les plus à risque.
2. `monthly_charge_ratio` — pression financière récente élevée relative au tenure.
3. `Number of Referrals` — l'absence de références est un fort signal de détraction.
4. Features géographiques — suggèrent des disparités de qualité réseau par zone.

### Drivers par Segment

Les drivers diffèrent significativement selon les segments clients :

- **Nouveaux clients (tenure < 12 mois)** : la détraction est principalement pilotée par le type de contrat et le ratio de charge. Ces clients sont coûteux à acquérir et partent rapidement si l'expérience initiale déçoit.
- **Clients anciens (tenure ≥ 12 mois)** : la détraction est davantage pilotée par la composition du bundle de services et l'engagement digital. Ces clients ne partent pas impulsivement — quelque chose de structurel a érodé leur satisfaction.

### Actionnable vs Non-Actionnable

63% du signal SHAP provient de variables actionnables — des features que le business peut influencer :

| Feature | Action |
|---|---|
| Type de contrat | Proposer migration vers contrat annuel |
| Charge mensuelle | Proposer une remise personnalisée |
| Bundle de services | Proposer des services complémentaires |
| Engagement digital | Programme d'onboarding aux outils digitaux |

Les 37% restants — principalement le tenure et la géographie — ne peuvent pas être changés directement. Le signal géographique reflète probablement des disparités de qualité réseau, qui nécessitent un investissement infrastructure plutôt que des actions de rétention.

**Mise en garde importante.** Ces résultats sont des corrélations, pas des relations causales. Le fait que la charge mensuelle soit associée à la détraction ne garantit pas qu'une réduction de prix la résoudra. Des A/B tests seraient nécessaires pour valider les effets causaux avant de déployer toute intervention à grande échelle.

---

## 7. Fairness et Biais

Le modèle alloue le budget retention en priorisant les Détracteurs prédits. S'il rate systématiquement les Détracteurs d'un groupe démographique spécifique, ce groupe reçoit moins de support proactif — non pas parce qu'il est moins insatisfait, mais parce que le modèle le rate.

Nous avons audité le Recall Détracteur sur quatre segments démographiques :

| Groupe | Recall Détracteur | Statut |
|---|---|---|
| Genre (Homme vs Femme) | 87.4% vs 89.7% | OK — écart < 10 pts |
| Senior Citizen (Oui vs Non) | 88.3% vs 88.6% | OK — écart négligeable |
| Moins de 30 ans (Oui vs Non) | 88.2% vs 88.6% | OK — écart négligeable |
| **Marié (Oui vs Non)** | **79.5% vs 93.3%** | **ALERTE — écart 13.8 pts** |

Le modèle ne capture que 79.5% des Détracteurs parmi les clients mariés, contre 93.3% parmi les non-mariés. Cela signifie que les clients mariés insatisfaits sont systématiquement moins susceptibles de recevoir un appel de rétention.

**Ce résultat doit être escaladé aux équipes Customer Experience et Legal avant tout déploiement en production.**

Sur les features géographiques : le Zip Code et la ville peuvent servir de proxies pour le statut socioéconomique. Nous les avons conservés car dans un contexte télécom, ils portent un signal légitime de qualité réseau. Si une revue juridique détermine qu'ils introduisent un proxy de caractéristique protégée, ils devront être supprimés — à un coût mesuré sur les performances du modèle.

---

## 8. Limites et Prochaines Étapes

### Ce qui est implémenté

- Construction du label depuis le Satisfaction Score avec enrichissement et bruit contrôlé
- Pipeline de modélisation complet : baseline → LightGBM avec threshold tuning
- Interprétabilité SHAP au niveau individuel et par segment
- Audit fairness sur quatre groupes démographiques
- Interface Streamlit pour les retention managers
- Couche de monitoring légère avec détection de drift

### Ce qui est approximatif

- Le label est un proxy : nous utilisons le Satisfaction Score (1-5) comme substitut au NPS réel (0-10). Le mapping est raisonnable mais introduit du bruit de label.
- Les verbatims synthétiques : générés par Llama3 conditionné sur les features client. Utile pour démontrer le pipeline, mais pas un substitut aux vraies données du call center.
- Le script de monitoring tourne sur une simulation (train vs test) — pas sur du vrai trafic de production.
- L'écart de fairness sur les clients mariés : identifié mais non corrigé. La correction nécessite une investigation sur la cause racine avant tout rééchantillonnage.

### Ce qui reste comme travail futur

1. **Vrais verbatims** : intégrer de vrais transcripts du call center ou des avis applicatifs pour tester si le texte apporte un signal au-delà des features tabulaires.
2. **Enrichissement Census** : ajouter des features socioéconomiques au niveau code postal pour améliorer le signal géographique sans s'appuyer sur des proxies de localisation bruts.
3. **Holdout group** : implémenter un groupe contrôle de 10% parmi les Détracteurs prédits pour mesurer le vrai impact causal des actions de rétention et éviter la contamination des données d'entraînement par le feedback loop.
4. **Pipeline de retraining** : automatiser le monitoring et le déclencheur de retraining une fois le modèle déployé sur du vrai trafic de production.
5. **Investigation clients mariés** : comprendre pourquoi le modèle sous-performe sur les clients mariés et corriger avant la production.

---

*Ce travail a été réalisé dans le cadre du challenge Artefact. Des outils LLM (Claude, Llama3) ont été utilisés pour le scaffolding du code, la rédaction de documentation et la génération de verbatims. L'ensemble des décisions de modélisation, des choix analytiques et des conclusions sont de la responsabilité du candidat.*