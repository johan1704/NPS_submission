# Section 4.9 - Monitoring & Retraining

## Ce qui est implémenté

### Module 1 - Input Drift (monitor.py)
Détecte si les features en entrée ont changé vs le train set.
- Méthode : TabularDrift (Alibi-Detect) — KS-test + Chi2
- Référence : X_train (répondants sondage)
- Production simulée : X_test (clients silencieux)
- Seuil alerte : >= 5 features avec p-value < 0.05

### Module 2 - Prediction Drift (monitor.py)
Détecte si la distribution des prédictions a changé.
- Méthode : PSI (Population Stability Index)
- PSI < 0.10 → Stable | 0.10-0.20 → Surveiller | > 0.20 → Alerte

### Module 3 - Retraining Trigger (monitor.py)
Déclenche une alerte retraining si Signal 1 ou Signal 2 est activé.

---

## Ce qui est documenté (non simulable sans données production)

### Performance sur nouveaux labels

En production, à chaque nouvelle vague de sondage :
```
1. Récupérer les nouvelles réponses NPS
2. Comparer y_pred (du modèle) vs y_true (nouvelle réponse)
3. Calculer Recall Détracteur sur ce batch
4. Si Recall Det < 0.70 → Signal 3 déclenché → retraining
```

### Feedback Loop — Le problème principal

**Schéma du problème :**
```
Modèle prédit Détracteur
    ↓
Équipe retention contacte le client
    ↓
Client devient satisfait
    ↓
Prochain sondage → label PROMOTEUR
    ↓
Modèle réentraîné sur ces données
    ↓
Modèle apprend : "ce profil = Promoteur"
    ↓
Modèle arrête de détecter ce profil comme Détracteur
    ↓
Dégradation progressive sans qu'on s'en aperçoive
```

**La solution : Holdout Group**
```
Sur 100 Détracteurs prédits :
  90 → contactés (groupe traitement)
  10 → NON contactés (groupe contrôle / holdout)

Les 10 non contactés :
  → Labels non contaminés par les actions retention
  → Permettent de mesurer le vrai Recall Détracteur
  → Servent de données saines pour le retraining
```

**Implémentation recommandée :**
```python
# Dans l'app de déploiement
HOLDOUT_RATE = 0.10

detractors_predicted = clients[clients['p_detracteur'] >= threshold]
n_holdout = int(len(detractors_predicted) * HOLDOUT_RATE)

holdout_group    = detractors_predicted.sample(n_holdout, random_state=42)
treatment_group  = detractors_predicted.drop(holdout_group.index)

# Envoyer treatment_group à l'équipe retention
# Garder holdout_group pour évaluation future
```

### Stratégie de retraining

**Sur quelles données réentraîner :**
```
Option A — Sliding window (recommandée) :
  Train = 12 derniers mois de réponses sondage validées
  Avantage : le modèle s'adapte aux nouvelles tendances
  Risque : perd la mémoire des patterns anciens

Option B — Cumulative :
  Train = tout l'historique disponible
  Avantage : stable
  Risque : les vieilles données pèsent trop
```

**Fréquence recommandée :**
```
- Minimum : 1 fois par trimestre (schedule fixe)
- Conditionnel : si Signal 1 ou Signal 2 déclenché
- Après chaque vague sondage si >= 500 nouveaux labels
```

**Critère de déploiement du nouveau modèle :**
```
Recall Détracteur sur holdout fixe >= 0.80
ET pas de régression > 5% sur Macro F1
```

---

## Limites de ce monitoring

1. La simulation (train vs test) n'est pas identique à un vrai monitoring
   production où les données arrivent en flux continu.

2. Le PSI calculé sur deux moitiés du test set sous-estime le drift réel
   car les deux populations viennent du même dataset.

3. Sans holdout group implémenté dès le départ, il est impossible
   de mesurer le vrai impact du feedback loop a posteriori.

## Outils recommandés pour aller plus loin

| Besoin | Outil |
|---|---|
| Monitoring avancé | Evidently AI |
| Drift temps réel | WhyLabs / whylogs |
| Experiment tracking | MLflow |
| Pipeline retraining | Airflow / Prefect |