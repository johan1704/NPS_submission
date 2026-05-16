# NPS Prediction for a Pan-African Telecom Operator
## Technical Write-Up — Artefact Data Science Challenge

---

## 1. Context and Business Problem

A pan-African telecom operator runs regular NPS surveys to track customer loyalty. The problem is structural : only 15% of customers respond. The retention team cannot act effectively when it only knows the satisfaction level of one in seven customers.

The goal of this project is to build a machine learning system that predicts the NPS category — Detractor, Passive, or Promoter — for the 85% of silent customers, using account and behavioral data. Predictions feed a retention workflow that prioritizes Detractors for proactive outreach before they churn.

This is not a pure prediction exercise. The system needs to be usable by a retention manager in the field, interpretable enough to justify individual actions, and honest about what it can and cannot tell us.

---

## 2. Building the NPS Target

### From Satisfaction Score to NPS Label

The IBM Telco dataset provides a Satisfaction Score from 1 to 5 — a real human-provided signal, not a fabricated label. The mapping to NPS categories is not trivial.

The baseline mapping recommended by the challenge assigns scores 1-3 to Detractors, 4 to Passives, and 5 to Promoters. We used this as a starting point, then challenged it.

**The ambiguity of score 3.** Score 3 represents 2,665 customers — 38% of the dataset. Treating all of them as Detractors is a coarse approximation. A customer who has been dissatisfied for three months is not the same as one who has stayed for four years despite moderate dissatisfaction. We used tenure as a discrimination signal :

- Score 3 + tenure < 12 months → **Detractor** (new and dissatisfied — high churn risk)
- Score 3 + tenure ≥ 12 months → **Passive** (stays despite dissatisfaction — less urgent)

We also used referral behavior to refine score 4 : a customer who has actively recommended the operator despite a neutral score exhibits Promoter behavior and is reclassified accordingly.

Finally, we introduced 10% controlled noise on score-3 customers to reflect the inherent uncertainty in human survey responses and prevent the model from being overconfident on a rule-based label.

**Validation without ground truth.** Since we have no true NPS scores, we validated the label indirectly by crossing it with the Churn Label (which was not used in construction). The result confirms coherence :

| Segment | Churn Rate |
|---|---|
| Detractor | 80.7% |
| Passive | 4.9% |
| Promoter | 0.0% |

The gradient is clear and in the expected direction.

### Data Leakage in Label Construction

The dataset contains variables that must be excluded from both the label and the model features :

- **Churn Value** (0/1) : this is the churn outcome itself. Using it in the label would mean predicting churn disguised as NPS — inflating metrics and making the system operationally useless.
- **Churn Score** and **CLTV** : computed from the churn outcome by IBM, not available at prediction time.
- **Churn Reason / Category** : only known after a customer has already left.

Excluding these variables is non-negotiable. A model that incorporates post-churn information cannot be deployed on active customers.

---

## 3. Data Preparation and Feature Engineering

### Dataset and Split

The IBM Telco dataset (v11.1.3+) covers 7,043 customers across six files joined on Customer ID. After label construction, the analytical dataset contains 40 features.

The train/test split reflects the real business problem : the model will be applied to customers who did not answer the survey. We used a 15/85 split — 15% as the training set (simulating survey respondents) and 85% as the test set (simulating silent customers). This is more conservative than a standard 80/20 split and provides a realistic estimate of generalization performance.

Class imbalance is significant :

| Class | Count | Share |
|---|---|---|
| Detractor | ~615 (train) | 58% |
| Passive | ~268 (train) | 25% |
| Promoter | ~173 (train) | 16% |

Standard accuracy would be misleading here. A naive model predicting always "Detractor" would achieve 58% accuracy while capturing zero Passives or Promoters.

### Feature Engineering

Beyond the 31 original variables, we built 9 features with explicit business rationale :

| Feature | Business hypothesis |
|---|---|
| `tenure_x_contract` | Tenure × contract type captures engagement depth better than either variable alone |
| `charge_per_service` | High cost per service subscribed signals poor perceived value |
| `refund_rate` | Refunds indicate incidents or disputes — a proxy for dissatisfaction |
| `monthly_charge_ratio` | Monthly charge relative to tenure captures recent financial pressure |
| `has_security` | Security service adoption signals trust in the operator |
| `has_streaming` | Streaming usage indicates entertainment engagement |
| `digital_engagement` | Paperless billing + auto-pay signals digital autonomy |
| `has_referred` | Referral behavior is the most direct Promoter signal available |
| `nb_services` | Service breadth is a proxy for customer engagement |

Geographic features (Zip Code, Latitude, Longitude) were retained because in a telecom context, geography is a legitimate proxy for network quality — not solely a socioeconomic marker. This decision is revisited in the fairness section.

---

## 4. Modelling and Evaluation

### Model Selection

NPS is an ordinal target : Detractor < Passive < Promoter. A model that treats these as three independent classes ignores this ordering — predicting Promoter when the truth is Detractor is a worse error than predicting Passive. We tested four approaches in increasing order of complexity :

**Dummy Classifier** — predicts the majority class. Establishes the performance floor. Any serious model must exceed this.

**Logistic Regression** — linear baseline with `class_weight='balanced'`. Fast, interpretable, and useful as a reference point before introducing non-linear complexity.

**Ordinal Regression (mord)** — respects the ordered nature of the target. Mathematically more appropriate for NPS. In practice, it delivers better QWK than LightGBM by penalizing ordinal errors proportionally, but falls short on Detractor Recall — our primary business metric.

**LightGBM** — gradient boosting with `class_weight='balanced'`. Selected as the final model for its combination of performance, SHAP interpretability, and scalability.

### Evaluation Metrics

| Metric | Rationale |
|---|---|
| **Recall Detractor** | Primary business metric : among all true Detractors, how many do we capture ? A missed Detractor is a customer who churns without intervention. |
| **Macro F1** | Balanced performance across all three classes — ensures Passives and Promoters are not ignored. |
| **Quadratic Weighted Kappa** | Penalizes errors proportionally to their ordinal distance. Predicting Promoter when truth is Detractor is penalized more than predicting Passive. |
| **Calibration** | Ensures predicted probabilities are reliable — critical for threshold tuning. |

Accuracy is explicitly excluded as a primary metric.

### Threshold Tuning

By default, the model classifies a customer as Detractor when P(Detractor) > 0.5. From a business perspective, a false negative (missing a Detractor) is more costly than a false positive (contacting a Passive unnecessarily). We optimized the classification threshold to maximize Detractor Recall :

| Model | Recall Detractor | Macro F1 | QWK |
|---|---|---|---|
| Dummy | 0.000 | 0.193 | 0.000 |
| Logistic Regression | 0.809 | 0.515 | 0.391 |
| Ordinal Regression | 0.586 | 0.542 | 0.462 |
| LightGBM | 0.702 | 0.579 | 0.428 |
| **LightGBM + Threshold (0.20)** | **0.885** | **0.525** | **0.393** |

The final model captures 88.5% of true Detractors in the test set. The trade-off : Macro F1 decreases slightly compared to default LightGBM, as some Passives are now classified as Detractors. This is an acceptable trade-off given the business objective.

**Note on TabICL.** We tested TabICL, a tabular foundation model (in-context learning, no hyperparameter tuning). Results were comparable to LightGBM on this dataset size. LightGBM was retained for production due to native SHAP support and lower inference latency. TabICL remains a strong candidate if the dataset grows significantly.

---

## 5. Drivers of Detraction

### Global Drivers

The most important features for predicting Detraction, by SHAP value magnitude :

1. `tenure_x_contract` — the dominant signal. Customers on month-to-month contracts with short tenure are at highest risk.
2. `monthly_charge_ratio` — high recent financial pressure relative to tenure.
3. `Number of Referrals` — absence of referrals is a strong detraction signal.
4. Geographic features — suggest network quality disparities by zone.

### Segment-Level Drivers

Drivers differ meaningfully across customer segments :

- **New customers (tenure < 12 months)** : detraction is driven primarily by contract type and charge ratio. These customers are expensive to acquire and quick to leave if the early experience disappoints.
- **Long-tenured customers (tenure ≥ 12 months)** : detraction is driven more by service bundle composition and digital engagement. These customers are not leaving impulsively — something structural has eroded their satisfaction.

### Actionable vs Non-Actionable

63% of the SHAP signal comes from actionable variables — features the business can influence :

| Feature | Action |
|---|---|
| Contract type | Offer migration to annual contract |
| Monthly charge | Propose a personalized discount |
| Service bundle | Offer complementary services |
| Digital engagement | Onboarding program for digital tools |

The remaining 37% — primarily tenure and geography — cannot be changed directly. Geographic signal likely reflects network quality disparities, which require infrastructure investment rather than retention actions.

**Important caveat.** These are correlations, not causal relationships. The fact that monthly charge is associated with detraction does not guarantee that a price reduction will resolve it. A/B testing would be required to validate causal effects before scaling any intervention.

---

## 6. Fairness and Bias

The model allocates retention budget by prioritizing predicted Detractors. If it systematically misses Detractors from a specific demographic group, that group receives less proactive support — not because they are less dissatisfied, but because the model fails them.

We audited Detractor Recall across four demographic segments :

| Group | Recall Detractor | Status |
|---|---|---|
| Gender (Male vs Female) | 87.4% vs 89.7% | OK — gap < 10 pts |
| Senior Citizen (Yes vs No) | 88.3% vs 88.6% | OK — negligible gap |
| Under 30 (Yes vs No) | 88.2% vs 88.6% | OK — negligible gap |
| **Married (Yes vs No)** | **79.5% vs 93.3%** | **ALERT — gap 13.8 pts** |

The model captures only 79.5% of Detractors among married customers, compared to 93.3% among unmarried customers. This means married customers who are dissatisfied are systematically less likely to receive a retention call.

**This finding should be escalated to the Customer Experience and Legal teams before any production deployment.**

On geographic features : Zip Code and city can act as proxies for socioeconomic status. We retained them because in a telecom context they carry legitimate network quality signal. If legal review determines they introduce protected-class proxying, they should be dropped — at a measured cost to model performance.

---

## 7. Limitations and Next Steps

### What is implemented

- Label construction from Satisfaction Score with enrichment and controlled noise
- Full modelling pipeline : baseline → LightGBM with threshold tuning
- SHAP-based interpretability at individual and segment level
- Fairness audit across demographic groups
- Streamlit interface for retention managers
- Lightweight monitoring layer with drift detection

### What is approximate

- The label is a proxy : we use Satisfaction Score (1-5) as a substitute for true NPS (0-10). The mapping is reasonable but introduces label noise.
- Synthetic verbatims : generated by Llama3 conditioned on customer features. Useful for demonstrating the pipeline, but not a substitute for real call center data.
- The monitoring script runs on a simulation (train vs test) — not on real production traffic.
- Married customer fairness gap : identified but not corrected. Correction requires investigation into the root cause before any reweighting.

### What is left as future work

1. **Real verbatims** : integrate actual call center transcripts or app reviews to test whether text adds signal beyond tabular features.
2. **Census enrichment** : add ZIP-code-level socioeconomic features to improve geographic signal without relying on raw location proxies.
3. **Holdout group** : implement a 10% holdout among predicted Detractors to measure the true causal impact of retention actions and prevent feedback loop contamination of training data.
4. **Retraining pipeline** : automate monitoring and retraining trigger once the model is deployed on real production traffic.
5. **Married customer investigation** : understand why the model underperforms on married customers and correct before production.

---

*This work was completed as part of the Artefact take-home challenge. LLM tools (Claude, Llama3) were used for code scaffolding, documentation drafting, and verbatim generation. All modelling decisions, analytical choices, and conclusions are the responsibility of the candidate.*