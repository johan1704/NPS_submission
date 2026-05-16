# NPS Prediction for a Pan-African Telecom Operator
## Technical Write-Up — Artefact Data Science Challenge

---

## 1. Context and Business Problem

A pan-African telecom operator runs regular NPS surveys to track customer loyalty. The problem is structural : only 15% of customers respond. The retention team cannot act effectively when it only knows the satisfaction level of one in seven customers.

This project addresses three simultaneous challenges that shape every technical decision downstream :

1. **A noisy label built from a proxy.** The dataset provides a Satisfaction Score (1-5), not a true NPS (0-10). The mapping is a reasonable approximation — not a ground truth. Any performance metric must be interpreted with this in mind.

2. **An ordinal, imbalanced target.** Detractor < Passive < Promoter is an ordered relationship, not three independent classes. Predicting Promoter when the truth is Detractor is a qualitatively worse error than predicting Passive. Standard classification ignores this. Class imbalance (58% Detractors in train) makes accuracy a misleading metric.

3. **A generalization gap from 15% to 85%.** The model learns from survey respondents and is applied to customers who never answered. These two populations are not necessarily identical — respondents may skew toward the very satisfied and the very dissatisfied. This is the hardest constraint to address and the most important one to acknowledge.

Every modelling choice in this submission is a response to one or more of these three constraints.

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

**Sensitivity analysis.** We trained a simple logistic regression on both the baseline mapping and the enriched mapping and compared Detractor Recall. The enriched mapping improved Recall from 49% to 76% on this baseline model — confirming that the label refinement is meaningful, not cosmetic.

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

The train/test split reflects the real business problem : the model will be applied to customers who did not answer the survey. We used a 15/85 split — 15% as the training set (simulating survey respondents) and 85% as the test set (simulating silent customers). A standard 80/20 split would overestimate generalization performance by training and testing on populations drawn from the same distribution.

Class imbalance is significant :

| Class | Count (train) | Share |
|---|---|---|
| Detractor | ~615 | 58% |
| Passive | ~268 | 25% |
| Promoter | ~173 | 16% |

A naive model predicting always "Detractor" would achieve 58% accuracy while capturing zero Passives or Promoters. This is why accuracy is excluded as a primary metric throughout this submission.

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


---

## 4. Synthetic Customer Verbatims (Section 4.4)

### Motivation

In a real telecom operator, call center notes, chat transcripts, and app reviews are among the richest sources of NPS signal — and among the least exploited. Structured tabular data tells us what a customer does. Text tells us how they feel about it.

The IBM Telco dataset contains no text data. We generated one synthetic verbatim per customer — a short 1-3 sentence note simulating the customer's last interaction with support — using Llama3 (via Ollama, running locally) conditioned on a subset of each customer's profile.

### Generation Approach

Each verbatim was generated using the following prompt template, stored in `notebook/verbatim_prompt.txt` for reproducibility :

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

Tone was mapped to NPS label : frustrated for Detractors, neutral for Passives, enthusiastic for Promoters. To reflect real-world noise, 15% of customers received a counter-intuitive tone — a Detractor who sounds satisfied, a Promoter who complains about a minor issue.

**Generation scope.** Generating 7,043 verbatims via a local LLM takes approximately 14 hours on CPU. We generated verbatims for a stratified sample of 499 customers (~7% of the dataset), preserving the NPS class distribution. The generated file (`notebook/telco_nps_verbatims.csv`) is committed to the repository to avoid re-running the generation.

### Signal Extraction

We extracted sentiment scores using VADER (Valence Aware Dictionary and sEntiment Reasoner), a rule-based sentiment analyzer well-suited for short customer feedback in English. VADER returns a compound score between -1 (very negative) and +1 (very positive).

Average sentiment by NPS segment, as expected :

| Segment | Mean VADER compound |
|---|---|
| Detractor | Negative |
| Passive | Near-neutral |
| Promoter | Positive |

The 15% noise introduces counter-intuitive cases that prevent the model from treating verbatim sentiment as a deterministic label.

### What the Text Adds — Honest Assessment

We compared LightGBM performance with and without `sentiment_score` as an additional feature, on the 499-client subsample with verbatims :

- The improvement in Detractor Recall was below 2 percentage points.
- The added complexity of the text pipeline — generation, storage, sentiment extraction — is not justified by this marginal gain on synthetic data.

**Conclusion.** On this dataset, synthetic verbatims do not meaningfully improve prediction beyond the tabular baseline. This is expected : the verbatims were generated from the same tabular features the model already sees, so they carry redundant signal.

In production with real call center transcripts, the conclusion would likely differ. Text data captures complaints, intent to cancel, and emotional tone that structured data cannot encode. This pipeline is designed to be ready for real verbatims — the generation and extraction steps are identical, only the data source changes.

## 5. Modelling and Evaluation

### Problem Formulation : Why Multiclass and Not Ordinal Regression ?

NPS is fundamentally an ordinal target. Detractor < Passive < Promoter is not just a categorical distinction — there is a meaningful ordering that standard multiclass classification ignores.

We tested ordinal logistic regression (mord / LogisticAT) specifically because it respects this ordering. It delivered a better Quadratic Weighted Kappa than LightGBM (0.462 vs 0.428), confirming that the ordinal structure carries real information.

However, we retained LightGBM multiclass as the final model for three reasons :

1. **Detractor Recall** — our primary business metric — is 58.6% for ordinal regression vs 88.5% for LightGBM with threshold tuning. The business cost of missing a Detractor outweighs the mathematical elegance of ordinal formulation.
2. **SHAP interpretability** — LightGBM integrates natively with SHAP, enabling individual-level explanations for the retention team. Ordinal regression does not offer equivalent interpretability tooling.
3. **Threshold flexibility** — LightGBM outputs calibrated class probabilities that can be tuned independently. Ordinal regression does not offer the same control over the Detractor decision boundary.

This is a deliberate trade-off, not an oversight. The ordinal nature of NPS is addressed in the evaluation metrics (QWK) and acknowledged as a limitation.

### Model Comparison

We tested four approaches in increasing order of complexity :

**Dummy Classifier** — predicts the majority class. Establishes the absolute performance floor.

**Logistic Regression** — linear baseline with `class_weight='balanced'`. Fast and interpretable. Required before introducing non-linear complexity.

**Ordinal Regression (mord)** — respects the ordered nature of the target. Best QWK among all models. Retained as a reference, not as the production choice.

**LightGBM** — gradient boosting with `class_weight='balanced'`. Selected as the final model.

### Evaluation Metrics

| Metric | Why it matters here |
|---|---|
| **Detractor Recall** | Primary business metric. A missed Detractor is a customer who churns without intervention. |
| **Macro F1** | Ensures Passives and Promoters are not ignored in a class-imbalanced setting. |
| **Quadratic Weighted Kappa** | Penalizes ordinal errors proportionally. Predicting Promoter when truth is Detractor costs more than predicting Passive. |
| **Calibration** | Reliable probabilities are essential for threshold tuning to hold in production. |

### Results

| Model | Detractor Recall | Macro F1 | QWK |
|---|---|---|---|
| Dummy | 0.000 | 0.193 | 0.000 |
| Logistic Regression | 0.809 | 0.515 | 0.391 |
| Ordinal Regression | 0.586 | 0.542 | 0.462 |
| LightGBM | 0.702 | 0.579 | 0.428 |
| **LightGBM + Threshold (0.20)** | **0.885** | **0.525** | **0.393** |

Lowering the classification threshold from 0.5 to 0.20 increases Detractor Recall from 70.2% to 88.5% at the cost of a moderate drop in Macro F1. This is the right trade-off : contacting a Passive unnecessarily costs a phone call. Missing a Detractor costs a customer.

**Note on TabICL.** We tested TabICL, a tabular foundation model using in-context learning with no hyperparameter tuning. Results were comparable to LightGBM on this dataset size. LightGBM was retained for production due to native SHAP support and lower inference latency. TabICL remains a strong candidate if the dataset grows significantly or if rapid prototyping is needed without tuning.

---

## 6. Drivers of Detraction

### Global Drivers

The most important features for predicting Detraction, by mean absolute SHAP value :

1. `tenure_x_contract` — dominant signal. Short-tenure customers on month-to-month contracts are at highest risk.
2. `monthly_charge_ratio` — high recent financial pressure relative to tenure.
3. `Number of Referrals` — absence of referrals is a strong detraction signal.
4. Geographic features — suggest network quality disparities by zone.

### Segment-Level Drivers

Drivers differ meaningfully across customer profiles :

- **New customers (tenure < 12 months)** : detraction is driven primarily by contract type and charge ratio. These customers are expensive to acquire and quick to leave if the early experience disappoints.
- **Long-tenured customers (tenure ≥ 12 months)** : detraction is driven more by service bundle composition and digital engagement. Something structural has eroded their satisfaction over time.
- **Month-to-month contracts** : charge per service is the dominant lever — these customers are price-sensitive and have no switching friction.
- **Long-term contracts** : referral behavior and service depth matter more — these customers have invested in the relationship and expect reciprocal value.

### Actionable vs Non-Actionable

63% of the SHAP signal comes from variables the business can act on :

| Feature | Action |
|---|---|
| Contract type | Offer migration to annual contract |
| Monthly charge | Propose a personalized discount |
| Service bundle | Offer complementary services |
| Digital engagement | Onboarding program for digital tools |

The remaining 37% — primarily tenure and geography — cannot be changed directly. Geographic signal likely reflects network quality disparities requiring infrastructure investment, not retention actions.

### Single Most Likely Lever by Detractor Profile

Given a predicted Detractor, the recommended first action depends on the customer profile :

**New Detractor (tenure < 12 months) :**
Contact within 30 days. The dominant SHAP driver is `tenure_x_contract`.
→ **Offer migration to an annual contract with a first-year discount.**
Rationale : locking in a commitment reduces immediate churn risk while the operator earns time to improve the experience.

**Long-tenured Detractor (tenure ≥ 12 months) :**
Investigate the incident history first. The dominant driver shifts to `monthly_charge_ratio` and `refund_rate`.
→ **Offer a loyalty discount or service upgrade, conditioned on resolving any open incidents.**
Rationale : a price reduction without addressing the root cause of dissatisfaction will not retain this customer long-term.

**Important caveat.** These are observed correlations, not proven causal relationships. A/B testing is required before scaling any intervention. Acting on correlations without causal validation risks wasting retention budget or, worse, reinforcing patterns that do not actually drive satisfaction.

---

## 7. Fairness and Bias

The model allocates retention budget by prioritizing predicted Detractors. If it systematically misses Detractors from a specific demographic group, that group receives less proactive support — not because they are less dissatisfied, but because the model fails them.

We audited Detractor Recall across four demographic segments :

| Group | Detractor Recall | Status |
|---|---|---|
| Gender (Male vs Female) | 87.4% vs 89.7% | OK — gap < 10 pts |
| Senior Citizen (Yes vs No) | 88.3% vs 88.6% | OK — negligible gap |
| Under 30 (Yes vs No) | 88.2% vs 88.6% | OK — negligible gap |
| **Married (Yes vs No)** | **79.5% vs 93.3%** | **ALERT — 13.8 pt gap** |

The model captures only 79.5% of Detractors among married customers versus 93.3% among unmarried customers. This 13.8-point gap means married dissatisfied customers are systematically less likely to receive a retention call.

**This finding must be escalated to the Customer Experience and Legal teams before any production deployment.**

On geographic features : Zip Code and city can proxy for socioeconomic status. We retained them because in a telecom context they carry legitimate network quality signal. If legal review determines they introduce protected-class proxying, they should be dropped — at a measured and documented cost to model performance.

---

## 8. Limitations and Next Steps

### What is implemented

- Label construction from Satisfaction Score with enrichment, sensitivity analysis, and controlled noise
- Full modelling pipeline : dummy → logistic → ordinal → LightGBM with threshold tuning
- SHAP interpretability at individual and segment level
- Fairness audit across four demographic groups with escalation guidance
- Streamlit interface for retention managers in the field
- Lightweight monitoring with TabularDrift (Alibi-Detect) for input and prediction drift

### What is approximate

- **The label is a proxy.** Satisfaction Score 1-5 is not NPS 0-10. The mapping introduces label noise that propagates through all downstream metrics.
- **Synthetic verbatims.** Generated by Llama3 conditioned on customer features. Useful for demonstrating the multimodal pipeline, not a substitute for real call center transcripts.
- **The monitoring runs on a simulation.** Without production deployment, monitor.py compares train vs test — not real incoming traffic vs a stable reference.
- **The married customer gap is identified but not corrected.** Root cause investigation is required before any reweighting or fairness constraint.

### Feedback Loop — The Production Risk

If the model identifies Detractors and the retention team contacts all of them, some will become satisfied. At the next survey wave, they respond as Promoters. The model, retrained on this data, learns that their profile corresponds to a Promoter — and stops flagging them.

Over time, the model quietly degrades without any visible signal of failure.

The solution is a **holdout group** : 10% of predicted Detractors are not contacted and serve as an uncontaminated control group for measuring true model performance and providing clean labels for retraining.

This must be designed into the deployment architecture from day one — it cannot be retrofitted after the fact.

### What is left as future work

1. **Real verbatims** : integrate actual call center transcripts to test whether text adds signal beyond tabular features.
2. **Census enrichment** : add ZIP-code-level socioeconomic features to improve geographic signal without raw location proxies.
3. **Holdout group** : implement 10% control group at deployment to prevent feedback loop contamination.
4. **Automated retraining pipeline** : trigger retraining when drift is detected on >= 5 features or PSI > 0.20.
5. **Married customer investigation** : understand the root cause of the 13.8-point fairness gap before production.

---

*This work was completed as part of the Artefact take-home challenge. LLM tools (Claude by Anthropic, Llama3 via Ollama) were used for code scaffolding, documentation drafting, and verbatim generation. All modelling decisions, analytical choices, and conclusions are the responsibility of the candidate.*