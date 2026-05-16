import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from alibi_detect.cd import TabularDrift

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── RECONSTRUCTION DES DONNEES ────────────────────────────────────────────────
# On reconstruit X_train et X_test depuis les fichiers bruts
# car le split_data.pkl peut avoir des problemes de compatibilite pandas
print("Chargement et reconstruction des donnees...")

services = pd.read_excel('../artifacts/raw/Telco_customer_churn_services.xlsx')
status   = pd.read_excel('../artifacts/raw/Telco_customer_churn_status.xlsx')
location = pd.read_excel('../artifacts/raw/Telco_customer_churn_location.xlsx')

status_cols = ['Customer ID','Satisfaction Score','Churn Label','Churn Value',
               'Churn Score','CLTV','Churn Category','Churn Reason','Customer Status']
services_cols = ['Customer ID','Referred a Friend','Number of Referrals',
                 'Tenure in Months','Offer','Phone Service',
                 'Avg Monthly Long Distance Charges','Multiple Lines',
                 'Internet Service','Internet Type','Avg Monthly GB Download',
                 'Online Security','Online Backup','Device Protection Plan',
                 'Premium Tech Support','Streaming TV','Streaming Movies',
                 'Streaming Music','Unlimited Data','Contract',
                 'Paperless Billing','Payment Method','Monthly Charge',
                 'Total Charges','Total Refunds','Total Extra Data Charges',
                 'Total Long Distance Charges','Total Revenue']
location_cols = ['Customer ID','City','Zip Code','Latitude','Longitude']

df = (services[services_cols]
      .merge(status[status_cols], on='Customer ID', how='left')
      .merge(location[location_cols], on='Customer ID', how='left'))

# Label NPS (meme logique que 4.1)
def build_nps(row):
    s, t, r = row['Satisfaction Score'], row['Tenure in Months'], row['Number of Referrals']
    if s == 1:   return 'Detracteur'
    elif s == 2: return 'Passif' if t > 36 else 'Detracteur'
    elif s == 3: return 'Detracteur' if t < 12 else 'Passif'
    elif s == 4: return 'Promoteur' if r > 0 else 'Passif'
    else:        return 'Promoteur'

df['NPS_enrichi'] = df.apply(build_nps, axis=1)
df['NPS_enrichi_v2'] = df['NPS_enrichi'].copy()
idx_score3 = df[df['Satisfaction Score'] == 3].index
idx_flip   = np.random.choice(idx_score3, size=int(len(idx_score3)*0.10), replace=False)
flip = lambda l: 'Passif' if l == 'Detracteur' else ('Detracteur' if l == 'Passif' else l)
df.loc[idx_flip, 'NPS_enrichi_v2'] = df.loc[idx_flip, 'NPS_enrichi_v2'].apply(flip)
df['NPS_label'] = df['NPS_enrichi_v2']

# Preparation features (meme logique que 4.2 + 4.3)
EXCLUDE = ['Customer ID','Satisfaction Score','NPS_baseline','NPS_enrichi',
           'NPS_enrichi_v2','NPS_label','Churn Score','Churn Value','CLTV',
           'Churn Reason','Churn Category','Churn Label','Customer Status']
feature_cols = [c for c in df.columns if c not in EXCLUDE]

df_model = df[feature_cols + ['NPS_label']].copy()
df_model['Offer']        = df_model['Offer'].fillna('No Offer')
df_model['Internet Type'] = df_model['Internet Type'].fillna('No Internet')
df_model['Avg Monthly GB Download'] = df_model['Avg Monthly GB Download'].fillna(0)
df_model['Avg Monthly Long Distance Charges'] = df_model['Avg Monthly Long Distance Charges'].fillna(0)

for col in df_model[feature_cols].select_dtypes(include='object').columns:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))

# Features construites (4.3)
service_cols = ['Online Security','Online Backup','Device Protection Plan',
                'Premium Tech Support','Streaming TV','Streaming Movies',
                'Streaming Music','Multiple Lines']
df_model['nb_services']          = df_model[service_cols].sum(axis=1)
df_model['tenure_x_contract']    = df_model['Tenure in Months'] * (df_model['Contract'] + 1)
df_model['charge_per_service']   = df_model['Monthly Charge'] / (df_model['nb_services'] + 1)
df_model['refund_rate']          = df_model['Total Refunds'] / (df_model['Total Revenue'] + 1)
df_model['monthly_charge_ratio'] = df_model['Monthly Charge'] / (df_model['Tenure in Months'] + 1)
df_model['has_security']         = df_model['Online Security'] + df_model['Device Protection Plan']
df_model['has_streaming']        = df_model['Streaming TV'] + df_model['Streaming Movies']
df_model['digital_engagement']   = df_model['Paperless Billing'] + df_model['Payment Method']
df_model['has_referred']         = (df_model['Number of Referrals'] > 0).astype(int)

all_features = [c for c in df_model.columns if c != 'NPS_label']
X = df_model[all_features]
y = df_model['NPS_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.85, stratify=y, random_state=RANDOM_SEED
)

# Chargement modele
model_data   = joblib.load('../notebook/nps_model_final.pkl')
lgbm         = model_data['model']
threshold    = model_data['threshold']
classes      = model_data['classes']
feature_cols = model_data['feature_cols']

print(f"Train (reference)   : {len(X_train)} clients")
print(f"Test  (production)  : {len(X_test)} clients")
print(f"Features surveillees: {len(feature_cols)}")

# ── MODULE 1 : INPUT DRIFT ────────────────────────────────────────────────────
print("\n" + "="*60)
print("MODULE 1 - Input Drift")
print("="*60)
print("""
Methode    : TabularDrift (Alibi-Detect)
             KS-test sur variables numeriques
             Chi2 sur variables categorielles
Reference  : X_train (repondants sondage - 15%)
Production : X_test  (clients silencieux - 85%)
Seuil      : p-value < 0.05 = drift sur cette feature
""")

X_ref_arr  = X_train[feature_cols].values.astype(float)
X_prod_arr = X_test[feature_cols].values.astype(float)

cat_idx = [i for i, col in enumerate(feature_cols)
           if X_train[col].nunique() <= 10]

detector     = TabularDrift(
    x_ref                  = X_ref_arr,
    p_val                  = 0.05,
    categories_per_feature = {i: None for i in cat_idx}
)
result       = detector.predict(X_prod_arr)
p_vals       = result['data']['p_val']
is_drift     = result['data']['is_drift']
n_drift      = int(np.sum(p_vals < 0.05))

drift_df = pd.DataFrame({
    'Feature': feature_cols,
    'p_value': p_vals.round(4),
    'Drift'  : p_vals < 0.05
}).sort_values('p_value')

print(f"Drift global detecte : {'OUI' if is_drift else 'NON'}")
print(f"Features en drift    : {n_drift}/{len(feature_cols)}")
print(f"\nTop 10 features les plus driftees :")
print(drift_df.head(10).to_string(index=False))

# ── MODULE 2 : PREDICTION DRIFT ───────────────────────────────────────────────
print("\n" + "="*60)
print("MODULE 2 - Prediction Drift (PSI)")
print("="*60)
print("""
Methode    : PSI (Population Stability Index)
             PSI < 0.10  -> Stable
             PSI 0.10-0.20 -> A surveiller
             PSI > 0.20  -> Alerte drift

Simulation : on divise X_test en deux periodes
             Periode A (reference)  = premiers 50%
             Periode B (production) = derniers 50%
""")

det_idx = list(classes).index('Detracteur')
mid     = len(X_test) // 2

def get_pred_dist(X_subset):
    proba = lgbm.predict_proba(X_subset[feature_cols])
    preds = np.where(
        proba[:, det_idx] >= threshold,
        'Detracteur',
        np.array(classes)[np.argmax(proba, axis=1)]
    )
    return {cls: (preds == cls).sum() / len(preds) for cls in classes}

dist_A = get_pred_dist(X_test.iloc[:mid])
dist_B = get_pred_dist(X_test.iloc[mid:])

eps = 1e-6
psi = sum(
    (dist_B.get(c, eps) - dist_A.get(c, eps)) *
    np.log(max(dist_B.get(c, eps), eps) / max(dist_A.get(c, eps), eps))
    for c in classes
)

psi_status = 'STABLE' if psi < 0.10 else 'A SURVEILLER' if psi < 0.20 else 'ALERTE'

print(f"PSI global : {psi:.4f} -> {psi_status}")
print(f"\nDetail par classe :")
for cls in classes:
    a = dist_A.get(cls, 0)
    b = dist_B.get(cls, 0)
    print(f"  {cls:12s} | Periode A: {a:.1%} | Periode B: {b:.1%} | Delta: {b-a:+.1%}")

# ── MODULE 3 : RETRAINING TRIGGER ─────────────────────────────────────────────
print("\n" + "="*60)
print("MODULE 3 - Retraining Trigger")
print("="*60)

DRIFT_THRESHOLD  = 5    # nb features en drift pour declencher
PSI_THRESHOLD    = 0.20 # seuil PSI alerte
LABELS_THRESHOLD = 500  # nb nouveaux labels pour declencher

trigger_drift  = n_drift  >= DRIFT_THRESHOLD
trigger_psi    = psi      >= PSI_THRESHOLD

print(f"Signal 1 - Input Drift     : {'DECLENCHE' if trigger_drift else 'OK':12s} "
      f"({n_drift} features en drift, seuil >= {DRIFT_THRESHOLD})")
print(f"Signal 2 - Prediction Drift: {'DECLENCHE' if trigger_psi else 'OK':12s} "
      f"(PSI={psi:.4f}, seuil >= {PSI_THRESHOLD})")
print(f"Signal 3 - Nouveaux labels : NON SIMULABLE  "
      f"(seuil >= {LABELS_THRESHOLD} labels, necessite donnees production)")

retraining = trigger_drift or trigger_psi

print(f"\n>>> {'RETRAINING RECOMMANDE' if retraining else 'MODELE STABLE - PAS DE RETRAINING NECESSAIRE'} <<<")

if retraining:
    print("""
Procedure recommandee :
  1. Collecter les nouvelles reponses sondage (>= 500)
  2. Verifier l'absence de biais feedback loop (voir README)
  3. Reentreiner sur : train original + nouveaux labels valides
  4. Evaluer Recall Detracteur avant/après sur holdout fixe
  5. Deployer uniquement si Recall Det >= 0.80
""")

# ── RESUME ─────────────────────────────────────────────────────────────────────
print("="*60)
print("RESUME MONITORING")
print("="*60)
print(f"  Input Drift     : {'ALERTE' if is_drift else 'OK':10s} ({n_drift} features en drift)")
print(f"  Prediction Drift: {psi_status:10s} (PSI={psi:.4f})")
print(f"  Retraining      : {'RECOMMANDE' if retraining else 'NON NECESSAIRE'}")
print(f"\nNote : ce monitoring est execute sur une simulation (train vs test).")
print(f"En production : reference = dernier batch valide, production = nouveau batch entrant.")