import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NPS Retention Manager",
    page_icon="📊",
    layout="wide"
)

COLORS = {
    'Detracteur': '#e74c3c',
    'Passif'    : '#f39c12',
    'Promoteur' : '#2ecc71'
}
# Leviers actionnables identifies en section 4.6
LEVERS = {
    'tenure_x_contract'   : 'Proposer migration vers contrat annuel',
    'monthly_charge_ratio': 'Proposer une remise sur la charge mensuelle',
    'charge_per_service'  : 'Proposer un bundle de services',
    'Contract'            : 'Proposer migration vers contrat annuel',
    'Monthly Charge'      : 'Proposer une remise personnalisee',
    'nb_services'         : 'Proposer des services complementaires',
    'digital_engagement'  : 'Programme onboarding digital',
    'has_security'        : 'Proposer services de securite en ligne',
    'Offer'               : 'Cibler avec une offre promotionnelle',
    'has_streaming'       : 'Proposer bundle streaming',
    'Number of Referrals' : 'Activer programme de parrainage',
    'has_referred'        : 'Activer programme de parrainage',
}

# Encodages (correspondent aux LabelEncoders de 4.2)
CONTRACT_ENC = {'Month-to-Month': 0, 'One Year': 1, 'Two Year': 2}
INTERNET_ENC = {'Cable': 0, 'DSL': 1, 'Fiber Optic': 2, 'No Internet': 3}
PAYMENT_ENC  = {'Bank Withdrawal': 0, 'Credit Card': 1, 'Mailed Check': 2}
OFFER_ENC    = {'Aucune': 0, 'Offer A': 1, 'Offer B': 2,
                'Offer C': 3, 'Offer D': 4, 'Offer E': 5}
YN           = {'No': 0, 'Yes': 1}

# ── CHARGEMENT ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model_data = joblib.load('../notebook/nps_model_final.pkl')
    data       = joblib.load('../notebook/split_data.pkl')
    master     = pd.read_csv('../notebook/telco_nps_master.csv')
    try:
        demo = pd.read_excel(
            '../artifacts/raw/Telco_customer_churn_demographics.xlsx'
        )
    except FileNotFoundError:
        demo = pd.DataFrame()
    return model_data, data, master, demo

@st.cache_resource
def load_explainer(_lgbm):
    return shap.TreeExplainer(_lgbm)

model_data, data, master, demo = load_artifacts()
lgbm         = model_data['model']
threshold    = model_data['threshold']
classes      = model_data['classes']
feature_cols = model_data['feature_cols']
X_test       = data['X_test']
explainer    = load_explainer(lgbm)
MEDIANS      = X_test[feature_cols].median().to_dict()

# ── FONCTIONS ─────────────────────────────────────────────────────────────────
def predict_nps(X_row):
    proba = lgbm.predict_proba(X_row)[0]
    p_det = proba[list(classes).index('Detracteur')]
    label = 'Detracteur' if p_det >= threshold \
            else lgbm.predict(X_row)[0]
    return label, dict(zip(classes, proba.round(3)))

def get_shap(X_row, n=5):
    sv      = explainer.shap_values(X_row)
    det_idx = list(classes).index('Detracteur')
    sv_det  = sv[:, :, det_idx][0]
    return pd.DataFrame({
        'Feature': feature_cols,
        'SHAP'   : sv_det,
        'Abs'    : np.abs(sv_det)
    }).sort_values('Abs', ascending=False).head(n)

def get_top_lever(shap_df):
    for _, r in shap_df.iterrows():
        if r['Feature'] in LEVERS:
            return r['Feature'], LEVERS[r['Feature']]
    return None, "Analyser le profil complet du client"

def build_row(**kwargs):
    X = pd.DataFrame([MEDIANS], columns=feature_cols)
    for k, v in kwargs.items():
        if k in X.columns:
            X[k] = v
    return X

def add_totals(X, tenure, monthly_chg, avg_ld):
    X['Total Charges']               = tenure * monthly_chg
    X['Total Revenue']               = tenure * monthly_chg * 1.05
    X['Total Long Distance Charges'] = tenure * avg_ld
    X['Total Refunds']               = 0.0
    X['Total Extra Data Charges']    = 0.0
    X['refund_rate']                 = 0.0
    return X

def add_engineered(X, tenure, monthly_chg, contract_enc, nb_svc,
                   referrals, paperless_enc, autopay,
                   online_sec, dev_prot, stream_tv, stream_mov):
    X['nb_services']          = nb_svc
    X['tenure_x_contract']    = tenure * (contract_enc + 1)
    X['charge_per_service']   = monthly_chg / (nb_svc + 1)
    X['monthly_charge_ratio'] = monthly_chg / (tenure + 1)
    X['has_security']         = online_sec + dev_prot
    X['has_streaming']        = stream_tv + stream_mov
    X['digital_engagement']   = paperless_enc + autopay
    X['has_referred']         = int(referrals > 0)
    return X

# ── DISPLAY ───────────────────────────────────────────────────────────────────
def show_prediction(label, probas):
    color = COLORS[label]
    st.markdown(
        f"<div style='background:{color}22;border-left:5px solid {color};"
        f"padding:15px;border-radius:5px;margin:10px 0'>"
        f"<h2 style='color:{color};margin:0'>🎯 {label}</h2></div>",
        unsafe_allow_html=True
    )
    c1, c2, c3 = st.columns(3)
    for col, cls in zip([c1, c2, c3],
                        ['Detracteur', 'Passif', 'Promoteur']):
        col.metric(cls, f"{probas.get(cls, 0):.1%}")

def show_shap(shap_df):
    fig, ax = plt.subplots(figsize=(8, 3))
    colors  = ['#e74c3c' if v > 0 else '#2ecc71' for v in shap_df['SHAP']]
    ax.barh(shap_df['Feature'], shap_df['SHAP'], color=colors)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('Impact sur P(Detracteur) | Rouge: aggrave | Vert: attenue')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

def show_recommendation(label, tenure, shap_df):
    feat, action = get_top_lever(shap_df)
    if label == 'Detracteur':
        profile = "Nouveau Detracteur" if tenure < 12 \
                  else "Detracteur ancien"
        timing  = "Contacter dans les 30 jours." if tenure < 12 \
                  else "Analyser l historique d incidents."
        st.warning(
            f"**{profile}** — {timing}  \n"
            f"Levier principal : **{feat}**  \n"
            f"Action : {action}"
        )
    elif label == 'Passif':
        st.info("**Passif** — Surveiller. "
                "Proposer une offre d engagement long terme.")
    else:
        st.success("**Promoteur** — Client satisfait. "
                   "Activer programme de parrainage.")

def show_fairness_alert(married):
    if married == 'Yes':
        st.warning(
            "⚠️ **Alerte Fairness (section 4.7)**  \n"
            "Ce client est marie.  \n"
            "Recall Detracteur : **79.5%** sur maries vs **93.3%** non-maries.  \n"
            "Prediction a interpreter avec precaution — escalader a l equipe CX."
        )

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
st.sidebar.title("📊 NPS Retention Manager")
st.sidebar.caption(f"Modele : LightGBM | Threshold : {threshold:.2f}")
st.sidebar.markdown("---")
mode = st.sidebar.radio(
    "Mode de saisie",
    ["🔍 Par Customer ID", "✏️ Saisie manuelle"]
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Couverture des features**  \n"
    "22 saisies | 9 reconstruites | 9 fixes  \n\n"
    "**Graceful** : champ vide → mediane dataset  \n\n"
    "**Bornes** : IBM Telco v11.1.3+"
)

st.title("📊 NPS Prediction — Retention Manager")
st.markdown(
    "Predit la categorie NPS **(Detracteur / Passif / Promoteur)** "
    "et identifie les leviers d action retention."
)
st.markdown("---")

# ── MODE 1 : CUSTOMER ID ──────────────────────────────────────────────────────
if mode == "🔍 Par Customer ID":
    st.subheader("🔍 Recherche par Customer ID")
    cid = st.text_input("Customer ID", placeholder="Ex: 0002-ORFBO")

    if cid:
        row = master[master['Customer ID'] == cid]
        if row.empty:
            st.error(f"Customer ID '{cid}' introuvable.")
            st.stop()

        st.success(f"Client trouve : {cid}")
        idx = row.index[0]

        with st.expander("📋 Profil du client", expanded=True):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Anciennete",
                      f"{row['Tenure in Months'].values[0]:.0f} mois")
            c2.metric("Contrat", row['Contract'].values[0])
            c3.metric("Charge mensuelle",
                      f"${row['Monthly Charge'].values[0]:.0f}")
            c4.metric("Internet",
                      str(row['Internet Type'].values[0])
                      if pd.notna(row['Internet Type'].values[0])
                      else "Sans internet")
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("NPS reel",
                      row['NPS_label'].values[0]
                      if 'NPS_label' in row.columns else "N/A")
            c6.metric("Churn",
                      row['Churn Label'].values[0]
                      if 'Churn Label' in row.columns else "N/A")
            c7.metric("Nb services",
                      f"{row['nb_services'].values[0]:.0f}"
                      if 'nb_services' in row.columns else "N/A")
            c8.metric("Referrals",
                      f"{row['Number of Referrals'].values[0]:.0f}")

        if idx not in X_test.index:
            st.info(
                "Ce client est dans le train set (repondant au sondage). "
                "Son NPS est deja connu."
            )
            st.stop()

        X_row = X_test.loc[[idx], feature_cols]
        label, probas = predict_nps(X_row)

        st.subheader("🎯 Prediction NPS")
        show_prediction(label, probas)

        if not demo.empty:
            demo_row = demo[demo['Customer ID'] == cid]
            if not demo_row.empty:
                show_fairness_alert(str(demo_row['Married'].values[0]))

        st.subheader("🔑 Drivers de cette prediction")
        shap_df = get_shap(X_row)
        show_shap(shap_df)

        st.subheader("💡 Recommandation")
        show_recommendation(label, row['Tenure in Months'].values[0], shap_df)

# ── MODE 2 : SAISIE MANUELLE ──────────────────────────────────────────────────
elif mode == "✏️ Saisie manuelle":
    st.subheader("✏️ Saisie manuelle du profil client")
    st.caption(
        "Bornes issues du dataset IBM Telco v11.1.3+. "
        "Champ non renseigne → mediane dataset. "
        "Variables financieres cumulees reconstruites automatiquement."
    )

    st.markdown("**📄 Contrat & Facturation**")
    c1, c2, c3 = st.columns(3)
    with c1:
        tenure   = st.slider("Anciennete (mois)", 1, 72, 29)
        contract = st.selectbox("Type de contrat",
                                ["Month-to-Month", "One Year", "Two Year"])
    with c2:
        monthly_chg = st.number_input("Charge mensuelle ($)",
                                      18.0, 119.0, 70.0, step=1.0)
        payment     = st.selectbox("Methode de paiement",
                                   ["Bank Withdrawal", "Credit Card",
                                    "Mailed Check"])
    with c3:
        paperless = st.selectbox("Facturation electronique", ["Yes", "No"])
        offer     = st.selectbox("Offre commerciale",
                                 ["Aucune", "Offer A", "Offer B",
                                  "Offer C", "Offer D", "Offer E"])

    st.markdown("**🌐 Services Internet**")
    c4, c5, c6 = st.columns(3)
    with c4:
        internet_svc  = st.selectbox("Service Internet", ["Yes", "No"])
        internet_type = st.selectbox(
            "Type d internet",
            ["Fiber Optic", "Cable", "DSL", "No Internet"],
            disabled=(internet_svc == "No")
        )
        avg_gb    = st.slider("Consommation avg (GB/mois)", 0, 85, 17,
                              disabled=(internet_svc == "No"))
        unlimited = st.selectbox("Donnees illimitees", ["No", "Yes"],
                                 disabled=(internet_svc == "No"))
    with c5:
        online_sec = st.selectbox("Securite en ligne",    ["No", "Yes"])
        online_bck = st.selectbox("Sauvegarde en ligne",  ["No", "Yes"])
        dev_prot   = st.selectbox("Protection appareil",  ["No", "Yes"])
        tech_sup   = st.selectbox("Support technique premium", ["No", "Yes"])
    with c6:
        stream_tv  = st.selectbox("Streaming TV",      ["No", "Yes"])
        stream_mov = st.selectbox("Streaming films",   ["No", "Yes"])
        stream_mus = st.selectbox("Streaming musique", ["No", "Yes"])

    st.markdown("**📱 Telephonie**")
    c7, c8 = st.columns(2)
    with c7:
        phone_svc   = st.selectbox("Service telephonie", ["Yes", "No"])
        multi_lines = st.selectbox("Lignes multiples", ["No", "Yes"],
                                   disabled=(phone_svc == "No"))
    with c8:
        avg_ld = st.number_input("Charges longue distance avg ($/mois)",
                                 0.0, 50.0, 23.0, step=0.5)

    st.markdown("**🤝 Engagement & Parrainage**")
    c9, c10 = st.columns(2)
    with c9:
        referrals = st.slider("References donnees", 0, 11, 0)
        referred  = st.selectbox("A recommande un ami", ["No", "Yes"])
    with c10:
        st.info(
            "**Reconstruites automatiquement :**  \n"
            "Total Charges = Tenure × Monthly Charge  \n"
            "Total Revenue = Total Charges × 1.05  \n"
            "Total LD = Tenure × Avg LD  \n"
            "Refund rate = 0 | Geo = medianes dataset"
        )

    st.markdown("**👤 Demographie** *(optionnel — alerte fairness 4.7)*")
    married = st.selectbox("Marie(e)", ["Non renseigne", "Yes", "No"])

    if st.button("🎯 Predire le NPS", type="primary"):
        contract_enc  = CONTRACT_ENC[contract]
        internet_enc  = INTERNET_ENC.get(
            internet_type if internet_svc == "Yes" else "No Internet", 3
        )
        paperless_enc = YN[paperless]
        autopay       = int(payment == "Bank Withdrawal")

        nb_svc = sum([
            YN[online_sec], YN[online_bck], YN[dev_prot], YN[tech_sup],
            YN[stream_tv], YN[stream_mov], YN[stream_mus],
            YN[multi_lines] if phone_svc == "Yes" else 0,
        ])

        X_row = build_row(**{
            'Tenure in Months'                 : tenure,
            'Contract'                         : contract_enc,
            'Monthly Charge'                   : monthly_chg,
            'Payment Method'                   : PAYMENT_ENC[payment],
            'Paperless Billing'                : paperless_enc,
            'Offer'                            : OFFER_ENC[offer],
            'Internet Service'                 : YN[internet_svc],
            'Internet Type'                    : internet_enc,
            'Avg Monthly GB Download'          : avg_gb if internet_svc == "Yes" else 0,
            'Unlimited Data'                   : YN[unlimited],
            'Online Security'                  : YN[online_sec],
            'Online Backup'                    : YN[online_bck],
            'Device Protection Plan'           : YN[dev_prot],
            'Premium Tech Support'             : YN[tech_sup],
            'Streaming TV'                     : YN[stream_tv],
            'Streaming Movies'                 : YN[stream_mov],
            'Streaming Music'                  : YN[stream_mus],
            'Phone Service'                    : YN[phone_svc],
            'Multiple Lines'                   : YN[multi_lines] if phone_svc == "Yes" else 0,
            'Number of Referrals'              : referrals,
            'Referred a Friend'                : YN[referred],
            'Avg Monthly Long Distance Charges': avg_ld,
        })

        X_row = add_totals(X_row, tenure, monthly_chg, avg_ld)
        X_row = add_engineered(
            X_row, tenure, monthly_chg, contract_enc, nb_svc,
            referrals, paperless_enc, autopay,
            YN[online_sec], YN[dev_prot], YN[stream_tv], YN[stream_mov]
        )

        st.markdown("---")
        st.subheader("🎯 Prediction NPS")
        label, probas = predict_nps(X_row[feature_cols])
        show_prediction(label, probas)

        if married == "Yes":
            show_fairness_alert("Yes")

        st.subheader("🔑 Drivers de cette prediction")
        shap_df = get_shap(X_row[feature_cols])
        show_shap(shap_df)

        st.subheader("💡 Recommandation")
        show_recommendation(label, tenure, shap_df)
