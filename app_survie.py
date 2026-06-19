import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
import pickle
import os

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de primo-nuptialité - Cameroun",
    page_icon="🇨🇲",
    layout="wide"
)

# CSS personnalisé
st.markdown("""
<style>
    .stat-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        border-bottom: 3px solid #007A5E;
        margin: 5px;
    }
    .stat-card .value {
        font-size: 28px;
        font-weight: bold;
        color: #1a1a2e;
    }
    .stat-card .label {
        font-size: 14px;
        color: #666;
    }
    .main-header {
        background: linear-gradient(90deg, #007A5E 0%, #007A5E 33%, #FCD116 33%, #FCD116 66%, #CE1126 66%, #CE1126 100%);
        padding: 5px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .main-title {
        background: rgba(255,255,255,0.95);
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        margin: 5px;
    }
    .main-title h1 {
        color: #1a1a2e;
        font-size: 32px;
        margin: 0;
    }
    .main-title p {
        color: #555;
        font-size: 16px;
        margin: 5px 0 0 0;
    }
    .result-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #ddd;
        margin: 10px 0;
    }
    .prob-high {
        background: #007A5E;
        color: white;
        padding: 5px 15px;
        border-radius: 5px;
        font-weight: bold;
    }
    .prob-moderate {
        background: #FCD116;
        color: #1a1a2e;
        padding: 5px 15px;
        border-radius: 5px;
        font-weight: bold;
    }
    .prob-low {
        background: #CE1126;
        color: white;
        padding: 5px 15px;
        border-radius: 5px;
        font-weight: bold;
    }
    .footer {
        text-align: center;
        padding: 20px 0 10px 0;
        color: #666;
        border-top: 2px solid #eee;
        margin-top: 30px;
    }
    
    /* Bouton Prédire en vert */
    .stButton > button {
        background-color: #007A5E !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        padding: 0.5rem 2rem !important;
        border-radius: 8px !important;
        width: 100% !important;
        transition: background-color 0.3s !important;
    }
    
    .stButton > button:hover {
        background-color: #005a44 !important;
        color: white !important;
        border: none !important;
    }
    
    .stButton > button:active {
        background-color: #004433 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_data():
    model_path = "gbs_model.pkl"
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model, scaler, feature_columns = pickle.load(f)
        return model, scaler, feature_columns
    return None, None, None

@st.cache_data
def load_data():
    if not os.path.exists('primo_nuptialite_clean.csv'):
        return None
    return pd.read_csv('primo_nuptialite_clean.csv')

def plot_survival_curve(surv_func, title):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.step(surv_func.x, surv_func(surv_func.x), where='post', 
            color='#007A5E', linewidth=3, label='Probabilite de survie')
    ax.fill_between(surv_func.x, 0, surv_func(surv_func.x), alpha=0.2, color='#007A5E')
    ax.axhline(y=0.5, color='#CE1126', linestyle='--', alpha=0.5)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel("Age (annees)")
    ax.set_ylabel("S(t) - Probabilite de ne pas etre en union")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig

def main():
    # En-tete
    st.markdown("""
    <div class="main-header">
        <div class="main-title">
            <h1>Prediction de primo-nuptialite au Cameroun</h1>
            <p>Modele Gradient Boosting Survival Analysis - EDS Cameroun 2018</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    model, scaler, feature_columns = load_model_and_data()
    data = load_data()
    
    if model is None or data is None:
        st.error("Erreur: Fichiers manquants. Veuillez verifier que 'gbs_model.pkl' et 'primo_nuptialite_clean.csv' sont presents.")
        return
    
    # Sidebar
    with st.sidebar:
        st.subheader("Caracteristiques de l'individu")
        st.divider()
        age = st.slider("Age actuel", 15, 55, 25)
        st.divider()
        milieu = st.selectbox("Milieu", ["Urbain", "Rural"])
        instruction = st.selectbox("Instruction", ["Aucun", "Primaire", "Secondaire", "Superieur"])
        region = st.selectbox("Region", ["Adamaoua", "Centre", "Est", "Extreme-Nord", "Littoral", 
                                         "Nord", "Nord-Ouest", "Ouest", "Sud", "Sud-Ouest"])
        religion = st.selectbox("Religion", ["Catholique", "Protestant", "Musulman", "Animiste", "Autre"])
        richesse = st.selectbox("Richesse", ["Pauvre", "Moyen", "Riche"])
        st.divider()
        predict = st.button("Predire", use_container_width=True)
    
    # Stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="value">{len(data):,}</div>
            <div class="label">Individus dans la base</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="value">{data['union'].mean():.1%}</div>
            <div class="label">Taux d'union</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        age_moyen = data[data['union']==1]['duree'].mean()
        st.markdown(f"""
        <div class="stat-card">
            <div class="value">{age_moyen:.1f} ans</div>
            <div class="label">Age moyen a l'union</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    if predict:
        input_data = {
            'age_actuel': age,
            'milieu': 0 if milieu == "Urbain" else 1,
            'instruction': ["Aucun", "Primaire", "Secondaire", "Superieur"].index(instruction),
            'region': ["Adamaoua", "Centre", "Est", "Extreme-Nord", "Littoral", 
                       "Nord", "Nord-Ouest", "Ouest", "Sud", "Sud-Ouest"].index(region),
            'religion': ["Catholique", "Protestant", "Musulman", "Animiste", "Autre"].index(religion),
            'richesse': ["Pauvre", "Moyen", "Riche"].index(richesse)
        }
        
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        surv_func = model.predict_survival_function(input_scaled)[0]
        
        def get_prob(age_val):
            idx = np.searchsorted(surv_func.x, age_val)
            if idx >= len(surv_func.x):
                return surv_func(surv_func.x[-1])
            elif surv_func.x[idx] == age_val:
                return surv_func(age_val)
            return surv_func(surv_func.x[idx-1]) if idx > 0 else 1.0
        
        probs = {a: get_prob(a) for a in [25, 30, 35, 40]}
        
        st.subheader("Resultats")
        
        cols = st.columns(4)
        colors = ["#007A5E", "#FCD116", "#CE1126", "#6c757d"]
        for i, (a, p) in enumerate(probs.items()):
            with cols[i]:
                st.markdown(f"""
                <div class="stat-card" style="border-bottom-color: {colors[i]};">
                    <div class="value" style="font-size: 20px;">{p:.1%}</div>
                    <div class="label">Probabilite a {a} ans</div>
                </div>
                """, unsafe_allow_html=True)
        
        prob_30 = probs[30]
        if prob_30 > 0.5:
            cls = "prob-high"
            txt = "Probabilite elevee de ne pas etre en union a 30 ans"
        elif prob_30 > 0.25:
            cls = "prob-moderate"
            txt = "Probabilite moderee de ne pas etre en union a 30 ans"
        else:
            cls = "prob-low"
            txt = "Probabilite faible de ne pas etre en union a 30 ans"
        
        st.markdown(f"""
        <div class="result-box">
            <span style="font-weight:bold;">Interpretation : </span>
            <span class="{cls}">{txt}</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("Courbe de survie")
        fig = plot_survival_curve(surv_func, f"Courbe de survie - {milieu} - {instruction}")
        st.pyplot(fig)
        
        with st.expander("Details"):
            st.write("**Caracteristiques:**")
            st.write(f"- Milieu: {milieu}")
            st.write(f"- Instruction: {instruction}")
            st.write(f"- Region: {region}")
            st.write(f"- Religion: {religion}")
            st.write(f"- Richesse: {richesse}")
            st.write("**Probabilites:**")
            for a in [18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 45, 50]:
                st.write(f"- A {a} ans: {get_prob(a):.1%}")
    
    with st.expander("A propos du modele"):
        st.markdown("""
        **Modele:** Gradient Boosting Survival Analysis
        
        Predis la probabilite de rester celibataire selon:
        - Age, Milieu, Instruction, Region, Religion, Richesse
        
        **Performance:**
        - C-index Train: 0.6824
        - C-index Test: 0.6762
        
        *Source: EDS Cameroun 2018*
        """)
    
    st.markdown("---")
    st.write("Developpe pour l'analyse de la primo-nuptialite au Cameroun")

if __name__ == "__main__":
    main()
