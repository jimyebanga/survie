import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
import pickle
import os

st.set_page_config(
    page_title="Prédiction primo-nuptialité Cameroun",
    page_icon="🇨🇲",
    layout="wide"
)

# CSS MINIMAL - pour éviter les bugs
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #007A5E 0%, #007A5E 33%, #FCD116 33%, #FCD116 66%, #CE1126 66%, #CE1126 100%);
        padding: 3px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .main-title {
        background: rgba(255,255,255,0.95);
        padding: 15px;
        border-radius: 6px;
        text-align: center;
        margin: 3px;
    }
    .main-title h1 {
        color: #1a1a2e;
        font-size: 28px;
        margin: 0;
    }
    .main-title p {
        color: #555;
        font-size: 14px;
        margin: 3px 0 0 0;
    }
    .stat-card {
        background: white;
        padding: 12px;
        border-radius: 6px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-bottom: 3px solid #007A5E;
    }
    .stat-card .value {
        font-size: 24px;
        font-weight: bold;
        color: #1a1a2e;
    }
    .stat-card .label {
        font-size: 13px;
        color: #666;
    }
    .result-box {
        background: #f8f9fa;
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #ddd;
        margin: 8px 0;
    }
    .prob-high {
        background: #007A5E;
        color: white;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: bold;
    }
    .prob-moderate {
        background: #FCD116;
        color: #1a1a2e;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: bold;
    }
    .prob-low {
        background: #CE1126;
        color: white;
        padding: 4px 12px;
        border-radius: 4px;
        font-weight: bold;
    }
    .stButton > button {
        background-color: #007A5E !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 6px !important;
        width: 100% !important;
    }
    .stButton > button:hover {
        background-color: #005a44 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    if os.path.exists("gbs_model.pkl"):
        with open("gbs_model.pkl", "rb") as f:
            model, scaler, cols = pickle.load(f)
        return model, scaler, cols
    return None, None, None

@st.cache_data
def load_data():
    if os.path.exists("primo_nuptialite_clean.csv"):
        return pd.read_csv("primo_nuptialite_clean.csv")
    return None

def plot_curve(surv_func, title):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.step(surv_func.x, surv_func(surv_func.x), where='post', 
            color='#007A5E', linewidth=3)
    ax.fill_between(surv_func.x, 0, surv_func(surv_func.x), alpha=0.2, color='#007A5E')
    ax.axhline(y=0.5, color='#CE1126', linestyle='--', alpha=0.5)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("Age (annees)")
    ax.set_ylabel("S(t)")
    ax.grid(True, alpha=0.2)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 55)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <div class="main-title">
            <h1>Prédiction primo-nuptialité Cameroun</h1>
            <p>Gradient Boosting Survival - EDS 2018</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    model, scaler, cols = load_model()
    data = load_data()
    
    if model is None or data is None:
        st.error("Fichiers manquants")
        return
    
    # Sidebar
    with st.sidebar:
        st.subheader("Caractéristiques")
        age = st.slider("Age", 15, 55, 25)
        milieu = st.selectbox("Milieu", ["Urbain", "Rural"])
        instruction = st.selectbox("Instruction", ["Aucun", "Primaire", "Secondaire", "Superieur"])
        region = st.selectbox("Region", ["Adamaoua", "Centre", "Est", "Extreme-Nord", "Littoral", 
                                         "Nord", "Nord-Ouest", "Ouest", "Sud", "Sud-Ouest"])
        religion = st.selectbox("Religion", ["Catholique", "Protestant", "Musulman", "Animiste", "Autre"])
        richesse = st.selectbox("Richesse", ["Pauvre", "Moyen", "Riche"])
        predict = st.button("Prédire", use_container_width=True)
    
    # Stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Individus", f"{len(data):,}")
    c2.metric("Taux d'union", f"{data['union'].mean():.1%}")
    age_moyen = data[data['union']==1]['duree'].mean()
    c3.metric("Age moyen union", f"{age_moyen:.1f} ans")
    
    st.divider()
    
    if predict:
        # Préparation
        vals = {
            'age_actuel': age,
            'milieu': 0 if milieu == "Urbain" else 1,
            'instruction': ["Aucun", "Primaire", "Secondaire", "Superieur"].index(instruction),
            'region': ["Adamaoua", "Centre", "Est", "Extreme-Nord", "Littoral", 
                       "Nord", "Nord-Ouest", "Ouest", "Sud", "Sud-Ouest"].index(region),
            'religion': ["Catholique", "Protestant", "Musulman", "Animiste", "Autre"].index(religion),
            'richesse': ["Pauvre", "Moyen", "Riche"].index(richesse)
        }
        
        X = pd.DataFrame([vals])
        X_scaled = scaler.transform(X)
        surv = model.predict_survival_function(X_scaled)[0]
        
        def get_p(a):
            idx = np.searchsorted(surv.x, a)
            if idx >= len(surv.x):
                return surv(surv.x[-1])
            if surv.x[idx] == a:
                return surv(a)
            return surv(surv.x[idx-1]) if idx > 0 else 1.0
        
        # Résultats
        st.subheader("Résultats")
        
        c = st.columns(4)
        ages = [25, 30, 35, 40]
        colors = ["#007A5E", "#FCD116", "#CE1126", "#6c757d"]
        for i, a in enumerate(ages):
            with c[i]:
                st.metric(f"Probabilité à {a} ans", f"{get_p(a):.1%}")
        
        # Interprétation
        p30 = get_p(30)
        if p30 > 0.5:
            cls = "prob-high"
            txt = "Probabilité élevée de ne pas être en union à 30 ans"
        elif p30 > 0.25:
            cls = "prob-moderate"
            txt = "Probabilité modérée de ne pas être en union à 30 ans"
        else:
            cls = "prob-low"
            txt = "Probabilité faible de ne pas être en union à 30 ans"
        
        st.markdown(f"""
        <div class="result-box">
            <b>Interprétation : </b>
            <span class="{cls}">{txt}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Courbe
        st.subheader("Courbe de survie")
        fig = plot_curve(surv, f"{milieu} - {instruction}")
        st.pyplot(fig)
        
        # Détails
        with st.expander("Détails"):
            st.write("**Caractéristiques:**")
            st.write(f"- Milieu: {milieu}")
            st.write(f"- Instruction: {instruction}")
            st.write(f"- Region: {region}")
            st.write(f"- Religion: {religion}")
            st.write(f"- Richesse: {richesse}")
            st.write("**Probabilités:**")
            for a in [18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 45, 50]:
                st.write(f"- {a} ans: {get_p(a):.1%}")
    
    # About
    with st.expander("À propos"):
        st.markdown("""
        **Modèle:** Gradient Boosting Survival Analysis  
        **Variables:** Age, Milieu, Instruction, Region, Religion, Richesse  
        **Performance:** C-index Train 0.6824, Test 0.6762  
        **Source:** EDS Cameroun 2018
        """)
    
    st.divider()
    st.caption("Développé pour l'analyse de la primo-nuptialité au Cameroun")

if __name__ == "__main__":
    main()
