import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
import pickle
import os
from PIL import Image
import requests
from io import BytesIO

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de primo-nuptialité - Cameroun",
    page_icon="🇨🇲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
<style>
    /* Couleurs du Cameroun */
    .main-header {
        background: linear-gradient(90deg, #007A5E 0%, #007A5E 33%, #FCD116 33%, #FCD116 66%, #CE1126 66%, #CE1126 100%);
        padding: 0.3rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
    }
    
    .main-title {
        background-color: rgba(255,255,255,0.92);
        padding: 1.2rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.3rem;
    }
    
    .main-title h1 {
        color: #1a1a2e;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .main-title p {
        color: #4a4a6a;
        font-size: 1rem;
        margin: 0.3rem 0 0 0;
    }
    
    .stat-card {
        background: white;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        border-bottom: 3px solid #007A5E;
        margin: 0.2rem;
    }
    
    .stat-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    
    .stat-card .label {
        font-size: 0.85rem;
        color: #6c757d;
        margin-top: 0.2rem;
    }
    
    .result-box {
        background: #f8f9fa;
        padding: 1.2rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 0.8rem 0;
    }
    
    .prob-high {
        background-color: #007A5E;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    
    .prob-moderate {
        background-color: #FCD116;
        color: #1a1a2e;
        padding: 0.3rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    
    .prob-low {
        background-color: #CE1126;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    
    .sidebar-title {
        color: #007A5E;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.8rem;
    }
    
    .footer {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        color: #6c757d;
        border-top: 2px solid #e9ecef;
        margin-top: 2rem;
    }
    
    .stButton > button {
        background: #007A5E;
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        width: 100%;
        transition: background 0.2s;
    }
    
    .stButton > button:hover {
        background: #005a44;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Chargement du modèle et des données
@st.cache_resource
def load_model_and_data():
    model_path = "gbs_model.pkl"
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model, scaler, feature_columns = pickle.load(f)
        return model, scaler, feature_columns
    else:
        return train_model()

@st.cache_data
def train_model():
    data = pd.read_csv('primo_nuptialite_clean.csv')
    data = data.dropna()
    
    cat_cols = ['milieu', 'instruction', 'richesse', 'region', 'religion']
    df_ml = data.copy()
    le = LabelEncoder()
    for col in cat_cols:
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    
    X = df_ml.drop(columns=['duree', 'union'])
    y = Surv.from_dataframe('union', 'duree', df_ml)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GradientBoostingSurvivalAnalysis(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    model.fit(X_scaled, y)
    
    with open("gbs_model.pkl", 'wb') as f:
        pickle.dump((model, scaler, X.columns.tolist()), f)
    
    return model, scaler, X.columns.tolist()

@st.cache_data
def load_data():
    data = pd.read_csv('primo_nuptialite_clean.csv')
    return data

# Fonction pour créer le graphique de survie
def plot_survival_curve(surv_func, title="Courbe de survie prédite"):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    ax.step(surv_func.x, surv_func(surv_func.x), where='post', 
            color='#007A5E', linewidth=3, label='Probabilité de survie')
    ax.fill_between(surv_func.x, 0, surv_func(surv_func.x), 
                     alpha=0.2, color='#007A5E')
    
    ax.axhline(y=0.5, color='#CE1126', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.text(surv_func.x[-1]*0.85, 0.52, 'Référence 0.5', color='#CE1126', fontsize=10)
    
    ax.set_title(title, fontsize=16, fontweight='bold', color='#1a1a2e')
    ax.set_xlabel("Âge (années)", fontsize=12)
    ax.set_ylabel("S(t) - Probabilité de ne pas être en union", fontsize=12)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 55)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    return fig

# Fonction principale
def main():
    # En-tête
    st.markdown("""
    <div class="main-header">
        <div class="main-title">
            <h1>Prédiction de primo-nuptialité au Cameroun</h1>
            <p>Modèle Gradient Boosting Survival Analysis - EDS Cameroun 2018</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Chargement
    with st.spinner("Chargement du modèle en cours..."):
        model, scaler, feature_columns = load_model_and_data()
        data = load_data()
    
    # Sidebar
    with st.sidebar:
        st.markdown('<p class="sidebar-title">Caractéristiques de l\'individu</p>', unsafe_allow_html=True)
        st.divider()
        
        age = st.slider("Âge actuel", 15, 55, 25, 1)
        
        st.divider()
        
        milieu = st.selectbox("Milieu de résidence", ["Urbain", "Rural"])
        instruction = st.selectbox("Niveau d'instruction", ["Aucun", "Primaire", "Secondaire", "Supérieur"])
        region = st.selectbox("Région", ["Adamaoua", "Centre", "Est", "Extrême-Nord", "Littoral", 
                                         "Nord", "Nord-Ouest", "Ouest", "Sud", "Sud-Ouest"])
        religion = st.selectbox("Religion", ["Catholique", "Protestant", "Musulman", "Animiste", "Autre"])
        richesse = st.selectbox("Niveau de richesse", ["Pauvre", "Moyen", "Riche"])
        
        st.divider()
        predict_button = st.button("Prédire la survie", use_container_width=True)
    
    # Statistiques générales
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(label="Individus dans la base", value=f"{len(data):,}")
    
    with col2:
        st.metric(label="Taux d'union", value=f"{data['union'].mean():.1%}")
    
    with col3:
        age_moyen = data[data['union'] == 1]['duree'].mean()
        st.metric(label="Âge moyen à l'union", value=f"{age_moyen:.1f} ans")
    
    st.divider()
    
    # Prédiction
    if predict_button:
        # Préparation des données
        input_data = {
            'age_actuel': age,
            'milieu': 0 if milieu == "Urbain" else 1,
            'instruction': ["Aucun", "Primaire", "Secondaire", "Supérieur"].index(instruction),
            'region': ["Adamaoua", "Centre", "Est", "Extrême-Nord", "Littoral", 
                       "Nord", "Nord-Ouest", "Ouest", "Sud", "Sud-Ouest"].index(region),
            'religion': ["Catholique", "Protestant", "Musulman", "Animiste", "Autre"].index(religion),
            'richesse': ["Pauvre", "Moyen", "Riche"].index(richesse)
        }
        
        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        
        # Prédiction
        surv_func = model.predict_survival_function(input_scaled)[0]
        
        def get_prob(age_val):
            idx = np.searchsorted(surv_func.x, age_val)
            if idx >= len(surv_func.x):
                return surv_func(surv_func.x[-1])
            elif surv_func.x[idx] == age_val:
                return surv_func(age_val)
            else:
                return surv_func(surv_func.x[idx-1]) if idx > 0 else 1.0
        
        probs = {age_val: get_prob(age_val) for age_val in [25, 30, 35, 40]}
        
        # Résultats
        st.markdown("## Résultats de la prédiction")
        
        cols = st.columns(4)
        for i, (age_val, prob) in enumerate(probs.items()):
            with cols[i]:
                st.metric(label=f"Probabilité à {age_val} ans", value=f"{prob:.1%}")
        
        # Interprétation
        prob_30 = probs[30]
        if prob_30 > 0.5:
            interpretation = "Probabilité élevée de ne pas être en union à 30 ans"
        elif prob_30 > 0.25:
            interpretation = "Probabilité modérée de ne pas être en union à 30 ans"
        else:
            interpretation = "Probabilité faible de ne pas être en union à 30 ans"
        
        st.markdown(f"**Interprétation :** {interpretation}")
        
        # Courbe de survie
        st.markdown("### Courbe de survie prédite")
        fig = plot_survival_curve(surv_func, f"Courbe de survie - {milieu} - {instruction}")
        st.pyplot(fig)
        
        # Détails
        with st.expander("Détails de la prédiction"):
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("**Caractéristiques de l'individu :**")
                st.write(f"- Milieu : {milieu}")
                st.write(f"- Instruction : {instruction}")
                st.write(f"- Région : {region}")
                st.write(f"- Religion : {religion}")
                st.write(f"- Niveau de richesse : {richesse}")
            
            with col_d2:
                st.markdown("**Probabilités de survie :**")
                for age_plot in [18, 20, 22, 25, 28, 30, 32, 35, 38, 40, 45, 50]:
                    st.write(f"- À {age_plot} ans : {get_prob(age_plot):.1%}")
    
    # Informations sur le modèle
    with st.expander("À propos du modèle"):
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.markdown("""
            **Modèle : Gradient Boosting Survival Analysis**
            
            Ce modèle prédit la probabilité qu'un individu reste célibataire (non en union) 
            en fonction de ses caractéristiques démographiques et socio-économiques.
            
            **Variables utilisées :**
            - Âge actuel
            - Milieu de résidence (Urbain/Rural)
            - Niveau d'instruction
            - Région
            - Religion
            - Niveau de richesse
            """)
        with col_m2:
            st.markdown("""
            **Performance du modèle :**
            
            | Métrique | Valeur |
            |----------|--------|
            | C-index (Train) | 0.6824 |
            | C-index (Test) | 0.6762 |
            
            **Source des données :**
            Enquête Démographique et de Santé (EDS) Cameroun 2018
            """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Développé pour l'analyse de la primo-nuptialité au Cameroun</p>
        <p style="font-size: 0.75rem; color: #adb5bd;">Données EDS Cameroun 2018 - Modèle Gradient Boosting Survival</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
