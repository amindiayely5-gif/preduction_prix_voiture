import streamlit as st
import joblib as jb
import numpy as np
from pathlib import Path

# 1. CONFIGURATION
st.set_page_config(
    page_title="Prédiction du prix d'une voiture",
    page_icon="🚗",
    layout="centered"
)

st.title("🚗 Prédiction du Prix de Vente d'un Véhicule")
st.write("Cette application utilise un modèle XGBoost pour prédire le prix de vente d'une voiture.")

# 2. CHARGEMENT DES FICHIERS
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_models():
    model = jb.load(BASE_DIR / "xgb_model.joblib")
    scaler = jb.load(BASE_DIR / "scaler.joblib")
    encoders = jb.load(BASE_DIR / "encoders.joblib")
    uniques = jb.load(BASE_DIR / "uniques.joblib")

    return model, scaler, encoders, uniques

try:
    xgb_model, scaler, encoders, uniques = load_models()
except Exception as e:
    st.error(f"Erreur lors du chargement des modèles : {e}")
    st.stop()

# 3. FORMULAIRE
st.subheader("Estimer le prix d'un véhicule")

Kms_Driven = st.number_input(
    "Kilométrage (Kms Driven)",
    min_value=0,
    value=50000
)

Present_Price = st.number_input(
    "Prix d'origine (en k$)",
    min_value=0.0,
    value=10.0
)

fuel_type_val = st.selectbox("Type de carburant", uniques[0])
seller_type_val = st.selectbox("Type de vendeur", uniques[1])
transmission_val = st.selectbox("Boîte de vitesse", uniques[2])

age_val = st.number_input(
    "Âge du véhicule (années)",
    min_value=0,
    value=5
)

# 4. PRÉDICTION
if st.button("Prédire le prix"):

    try:
        fuel_type_encoded = encoders[0].transform([fuel_type_val])[0]
        seller_type_encoded = encoders[1].transform([seller_type_val])[0]
        transmission_encoded = encoders[2].transform([transmission_val])[0]

        x_new = np.array([[
            Kms_Driven,
            Present_Price,
            fuel_type_encoded,
            seller_type_encoded,
            transmission_encoded,
            age_val
        ]])

        # Vérifier le prétraitement utilisé pendant l'entraînement
        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == x_new.shape[1]:
            x_new = scaler.transform(x_new)

        y_pred = xgb_model.predict(x_new)[0]

        st.success(
            f"Le prix de vente estimé est de : **{float(y_pred):.2f} k$**"
        )

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
