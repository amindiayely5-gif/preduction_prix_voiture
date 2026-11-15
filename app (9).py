
import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# =====================================
# CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Prédiction du prix d'une voiture",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Prédiction du prix de vente des véhicules")
st.write("Application utilisant un modèle XGBoost.")

BASE_DIR = Path(__file__).resolve().parent


# =====================================
# RECHERCHE DES FICHIERS
# =====================================

def find_file(filename):
    fichiers = list(BASE_DIR.rglob(filename))

    if not fichiers:
        raise FileNotFoundError(
            f"{filename} introuvable dans {BASE_DIR}"
        )

    return fichiers[0]


# =====================================
# CHARGEMENT DES MODELES
# =====================================

@st.cache_resource
def load_models():

    encoders = joblib.load(find_file("encoders.joblib"))
    scaler = joblib.load(find_file("scaler.joblib"))
    uniques = joblib.load(find_file("uniques.joblib"))
    xgb_model = joblib.load(find_file("xgb_model.joblib"))

    return xgb_model, encoders, scaler, uniques


try:
    xgb_model, encoders, scaler, uniques = load_models()

except Exception as e:
    st.error(f"Erreur lors du chargement des modèles : {e}")
    st.write("Dossier de l'application :", str(BASE_DIR))
    st.write(
        "Fichiers présents :",
        [f.name for f in BASE_DIR.rglob("*") if f.is_file()]
    )
    st.stop()


# =====================================
# ENCODAGE ET PREDICTION
# =====================================

def predict_prices(df):

    df = df.copy()

    colonnes = [
        "Kms_Driven",
        "Present_Price",
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "age"
    ]

    for col in colonnes:
        if col not in df.columns:
            raise ValueError(f"Colonne manquante : {col}")

    # Encodage des variables catégorielles
    df["Fuel_Type"] = encoders[0].transform(
        df["Fuel_Type"].astype(str).str.strip()
    )

    df["Seller_Type"] = encoders[1].transform(
        df["Seller_Type"].astype(str).str.strip()
    )

    df["Transmission"] = encoders[2].transform(
        df["Transmission"].astype(str).str.strip()
    )

    # Respecter exactement l'ordre des variables
    X = df[colonnes].to_numpy(dtype=float)

    # Prédiction
    predictions = xgb_model.predict(X)

    return predictions


# =====================================
# MENU LATERAL
# =====================================

choix = st.sidebar.radio(
    "Choisir le type de prédiction",
    ["Prédiction simple", "Prédiction multiple"]
)


# =====================================
# PREDICTION SIMPLE
# =====================================

if choix == "Prédiction simple":

    st.header("🚘 Estimer le prix d'une voiture")

    with st.form("formulaire_simple"):

        kms = st.number_input(
            "Kilométrage",
            min_value=0,
            value=50000
        )

        prix = st.number_input(
            "Prix d'origine (k$)",
            min_value=0.0,
            value=10.0
        )

        carburant = st.selectbox(
            "Type de carburant",
            uniques[0]
        )

        vendeur = st.selectbox(
            "Type de vendeur",
            uniques[1]
        )

        transmission = st.selectbox(
            "Boîte de vitesse",
            uniques[2]
        )

        age = st.number_input(
            "Âge du véhicule",
            min_value=0,
            value=5
        )

        bouton = st.form_submit_button("Prédire le prix")

    if bouton:

        try:
            voiture = pd.DataFrame([{
                "Kms_Driven": kms,
                "Present_Price": prix,
                "Fuel_Type": carburant,
                "Seller_Type": vendeur,
                "Transmission": transmission,
                "age": age
            }])

            prediction = predict_prices(voiture)

            st.success(
                f"Prix de vente estimé : {float(prediction[0]):.2f} k$"
            )

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")


# =====================================
# PREDICTION MULTIPLE
# =====================================

elif choix == "Prédiction multiple":

    st.header("📊 Estimer le prix de plusieurs voitures")

    st.write(
        "Importez un fichier CSV contenant les informations "
        "des véhicules."
    )

    st.info("""
    Le fichier CSV doit contenir exactement les colonnes suivantes :

    - Kms_Driven
    - Present_Price
    - Fuel_Type
    - Seller_Type
    - Transmission
    - age
    """)

    fichier = st.file_uploader(
        "Importer un fichier CSV",
        type=["csv"]
    )

    if fichier is not None:

        try:
            donnees = pd.read_csv(fichier)

            st.subheader("Aperçu des données")
            st.dataframe(donnees.head())

            if st.button("Lancer les prédictions"):

                resultats = donnees.copy()

                predictions = predict_prices(donnees)

                resultats["Prix_estime"] = predictions

                st.subheader("Résultats des prédictions")
                st.dataframe(resultats)

                csv = resultats.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="Télécharger les résultats CSV",
                    data=csv,
                    file_name="predictions_voitures.csv",
                    mime="text/csv"
                )

                st.success(
                    f"{len(resultats)} véhicules traités avec succès."
                )

        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")
