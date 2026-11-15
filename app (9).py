
import streamlit as st
import joblib as jb
import pandas as pd
import numpy as np
from pathlib import Path

# CONFIGURATION
st.set_page_config(
    page_title="Prédiction du prix d'une voiture",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Prédiction du prix de vente des véhicules")
st.write("Application utilisant un modèle XGBoost.")

# CHARGEMENT DES MODELES
st.write("Dossier utilisé :", str(BASE_DIR))
st.write("Fichiers trouvés :", [f.name for f in BASE_DIR.iterdir()])

@st.cache_resource
def load_models():
    encoders = jb.load(BASE_DIR / "encoders.joblib")
    scaler = jb.load(BASE_DIR / "scaler.joblib")
    uniques = jb.load(BASE_DIR / "uniques.joblib")
    xgb_model = jb.load(BASE_DIR / "xgb_model.joblib")
    return xgb_model, scaler, encoders, uniques

try:
    xgb_model, scaler, encoders, uniques = load_models()
except Exception as e:
    st.error(f"Erreur lors du chargement des modèles : {e}")
    st.stop()


# PREPARATION DES DONNEES
def prepare_data(df):
    df = df.copy()

    df["Fuel_Type"] = encoders[0].transform(df["Fuel_Type"])
    df["Seller_Type"] = encoders[1].transform(df["Seller_Type"])
    df["Transmission"] = encoders[2].transform(df["Transmission"])

    columns = [
        "Kms_Driven",
        "Present_Price",
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "age"
    ]

    X = df[columns].to_numpy()

    if scaler.n_features_in_ == X.shape[1]:
        X = scaler.transform(X)

    return X


def predict_prices(df):
    X = prepare_data(df)
    return xgb_model.predict(X)


# MENU LATERAL
choix = st.sidebar.radio(
    "Choisir le type de prédiction",
    ["Prédiction simple", "Prédiction multiple"]
)


# =====================================
# PREDICTION SIMPLE
# =====================================
if choix == "Prédiction simple":

    st.header("🚘 Prédiction pour une voiture")

    with st.form("formulaire_simple"):

        kms = st.number_input("Kilométrage", min_value=0, value=50000)
        prix = st.number_input("Prix d'origine (k$)", min_value=0.0, value=10.0)

        carburant = st.selectbox("Type de carburant", uniques[0])
        vendeur = st.selectbox("Type de vendeur", uniques[1])
        transmission = st.selectbox("Boîte de vitesse", uniques[2])

        age = st.number_input("Âge du véhicule", min_value=0, value=5)

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

    st.header("📊 Prédiction pour plusieurs voitures")

    st.write("Importez un fichier CSV contenant plusieurs véhicules.")

    st.info("""
    Le fichier CSV doit contenir les colonnes suivantes :
    Kms_Driven, Present_Price, Fuel_Type, Seller_Type,
    Transmission et age.
    """)

    fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if fichier is not None:

        try:
            donnees = pd.read_csv(fichier)

            st.subheader("Aperçu des données")
            st.dataframe(donnees.head())

            colonnes_requises = [
                "Kms_Driven",
                "Present_Price",
                "Fuel_Type",
                "Seller_Type",
                "Transmission",
                "age"
            ]

            manquantes = [
                col for col in colonnes_requises
                if col not in donnees.columns
            ]

            if manquantes:
                st.error(f"Colonnes manquantes : {manquantes}")

            elif st.button("Lancer les prédictions"):

                resultats = donnees.copy()
                resultats["Prix_estime"] = predict_prices(donnees)

                st.subheader("Résultats")
                st.dataframe(resultats)

                csv = resultats.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Télécharger les résultats",
                    data=csv,
                    file_name="predictions_voitures.csv",
                    mime="text/csv"
                )

                st.success(
                    f"{len(resultats)} véhicules traités avec succès."
                )

        except Exception as e:
            st.error(f"Erreur : {e}")







    

        
        

   
