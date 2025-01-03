
import streamlit as st
import pandas as pd
import pickle
import shap
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# === 1) CHARGEMENT DU MODÈLE ET DES DONNÉES ===
model = pickle.load(open('mlflow_model/model.pkl', 'rb'))
data = pd.read_csv('test_df.csv')
data_train = pd.read_csv('train_df.csv')

#  Prétraitement / scaling
data_scaled = data.copy()
data_train_scaled = data_train.copy()


# Création de l'explainer SHAP (modèle type arbre)
explainer = shap.TreeExplainer(model['classifier'])

# === 2) SIDEBAR POUR CHOISIR LE CLIENT ===
st.sidebar.header("Client ID Selection")
client_id = st.sidebar.selectbox("Choose a Client ID:", data['SK_ID_CURR'])

# === 3) TITRE ET INTRODUCTION ===
st.title(" Dashboard crédit d'un client")
st.write("Ce dashboard affiche la prédiction et l'explication SHAP pour un client donné.")

# === 4) VÉRIFICATION DE L'EXISTENCE DU CLIENT ===
if client_id:
    if client_id not in list(data['SK_ID_CURR']):
        st.error("Client ID not found in the database.")
    else:
        # === 4.1) INFORMATIONS DU CLIENT ===
        st.subheader("Client Information")
        client_data = data[data['SK_ID_CURR'] == client_id]
        st.write(client_data)

        # === 4.2) PRÉDICTION DU RISQUE DE DÉFAUT ===
        st.subheader("Default Probability Prediction")
        info_client = client_data.drop('SK_ID_CURR', axis=1)
        prediction = model.predict_proba(info_client)[0][1]
        st.write(f"Default Probability: {prediction:.3f}")

        # Décision selon un seuil
        threshold = 0.5
        decision = "Approved" if prediction < threshold else "Rejected"
        decision_color = "green" if decision == "Approved" else "red"
        st.markdown(f"<h3 style='color:{decision_color};'>Loan Decision: {decision}</h3>", unsafe_allow_html=True)

# === Création de la jauge ===
        fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number",
    value=prediction,
    domain={'x': [0, 1], 'y': [0, 1]},
    title={'text': "Default Probability"},
    gauge={
        'axis': {'range': [0, 1]},  # La plage de la jauge va de 0 à 1
        'bar': {'color': "black"},  # Couleur du curseur
        'steps': [
            {'range': [0, 0.5], 'color': "green"},  # Zone verte
            {'range': [0.5, 1], 'color': "red"}    # Zone rouge
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': prediction  # Position du curseur
           }
        }
      ))

       # Affichage de la jauge dans Streamlit
        st.plotly_chart(fig_gauge)


        # === 5) EXPLICATION SHAP GLOBALE ===
        st.subheader("SHAP Global Explanation")
        # Calcul des SHAP values globales
        shap_vals_global = explainer.shap_values(data_scaled.drop('SK_ID_CURR', axis=1))

        # Graphique en barres (summary_plot global)
        fig_global, ax_global = plt.subplots()
        shap.summary_plot(
            shap_vals_global,
            data_scaled.drop('SK_ID_CURR', axis=1),
            plot_type='bar',
            show=False
        )
        st.pyplot(fig_global)

      # === 5) EXPLICATION SHAP GLOBALE ===
        st.subheader("SHAP Global Explanation")

       # 1) On calcule les shap values
        shap_vals_global = explainer.shap_values(data_scaled.drop('SK_ID_CURR', axis=1))

       # 2) On sélectionne la bonne classe 
       #    Si shap_vals_global est une liste pour la classe 0 et 1 :
        if isinstance(shap_vals_global, list) and len(shap_vals_global) == 2:
          shap_vals_global_class1 = shap_vals_global[1]
        else:
          shap_vals_global_class1 = shap_vals_global  # tableau direct

         # 3) On construit l'objet Explanation
          X_global = data_scaled.drop('SK_ID_CURR', axis=1)
          shap_values_exp = shap.Explanation(
           values=shap_vals_global_class1,
           data=X_global,
           feature_names=X_global.columns
)

            # 4) On plot la beeswarm
        fig_global, ax_global = plt.subplots()
        shap.plots.beeswarm(shap_values_exp, max_display=20, show=False)
        st.pyplot(fig_global)

        # === 6) EXPLICATION SHAP LOCALE ===
        st.subheader("SHAP Local Explanation")

        # On retire la colonne SK_ID_CURR pour l'inférence SHAP
        X_client = client_data.drop('SK_ID_CURR', axis=1)
        

        # Calcul des SHAP values pour cette unique observation
        shap_values_local = explainer(X_client)  # Peut renvoyer un shap.Explanation ou une liste

        # -- Gérer la forme retournée par SHAP --
        # 1) Si c'est une liste (ex: shap_values_local[0] pour la classe 0, shap_values_local[1] pour la classe 1)
        if isinstance(shap_values_local, list):
            # Pour un modèle binaire, on prend souvent la classe 1
            shap_for_class1 = shap_values_local[1]
            # Comme X_client = 1 ligne, shap_for_class1.shape = (1, n_features)
            local_explanation = shap_for_class1[0]  # vecteur SHAP pour le client
        else:
            # Sinon, c'est un unique tableau (ou shap.Explanation) pour la classe positive
            # de forme (1, nb_features)
            local_explanation = shap_values_local[0]

        # Affichage du waterfall plot
        fig_local = plt.figure()
        shap.waterfall_plot(local_explanation, show=False)
        st.pyplot(fig_local)



        
       
