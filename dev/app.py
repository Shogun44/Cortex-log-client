import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
import joblib
import os
import re
import spacy
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from datetime import datetime
from semantic_model import cluster_erreurs
from core.preprocessing import nettoyer_message
from core.explication import  expliquer, suggerer_solution
from core.historique import ajouter_erreur_dans_historique

# Logo centré
# Configuration de la page
st.set_page_config(page_title="CortexLog",
    page_icon="🧠",
    layout="wide")  

st.markdown("""
<div style="position: absolute; top: 10px; left: 10px;">
    <a href="?reset=1"><img src="Reload.png" width="40"></a>
</div>

<div style="position: absolute; bottom: 10px; left: 10px;">
    <a href="?historique=1"><img src="Magnifying-glass.png" width="40"></a>
</div>

<div style="position: absolute; bottom: 10px; right: 10px;">
    <a href="?aide=1"><img src="Information-button.png" width="40"></a>
</div>
""", unsafe_allow_html=True)

from urllib.parse import parse_qs
query_params = st.experimental_get_query_params()

if "reset" in query_params:
    for file in ["modele_incremental.pkl", "vectorizer.pkl", "label_encoder.pkl"]:
        if os.path.exists(file):
            os.remove(file)
    st.success("Modèle réinitialisé avec succès.")

# Appliquer un fond personnalisé et du CSS global
def inject_css():
    st.markdown("""
        <style>
        /* Masquer header/footer */
        header, footer {visibility: hidden;}

        /* Fond avec image */
        .stApp {
            background-image: url("350.png");
            background-size: cover;
            background-position: center;
        }

        /* Logo centré */
        #logo-container {
            display: flex;
            justify-content: center;
            margin-top: -30px;
            margin-bottom: 20px;
        }

        #logo-container img {
            width: 100px;
        }

        /* Position des boutons */
        #top-left, #bottom-left, #bottom-right {
            position: fixed;
            z-index: 9999;
        }

        #top-left {
            top: 15px;
            left: 15px;
        }

        #bottom-left {
            bottom: 15px;
            left: 15px;
        }

        #bottom-right {
            bottom: 15px;
            right: 15px;
        }

        /* Style des blocs centraux */
        .bloc-action {
            background: transparent;
            border: 2px solid red;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }

        .bloc-img {
            width: 80px;
            margin-bottom: 10px;
        }

        </style>
    """, unsafe_allow_html=True)

inject_css()


# Logo centré
st.markdown('<div id="logo-container"><img src="logo222.png"></div>', unsafe_allow_html=True)

# Bouton réinitialisation en haut à gauche
st.markdown(f'''
<div id="top-left">
    <a href="?reset=1"><img src="Reload.png" width="40"></a>
</div>
''', unsafe_allow_html=True)

# Bloc principal central : CSV et Log
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="bloc-action">', unsafe_allow_html=True)
    st.image("Document.png", width=60)
    csv_file = st.file_uploader("Importer un fichier CSV avec colonnes `message` et `type_erreur`", type="csv", label_visibility="collapsed", key="csv_import")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="bloc-action">', unsafe_allow_html=True)
    st.image("Download.png", width=60)
    log_file = st.file_uploader("Importer un fichier .log ou .txt", type=["log", "txt"], label_visibility="collapsed", key="log_import")
    st.markdown('</div>', unsafe_allow_html=True)

# Bouton loupe (historique) en bas à gauche
st.markdown(f'''
<div id="bottom-left">
    <a href="#historique"><img src="Magnifying-glass.png" width="40"></a>
</div>
''', unsafe_allow_html=True)

# Bouton aide (point d’interrogation) en bas à droite
st.markdown(f'''
<div id="bottom-right">
    <a href="#aide"><img src="Information-button.png" width="40"></a>
</div>
''', unsafe_allow_html=True)

# Fichier historique
HISTORIQUE_PATH = "historique_erreurs.csv"

# Création du fichier si inexistant
if not os.path.exists(HISTORIQUE_PATH):
    with open(HISTORIQUE_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["horodatage", "message", "type_prédit", "origine", "modèle"])




# Zone d'Aide (version stylée)
if "aide" in query_params:

    st.markdown("""
    Voici comment utiliser CortexLog :

    - 📥 **Importer un CSV** : pour entraîner l’IA à partir de vos messages d’erreur connus.
    - 📄 **Analyser un fichier .log brut** : pour classer automatiquement chaque ligne.
    - 🔄 **Réinitialiser le modèle** : pour repartir de zéro.
    - 🔍 **Historique** : suivez toutes les erreurs analysées et corrigées.
    - ✍️ **Corriger une prédiction** :  
      Corrigez manuellement si l'IA se trompe sur une erreur, elle apprendra immédiatement de vos corrections !


    ---
    Pensez à entraîner régulièrement l’IA avec vos exemples pour l’améliorer.
    """)

# Réinitialisation du modèle
col1, col2 = st.columns([0.08, 0.92])
with col1:
    if st.button("", key="reset_btn"):
        for file in ["modele_incremental.pkl", "vectorizer.pkl", "label_encoder.pkl"]:
            if os.path.exists(file):
                os.remove(file)
        st.success("Modèle réinitialisé avec succès.")
    st.image("asset/Reload.png", width=32)

st.markdown('<div class="custom-box">', unsafe_allow_html=True)
# Chargement CSV
col_csv1, col_csv2 = st.columns([0.08, 0.92])
with col_csv1:
    st.image("asset/histo.png", width=32)
with col_csv2:
    uploaded_file = st.file_uploader("Importer un fichier CSV avec colonnes `message` et `type_erreur`", type="csv")
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Aperçu des données importées")
    st.write(df.head())

    if 'message' in df.columns and 'type_erreur' in df.columns:
        st.subheader("Répartition des erreurs")
        st.bar_chart(df['type_erreur'].value_counts())

        X = df['message']
        y = df['type_erreur']

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        model_loaded = False
        if all(os.path.exists(f) for f in ["modele_incremental.pkl", "vectorizer.pkl", "label_encoder.pkl"]):
            model = joblib.load("modele_incremental.pkl")
            vectorizer = joblib.load("vectorizer.pkl")
            label_encoder = joblib.load("label_encoder.pkl")
            model_loaded = True
        else:
            X_nettoye = X.apply(nettoyer_message)
            vectorizer = TfidfVectorizer()
            X_vectorized = vectorizer.fit_transform(X_nettoye)
           
            class_counts = pd.Series(y_encoded).value_counts()
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_encoded, test_size=0.2, random_state=42)
            if (class_counts < 2).any():
                st.error("Chaque classe doit avoir au moins 2 exemples.")
            else:
                model = SGDClassifier()
                model.partial_fit(X_train, y_train, classes=list(set(y_encoded)))

            # Test du modèle Random Forest (entraînement classique)
                rf_model = RandomForestClassifier()
                rf_model.fit(X_train, y_train)

                # Évaluation
                y_pred_rf = rf_model.predict(X_test)
                report_rf = classification_report(
                    y_test,
                    y_pred_rf,
                    labels=unique_labels(y_test, y_pred_rf),
                    target_names=label_encoder.inverse_transform(unique_labels(y_test, y_pred_rf)),
                    output_dict=True
                )

                # Affichage des performances
                st.subheader("Rapport de classification - Random Forest")
                st.dataframe(pd.DataFrame(report_rf).transpose())

                # Sauvegarde du modèle
                joblib.dump(rf_model, "models/modele_rf.pkl")
                # Entraînement du modèle SVM
                svm_model = LinearSVC()
                svm_model.fit(X_train, y_train)
                joblib.dump(svm_model, "models/modele_svm.pkl")

                # Sauvegarde
                joblib.dump(model, "models/modele_incremental.pkl")
                joblib.dump(vectorizer, "models/vectorizer.pkl")
                joblib.dump(label_encoder, "models/label_encoder.pkl")
                model_loaded = True
                

        if model_loaded:
            # Évaluation des performances uniquement sur les données de test
            st.subheader("Évaluation sur les données de test")
            X_nettoye = X.apply(nettoyer_message)
            X_vectorized = vectorizer.transform(X_nettoye)
            X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_encoded, test_size=0.2, random_state=42)

            # SGDClassifier (incrémental)
            y_pred_sgd = model.predict(X_test)
            report_sgd = classification_report(
            y_test,
            y_pred_sgd,
            labels=unique_labels(y_test, y_pred_sgd),
            target_names=label_encoder.inverse_transform(unique_labels(y_test, y_pred_sgd)),
            output_dict=True
            )
            st.subheader("Rapport de classification - SGDClassifier")
            st.dataframe(pd.DataFrame(report_sgd).transpose())

            # Random Forest
            rf_model = joblib.load("modele_rf.pkl")
            y_pred_rf = rf_model.predict(X_test)
            report_rf = classification_report(
            y_test,
            y_pred_rf,
            labels=unique_labels(y_test, y_pred_rf),
            target_names=label_encoder.inverse_transform(unique_labels(y_test, y_pred_rf)),
            output_dict=True
            )
            st.subheader("Rapport de classification - Random Forest")
            st.dataframe(pd.DataFrame(report_rf).transpose())

            # SVM
            svm_model = joblib.load("modele_svm.pkl")
            y_pred_svm = svm_model.predict(X_test)
            report_svm = classification_report(
            y_test,
            y_pred_svm,
            labels=unique_labels(y_test, y_pred_svm),
            target_names=label_encoder.inverse_transform(unique_labels(y_test, y_pred_svm)),
            output_dict=True
            )
            st.subheader("Rapport de classification - SVM")
            st.dataframe(pd.DataFrame(report_svm).transpose())

            st.subheader("Comparaison des modèles")

            cols = st.columns(3)

            # === Bloc 1 : SGD ===
            with cols[0]:
                st.markdown("### SGDClassifier")
                st.dataframe(pd.DataFrame(report_sgd).transpose())

            # === Bloc 2 : Random Forest ===
            with cols[1]:
                if report_rf:
                    st.markdown("### Random Forest")
                    st.dataframe(pd.DataFrame(report_rf).transpose())
                else:
                    st.warning("Modèle Random Forest non trouvé.")

            # === Bloc 3 : SVM ===
            with cols[2]:
                if report_svm:
                    st.markdown("### SVM (LinearSVC)")
                    st.dataframe(pd.DataFrame(report_svm).transpose())
                else:
                    st.warning("Modèle SVM non trouvé.")
            # Comparaison automatique des performances moyennes (f1-score)
            scores_moyens = {}

            # Extraire le f1-score global de chaque modèle s’il existe
            try:
                    scores_moyens["SGDClassifier"] = report_sgd["weighted avg"]["f1-score"]
            except:
                pass

            if report_rf:
                try:
                        scores_moyens["Random Forest"] = report_rf["weighted avg"]["f1-score"]
                except:
                    pass

            if report_svm:
                try:
                    scores_moyens["SVM"] = report_svm["weighted avg"]["f1-score"]
                except:
                    pass

            # Afficher le meilleur
            if scores_moyens:
                meilleur_modele = max(scores_moyens, key=scores_moyens.get)
                st.success(f"**Modèle le plus performant actuellement : {meilleur_modele}** (F1-score : {scores_moyens[meilleur_modele]:.2f})")
            else:
                st.info("Aucun modèle complet disponible pour comparaison.")
            # Test manuel
            st.subheader("Tester une prédiction manuelle")
            user_input = st.text_area("Tape un message d’erreur pour prédiction")
            if user_input:
                X_test = vectorizer.transform([nettoyer_message(user_input)])
                y_pred = model.predict(X_test)
                pred_label = label_encoder.inverse_transform(y_pred)[0]
                ajouter_erreur_dans_historique(user_input, pred_label, origine="manuel", modele=meilleur_modele)
                st.success(f"Type d'erreur prédit  : **{pred_label}**")
                # Ajout d’une explication et d’une solution avec style
                explication = expliquer(pred_label)
                suggestion = suggerer_solution(pred_label)

                st.info(f"**Explication :** {explication}")
                st.warning(f"**Suggestion :** {suggestion}")

            # Ajout d'un exemple manuel
            st.subheader("Apprentissage d’un nouveau message")
            new_message = st.text_input("Message")
            new_label = st.text_input("Type d'erreur associé")

            if st.button("Apprendre ce nouvel exemple"):
                if new_message and new_label:
                    try:
                        new_vector = vectorizer.transform([nettoyer_message(new_message)])
                        new_encoded = label_encoder.transform([new_label])
                        model.partial_fit(new_vector, new_encoded)
                        joblib.dump(model, "modele_incremental.pkl")
                        st.success("Le modèle a appris ce nouveau message.")
                    except:
                        st.warning("Cette classe est inconnue. Merci d’ajouter cette classe via un nouveau CSV et réentraîner.")
                else:
                    st.warning("Remplis les deux champs.")
    else:
        st.error("Le fichier doit contenir les colonnes `message` et `type_erreur`.")
else:
    st.info("Veuillez importer un fichier CSV pour commencer.")

st.markdown('<div class="custom-box">', unsafe_allow_html=True)
# Évaluation fichier .log brut
st.subheader("Analyser un fichier de logs brut (.log / .txt)")
col_log1, col_log2 = st.columns([0.08, 0.92])
with col_log1:
    st.image("asset/Document.png", width=32)
with col_log2:
    log_file = st.file_uploader("Importer un fichier .log ou .txt", type=["log","txt"], key="log_file")
    st.markdown('</div>', unsafe_allow_html=True)



if log_file and os.path.exists("modele_incremental.pkl"):
    try:
        model = joblib.load("modele_incremental.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        label_encoder = joblib.load("label_encoder.pkl")

        # Lecture lignes
        log_lines = log_file.read().decode("utf-8").splitlines()
        log_lines = [line for line in log_lines if line.strip() != ""]

        if len(log_lines) == 0:
            st.warning("Fichier vide.")
        else:
            X_log = vectorizer.transform([nettoyer_message(line) for line in log_lines])
            preds = model.predict(X_log)
            labels = label_encoder.inverse_transform(preds)

            log_df = pd.DataFrame({
                "Ligne du log": log_lines,
                "Type d'erreur prédit": labels
            })
            meilleur_modele = "SVM"
            for msg, typ in zip(log_lines, labels):
                ajouter_erreur_dans_historique(msg, typ, origine="log", modele=meilleur_modele)
            def color_ligne(row):
                couleur = {
                    "api": "background-color: #FFD700",
                    "ldap": "background-color: #ADD8E6",
                    "authentification": "background-color: #FFB6C1",
                    "base_de_donnees": "background-color: #90EE90",
                }
                return [couleur.get(row["Type d'erreur prédit"], "")] * 2
            st.write("### Résultat de l’analyse avec surlignage par type d'erreur")
            # Liste des types d’erreurs prédits
            types_uniques = sorted(log_df["Type d'erreur prédit"].unique())
            # Filtre multi-sélection
            filtre = st.multiselect("Filtrer par type d'erreur :", types_uniques, default=types_uniques)
            # Filtrage du tableau
            log_df_filtré = log_df[log_df["Type d'erreur prédit"].isin(filtre)]
            # Compteur dynamique
            st.markdown("#### Résumé des types sélectionnés :")
            for type_ in filtre:
                nb = (log_df_filtré["Type d'erreur prédit"] == type_).sum()
                st.markdown(f"- **{type_}** : {nb} logs")
            # Affichage stylé
            st.dataframe(log_df_filtré.style.apply(color_ligne, axis=1))
            st.subheader("Corriger manuellement une ligne analysée")

            # Sélection de la ligne à corriger
            ligne_selectionnee = st.selectbox("Choisir une ligne de log :", log_df_filtré["Ligne du log"].tolist())

            # Prédiction actuelle
            prediction_actuelle = log_df_filtré[log_df_filtré["Ligne du log"] == ligne_selectionnee]["Type d'erreur prédit"].values[0]
            st.markdown(f"**Prédiction actuelle** : `{prediction_actuelle}`")

            # Sélection de la bonne classe
            type_correction = st.selectbox("Sélectionner le bon type :", label_encoder.classes_, key="correction_log")

            if st.button("Corriger et réentraîner avec cette ligne"):
                X_correction = vectorizer.transform([nettoyer_message(ligne_selectionnee)])
                y_correction = label_encoder.transform([type_correction])
                model.partial_fit(X_correction, y_correction)
                joblib.dump(model, "modele_incremental.pkl")
                st.success(f"Correction enregistrée : le modèle a appris que cette ligne est une erreur **{type_correction}**")
                X_log = vectorizer.transform([nettoyer_message(line) for line in log_lines])
                preds = model.predict(X_log)
                labels = label_encoder.inverse_transform(preds)
                log_df["Type d'erreur prédit"] = labels 
                log_df_filtré = log_df[log_df["Type d'erreur prédit"].isin(filtre)]
                st.rerun()



            csv = log_df.to_csv(index=False).encode("utf-8")
            col_dl1, col_dl2 = st.columns([0.08, 0.92])
            with col_dl1:
                st.image("asset/Download.png", width=32)
            with col_dl2:
                st.download_button("Télécharger les prédictions (CSV)", csv, file_name="logs_analyzes.csv")

    except Exception as e:
        st.error(f"Erreur : {e}")
# === Historique des erreurs analysées ===
if "show_history" in st.session_state and st.session_state["show_history"]:
    st.subheader("Historique des erreurs analysées")
   
import csv

HISTORIQUE_PATH = "historique_erreurs.csv"

# Vérifie si le fichier existe
if os.path.exists(HISTORIQUE_PATH):
    historique_df = pd.read_csv(HISTORIQUE_PATH)
    
    

    # Option de filtre par type ou origine
    with st.expander("🔍 Filtrer l'historique", expanded=False):
        types_disponibles = historique_df["type_prédit"].dropna().unique().tolist()
        origines_disponibles = historique_df["origine"].dropna().unique().tolist()

        filtre_type = st.multiselect("Filtrer par type :", types_disponibles, default=types_disponibles)
        filtre_origine = st.multiselect("Filtrer par origine :", origines_disponibles, default=origines_disponibles)

        filtre_df = historique_df[
            (historique_df["type_prédit"].isin(filtre_type)) &
            (historique_df["origine"].isin(filtre_origine))
        ]

        if not filtre_df.empty:
            st.dataframe(filtre_df)
        else:
            st.warning("aucune erreur correspond au filtre")    

    # Export CSV
    csv_export = historique_df.to_csv(index=False).encode("utf-8")
    col_hist1, col_hist2 = st.columns([0.08, 0.92])
    with col_hist1:
        st.image("asset/Download.png", width=32)
    with col_hist2:
        st.download_button("Télécharger l'historique complet", csv_export, file_name="historique_erreurs.csv")
    # Option : Vider l’historique
    if st.button("🗑️ Vider l’historique"):
        os.remove(HISTORIQUE_PATH)
        st.success("Historique vidé.")
else:
    st.info("Aucune erreur historisée pour le moment.")
# Bouton Historique (loupe)


    st.markdown("---")
    if st.button("Historique", key="history_btn"):
        st.session_state["show_history"] = True
    st.image("asset/Magnifying-glass.png", width=32)
# Bouton Aide (?)

if st.button("Aide", key="help_btn"):
    st.session_state["show_help"] = True
st.image("asset/Information-button.png", width=32)

# === Regroupement sémantique et rapport intelligent ===
st.subheader("🔎 Regroupement sémantique et rapport d’erreurs intelligent")

from semantic_model import cluster_erreurs, generer_rapport_clusters, generer_pdf
import io

input_text = st.text_area("Colle ici les erreurs à regrouper (une par ligne)")

if st.button("Analyser et générer rapport"):
    if input_text.strip():
        messages = [line.strip() for line in input_text.splitlines() if line.strip()]
        clusters = cluster_erreurs(messages, n_clusters=3)

        st.success(f"{len(messages)} erreurs regroupées en {len(clusters)} clusters.")
        for cluster_id, group in clusters.items():
            st.markdown(f"### 🔹 Groupe {cluster_id + 1}")
            for msg in group:
                st.markdown(f"- {msg}")

        # Génération du rapport
        rapport_txt = generer_rapport_clusters(clusters)

        # Aperçu texte
        st.markdown("### 📄 Aperçu du rapport")
        st.text(rapport_txt)

        # Export TXT
        buffer_txt = io.StringIO()
        buffer_txt.write(rapport_txt)
        buffer_txt.seek(0)

        st.download_button(
            label="⬇️ Télécharger le rapport (.txt)",
            data=buffer_txt,
            file_name="rapport_erreurs.txt",
            mime="text/plain"
        )

        # Export PDF
        pdf = generer_pdf(rapport_txt)
        pdf_output = io.BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)

        st.download_button(
            label="⬇️ Télécharger le rapport (.pdf)",
            data=pdf_output,
            file_name="rapport_erreurs.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("Aucun message détecté.")

        