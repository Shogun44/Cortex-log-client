import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
import joblib
import os
# === Fonctions d’explication et suggestion ===
def expliquer(type_erreur):
    explications = {
        "api": "Erreur sur une route API. Cela peut venir d'une mauvaise URL, méthode ou paramètre.",
        "ldap": "Erreur liée à l’annuaire LDAP (identifiant ou serveur).",
        "authentification": "Les identifiants fournis semblent incorrects.",
        "base_de_donnees": "Problème d’accès ou de requête vers la base PostgreSQL.",
        "réseau": "Le service distant n’a pas répondu ou a mis trop de temps.",
        "autre": "Erreur inconnue ou pas encore catégorisée."
    }
    return explications.get(type_erreur, "Pas d’explication disponible.")

def suggerer_solution(type_erreur):
    suggestions = {
        "api": "Vérifie le endpoint, les paramètres et regarde les logs du backend.",
        "ldap": "Confirme l'identifiant ou contacte un administrateur LDAP.",
        "authentification": "Teste avec un autre utilisateur ou réinitialise le mot de passe.",
        "base_de_donnees": "Teste la connexion à la base, vérifie les credentials ou les requêtes SQL.",
        "réseau": "Vérifie la connexion internet ou le pare-feu.",
        "autre": "Consulte un développeur ou analyse le log complet."
    }
    return suggestions.get(type_erreur, "Pas de suggestion disponible.")

# Titre
st.set_page_config(page_title="Classification des erreurs de logs", layout="wide")
st.title("Détection et apprentissage incrémental des erreurs de logs")

st.markdown("""
Ce mini outil vous permet de :
- Visualiser les erreurs dans les logs sous forme de graphiques
- Entraîner un modèle IA incrémental pour classifier automatiquement les messages d'erreur
- Tester un message d'erreur
- Évaluer un fichier .log brut
""")
# Zone d'Aide (version stylée)
with st.expander("ℹ️ Besoin d'aide pour utiliser l'application ?", expanded=False):
    st.markdown("""
    Bienvenue dans l'outil de **détection intelligente des erreurs** !

    Voici comment utiliser les fonctionnalités :

    - 📥 **Importer un fichier CSV** :  
      Chargez un fichier avec vos messages d'erreur et leur type connu pour entraîner l'intelligence artificielle.

    - 🧠 **Tester un message d'erreur** :  
      Tapez un message libre pour que l'IA devine automatiquement son type d'erreur.

    - 📄 **Analyser un fichier .log ou .txt** :  
      Uploadez un fichier brut de logs pour obtenir une analyse automatique de toutes les lignes.

    - ✍️ **Corriger une prédiction** :  
      Corrigez manuellement si l'IA se trompe sur une erreur, elle apprendra immédiatement de vos corrections !

    - ♻️ **Réinitialiser le modèle** :  
      Si besoin, repartez de zéro en supprimant l'ancien apprentissage.

    ---
    👉 *Pensez à entraîner régulièrement l'IA avec des exemples pour la rendre plus intelligente !*
    """)

# Réinitialisation du modèle
st.sidebar.header("Options")
if st.sidebar.button("🔄 Réinitialiser le modèle"):
    for file in ["modele_incremental.pkl", "vectorizer.pkl", "label_encoder.pkl"]:
        if os.path.exists(file):
            os.remove(file)
    st.sidebar.success("Modèle réinitialisé avec succès.")

# Chargement CSV
uploaded_file = st.file_uploader("Importer un fichier CSV avec colonnes `message` et `type_erreur`", type="csv")

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
            vectorizer = CountVectorizer()
            X_vectorized = vectorizer.fit_transform(X)

            class_counts = pd.Series(y_encoded).value_counts()
            if (class_counts < 2).any():
                st.error("Chaque classe doit avoir au moins 2 exemples.")
            else:
                model = SGDClassifier()
                model.partial_fit(X_vectorized, y_encoded, classes=list(set(y_encoded)))

                # Sauvegarde
                joblib.dump(model, "modele_incremental.pkl")
                joblib.dump(vectorizer, "vectorizer.pkl")
                joblib.dump(label_encoder, "label_encoder.pkl")
                model_loaded = True

        if model_loaded:
            X_vectorized = vectorizer.transform(X)
            y_pred = model.predict(X_vectorized)

            labels = unique_labels(y_encoded, y_pred)
            report = classification_report(
                y_encoded, y_pred,
                labels=labels,
                target_names=label_encoder.inverse_transform(labels),
                output_dict=True
            )
            st.subheader("Rapport de classification")
            st.dataframe(pd.DataFrame(report).transpose())

            # Test manuel
            st.subheader("Tester une prédiction manuelle")
            user_input = st.text_area("Tape un message d’erreur pour prédiction")
            if user_input:
                X_test = vectorizer.transform([user_input])
                y_pred = model.predict(X_test)
                pred_label = label_encoder.inverse_transform(y_pred)[0]

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
                        new_vector = vectorizer.transform([new_message])
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

# Évaluation fichier .log brut
st.subheader("Analyser un fichier de logs brut (.log / .txt)")
log_file = st.file_uploader("Importer un fichier .log ou .txt", type=["log", "txt"], key="log_file")

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
            X_log = vectorizer.transform(log_lines)
            preds = model.predict(X_log)
            labels = label_encoder.inverse_transform(preds)

            log_df = pd.DataFrame({
                "Ligne du log": log_lines,
                "Type d'erreur prédit": labels
            })

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
                X_correction = vectorizer.transform([ligne_selectionnee])
                y_correction = label_encoder.transform([type_correction])
                model.partial_fit(X_correction, y_correction)
                joblib.dump(model, "modele_incremental.pkl")
                st.success(f"Correction enregistrée : le modèle a appris que cette ligne est une erreur **{type_correction}**")


            csv = log_df.to_csv(index=False).encode("utf-8")
            st.download_button("Télécharger les prédictions (CSV)", csv, file_name="logs_analyzes.csv")
    except Exception as e:
        st.error(f"Erreur : {e}")