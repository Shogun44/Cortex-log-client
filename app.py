import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels
import joblib
import os

# Titre
st.set_page_config(page_title="Classification des erreurs de logs", layout="wide")
st.title("Détection et classification automatique des erreurs de logs")

st.markdown("""
Ce mini outil vous permet de :
- Visualiser les erreurs dans les logs sous forme de graphiques
- Entraîner un modèle IA pour classifier automatiquement les messages d'erreur
- Tester la classification d’un message d'erreur en direct
""")

# Sidebar – bouton de réinitialisation
st.sidebar.header("Options")
if st.sidebar.button("🔄 Réinitialiser le modèle"):
    for file in ["modele_classification.pkl", "vectorizer.pkl", "label_encoder.pkl"]:
        if os.path.exists(file):
            os.remove(file)
    st.sidebar.success("Modèle supprimé avec succès. Il sera réentraîné au prochain chargement.")

# Upload CSV
uploaded_file = st.file_uploader("Importer un fichier CSV avec colonnes `message` et `type_erreur`", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Aperçu des données importées")
    st.write(df.head())

    if 'message' in df.columns and 'type_erreur' in df.columns:

        st.subheader("Répartition des erreurs (type_erreur)")
        error_counts = df['type_erreur'].value_counts()
        st.bar_chart(error_counts)

        X = df['message']
        y = df['type_erreur']

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        model_loaded = False
        if all(os.path.exists(f) for f in ["modele_classification.pkl", "vectorizer.pkl", "label_encoder.pkl"]):
            model = joblib.load("modele_classification.pkl")
            vectorizer = joblib.load("vectorizer.pkl")
            label_encoder = joblib.load("label_encoder.pkl")
            model_loaded = True
        else:
            vectorizer = CountVectorizer()
            X_vectorized = vectorizer.fit_transform(X)

            class_counts = pd.Series(y_encoded).value_counts()
            if (class_counts < 2).any():
                st.error("Certaines classes ont moins de 2 exemples. Merci d’en fournir au moins 2 pour chaque type.")
            else:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_vectorized, y_encoded, test_size=0.4, stratify=y_encoded, random_state=42
                )

                model = MultinomialNB()
                model.fit(X_train, y_train)

                # Sauvegarde
                joblib.dump(model, "modele_classification.pkl")
                joblib.dump(vectorizer, "vectorizer.pkl")
                joblib.dump(label_encoder, "label_encoder.pkl")
                model_loaded = True

        # Si modèle dispo
        if model_loaded:
            X_vectorized = vectorizer.transform(X)
            y_pred = model.predict(X_vectorized)

            labels = unique_labels(y_encoded, y_pred)
            report = classification_report(y_encoded,y_pred,labels=labels,target_names=label_encoder.inverse_transform(labels),output_dict=True)
            st.subheader("Rapport de classification")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

            st.subheader("Tester une prédiction manuelle")
            user_message = st.text_area("Tapez un message d'erreur pour prédiction")
            if user_message:
                user_vector = vectorizer.transform([user_message])
                pred = model.predict(user_vector)
                pred_label = label_encoder.inverse_transform(pred)
                st.success(f"Type d'erreur prédit : **{pred_label[0]}**")

    else:
        st.error("Le fichier doit contenir les colonnes 'message' et 'type_erreur'.")

else:
    st.info("Veuillez importer un fichier CSV pour commencer.")