import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

# Titre de l'application
st.set_page_config(page_title="Classification des erreurs de logs", layout="wide")
st.title("Détection et classification automatique des erreurs de logs")

st.markdown("""
Ce mini outil vous permet de :
- Visualiser les erreurs dans les logs sous forme de graphiques
- Entraîner un modèle IA pour classifier automatiquement les messages d'erreur
- Tester la classification d’un message d'erreur en direct
""")

# Upload du fichier
uploaded_file = st.file_uploader("Importer un fichier CSV avec colonnes `message` et `type_erreur`", type="csv")

if uploaded_file:
    # Lecture du CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("Aperçu des données importées")
    st.write(df.head())

    if 'message' in df.columns and 'type_erreur' in df.columns:

        # Affichage de la répartition des erreurs
        st.subheader("Répartition des erreurs (type_erreur)")
        error_counts = df['type_erreur'].value_counts()
        st.bar_chart(error_counts)

        # Préparation des données
        X = df['message']
        y = df['type_erreur']

        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        vectorizer = CountVectorizer()
        X_vectorized = vectorizer.fit_transform(X)

        # Vérifier que chaque classe a au moins 2 occurrences
        class_counts = pd.Series(y_encoded).value_counts()
        if (class_counts < 2).any():
            st.error("Certaines classes ont moins de 2 exemples. Merci de fournir plus d'exemples pour chaque type d'erreur.")
        else:
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X_vectorized, y_encoded, test_size=0.5, stratify=y_encoded, random_state=42
            )

            # Modèle
            model = MultinomialNB()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Rapport
            report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
            st.subheader("Rapport de classification")
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

            # Test manuel d'un message
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