


⸻

# Classification intelligente des erreurs de logs – IA pour les tests logiciels

Projet académique – Bachelor 3 IABD
Réalisé chez KORI Asset Management 

## Présentation du projet

Ce projet vise à créer une intelligence artificielle capable de :
	•	Lire automatiquement les messages d’erreurs issus de tests manuels ou de CI/CD
	•	Classer chaque message dans une catégorie (API, LDAP, réseau, etc.)
	•	Expliquer humainement l’origine probable du problème
	•	Proposer une piste de correction
	•	Apprendre progressivement à partir des retours utilisateurs

L’outil est basé sur :
	•	Python (IA)
	•	Streamlit (interface)
	•	Scikit-learn (modèle de classification)
	•	NLTK / SpaCy (nettoyage NLP)

⸻

## Fonctionnalités clés
	•	Entraînement d’un modèle IA depuis un fichier .csv
	•	Visualisation des types d’erreurs sous forme de graphique
	•	Prédiction en direct d’un message d’erreur saisi manuellement
	•	Réinitialisation possible du modèle pour un nouvel apprentissage
	•	Extension possible aux erreurs CI/CD (GitLab, Jenkins…)

⸻

## Installation

git clone https://gitlab.com/ton-projet.git
cd ton-projet
pip install -r requirements.txt
streamlit run app.py



⸻

## Structure du projet

├── app.py                # Interface IA Streamlit
├── modele_classification.pkl
├── vectorizer.pkl
├── label_encoder.pkl
├── README.md
├── .gitignore
└── data/                 # Exemples de fichiers .csv ou .log



⸻

## Exemple d’utilisation
	1.	Importer un fichier .csv avec deux colonnes : message, type_erreur
	2.	Visualiser les catégories existantes
	3.	Lancer l’apprentissage automatique
	4.	Taper un message d’erreur à classer
	5.	Obtenir la catégorie + explication

⸻

## À venir
	•	Amélioration du modèle avec SVM / Random Forest
	•	Détection d’erreurs inconnues / ambigües
	•	Intégration des logs CI/CD (GitLab)
	•	Historique et feedback utilisateur intégré

⸻

## Auteur

Joan MBALLA
Bachelor 3 – Intelligence Artificielle et Big Data
Stage chez KORI Asset Management 
koriassetmanagement.com

⸻
