


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

# Évaluation des Modèles de Classification des Erreurs de Logs

## Objectif

L'objectif est de sélectionner le meilleur modèle de machine learning pour classer automatiquement les erreurs extraites des logs applicatifs.

## Dataset

- Fichier CSV contenant deux colonnes : `message` (texte du log) et `type_erreur` (classe de l’erreur).
- Nettoyage effectué : minuscules, suppression des caractères spéciaux, lemmatisation, stopwords retirés.
- Vectorisation : `TfidfVectorizer`

## Méthode de validation

- Division du dataset en **80% entraînement / 20% test** avec `train_test_split`
- Évaluation sur le set de test avec la métrique **F1-score macro**

## Modèles testés

| Modèle          | F1-score | Commentaire                             |
|------------------|----------|------------------------------------------|
| SVM (Support Vector Machine) | **0.78** | Meilleur score actuel, choix retenu     |
| Sgdclassifier     | 0.73     | Bon score mais moins stable             |
| Random Forest     | 0.69     | Bon score mais moins stable                |

## Modèle retenu

Le modèle **SVM** est retenu pour la suite du projet.  
Il offre le meilleur compromis entre performance et généralisation sur les données actuelles.

## Prochaines étapes

- Enrichissement du dataset avec de nouveaux exemples d’erreurs
- Apprentissage incrémental pour intégrer les corrections manuelles
- Intégration dans l’interface utilisateur Streamlit
- Surveillance continue des performances lors des phases de test et de CI/CD

---