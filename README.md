# Assistant intelligent de classification des erreurs de logs

Ce projet est une application développée avec **Streamlit** permettant d’analyser automatiquement des fichiers de logs, de détecter et classifier les erreurs rencontrées, d’expliquer leur nature et de proposer des pistes de résolution.

## Objectif

L’objectif de ce projet est de fournir un assistant intelligent aux développeurs et équipes DevOps pour :

- Lire et analyser les logs bruts (.log, .txt, etc.)
- Classifier automatiquement les types d'erreurs
- Suggérer des explications ou actions à prendre
- Apprendre de nouvelles erreurs grâce à la correction manuelle
- Suivre l’évolution des erreurs dans les phases de CI/CD

---

## Fonctionnalités

- **Lecture de fichiers logs**
- **Nettoyage intelligent** des messages (lowercase, ponctuation, etc.)
- **Vectorisation TF-IDF**
- **Classification via plusieurs modèles** :
  - SGDClassifier (incrémental)
  - Random Forest
  - SVM (meilleure performance actuelle)
- **Évaluation automatique** (f1-score)
- **Sauvegarde et réutilisation des modèles**
- **Correction manuelle** et apprentissage continu

---

## Résultats (f1-score)

| Modèle              | f1-score |
|---------------------|----------|
| SGDClassifier       | 0.73     |
| Random Forest       | 0.69     |
| SVM (LinearSVC)     | **0.78** |

> Le modèle SVM est actuellement le plus performant.
>
> 
⸻

📊 Détection intelligente des erreurs de logs

Bienvenue dans cette application Streamlit d’analyse intelligente des messages d’erreurs et logs systèmes.
Ce projet vise à fournir un assistant intelligent durant les phases de tests et de CI/CD, pour mieux comprendre, diagnostiquer, et corriger les erreurs.

⸻

🚀 Objectifs
	•	Lire automatiquement des messages de logs (.csv, .log, .txt)
	•	Classer les erreurs selon leur type (api, ldap, base_de_donnees, etc.)
	•	Donner une explication claire de l’erreur détectée
	•	Proposer une piste de résolution
	•	Permettre un apprentissage continu à partir des corrections manuelles
	•	Historiser chaque analyse dans un fichier CSV

⸻

🧠 Fonctionnalités principales

📥 Import de données
	•	Import d’un fichier CSV avec deux colonnes : message, type_erreur
	•	Nettoyage automatique des messages : minuscules, suppression des caractères spéciaux, lemmatisation, suppression des stopwords

🧠 Entraînement & comparaison de modèles
	•	Trois modèles sont entraînés :
	•	SGDClassifier
	•	RandomForestClassifier
	•	SVM (LinearSVC)
	•	Le texte est vectorisé avec TfidfVectorizer
	•	Un tableau compare les F1-scores de chaque modèle
	•	Le modèle le plus performant est automatiquement sélectionné

🧪 Test manuel
	•	L’utilisateur peut saisir un message d’erreur libre
	•	Le modèle prédit le type, explique le contexte, et propose une action

📄 Analyse de fichier .log ou .txt
	•	Lecture de chaque ligne
	•	Prédiction automatique du type d’erreur
	•	Affichage des résultats avec surlignage
	•	Possibilité de corriger manuellement
	•	Téléchargement du fichier annoté

⸻

🧠 Historique & Apprentissage

🔁 Correction manuelle
	•	Si le modèle se trompe, l’utilisateur peut sélectionner la bonne classe
	•	Le modèle apprend immédiatement grâce à partial_fit()

📊 Historique des erreurs
	•	Chaque prédiction est enregistrée dans historique_erreurs.csv avec :
	•	Le message
	•	Le type prédit
	•	Le modèle utilisé
	•	L’origine (log ou manuel)
	•	La date/heure
	•	L’utilisateur peut :
	•	Filtrer par type ou origine
	•	Visualiser l’évolution des erreurs
	•	Exporter l’historique

⸻

📦 Structure du projet

├── app.py                         # Code principal de l'application Streamlit
├── modele_incremental.pkl        # Modèle SGD entraîné de façon incrémentale
├── modele_rf.pkl                 # Modèle Random Forest
├── modele_svm.pkl                # Modèle SVM
├── vectorizer.pkl                # TfidfVectorizer
├── label_encoder.pkl             # Encodage des classes
├── historique_erreurs.csv        # Historique de toutes les erreurs analysées
├── README.md                     # Ce fichier
└── requirements.txt              # Fichier des dépendances


⸻

🛠️ Installation
	1.	Créer un environnement :

conda create -n log_ai_env python=3.10
conda activate log_ai_env

	2.	Installer les dépendances :

pip install -r requirements.txt
python -m nltk.downloader stopwords
python -m spacy download fr_core_news_sm

	3.	Lancer l’application :

streamlit run app.py


⸻

📅 Dernière mise à jour

21/05/2025

⸻


---

## Installation

### Prérequis

- Python 3.8+
- pip
- virtualenv (optionnel mais recommandé)
  
 ###  Utilisation
	1.	Lancer l’application Streamlit
	2.	Charger un fichier de logs
	3.	L’outil nettoie, vectorise, classifie les erreurs et affiche les rapports
	4.	Possibilité de corriger manuellement une erreur si mal détectée

⸻

 ### Fichiers importants
	•	app.py : application principale
	•	modele_svm.pkl, modele_rf.pkl, modele_incremental.pkl : modèles entraînés
	•	vectorizer.pkl, label_encoder.pkl : outils de transformation
	•	requirements.txt : dépendances Python
	•	README.md : documentation technique

⸻

 ### Auteurs
	•	Joan MBALLA — Étudiant IA chez Keyce Informatique
	•	Projet réalisé dans le cadre d’un stage chez Zenity / KORI Asset Management

### Installation locale

```bash
git clone http://192.168.1.23:8084/joan/classification-logs-app.git
cd  C:\Users\Pawk68\OneDrive\Bureau\mon_projet_steamlit
pip install -r requirements.txt
streamlit run app.py

