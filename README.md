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

