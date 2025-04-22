# Projet IA – Classification intelligente des erreurs de logs

**Objectif :** Créer une application capable de lire automatiquement des fichiers de logs, de détecter les erreurs, de les classer, et de s’améliorer au fur et à mesure grâce au feedback utilisateur.

---

## Fonctionnalités

- Upload de fichiers `.csv` (avec colonnes `message` et `type_erreur`)
- Upload de fichiers `.log` bruts
- Prédiction automatique du type d’erreur (API, LDAP, Authentification, etc.)
- Apprentissage incrémental (le modèle apprend des nouveaux exemples)
- Interface intuitive avec Streamlit
- Visualisation colorée + filtres dynamiques
- Export des résultats en CSV

---

## Lancer l’application

1. Active ton environnement :

```bash
conda activate log_ai_env