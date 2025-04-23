
Étude de l’existant

Projet IA – Classification intelligente des erreurs de logs
Auteur : Joan MBALLA – pawk 2025

⸻

1. Contexte général du projet

L’entreprise pawk , spécialisée dans le conseil en systèmes et logiciels informatiques, développe une application de gestion interne nommée GAP (Gestion des Agents et du Personnel). GAP est une application Web composée :
	•	D’un frontend accessible aux agents
	•	D’un backend développé en Java avec Spring Boot
	•	D’un système d’authentification connecté à un serveur LDAP
	•	D’une infrastructure conteneurisée avec Docker

Pendant ses phases de développement, de test et de production, cette application génère de nombreux logs techniques, indispensables pour surveiller l’état du système, détecter les erreurs et diagnostiquer les dysfonctionnements.



⸻

2. Fonctionnement de GAP

| Élément     | Description                                             |
|-------------|---------------------------------------------------------|
| Backend     | Java – Spring Boot (expose des endpoints `/actuator`)  |
| Frontend    | Interface utilisée par les agents                       |
| LDAP        | Gestion des comptes utilisateurs                        |
| Docker      | Conteneurisation du backend                             |
| Monitoring  | Exposé via Spring Actuator (`/health`, `/loggers`)      |

⸻

3. Identification des sources de logs potentiels

| Source              | Description                        | Exemples d’erreurs attendues         |
|---------------------|------------------------------------|--------------------------------------|
| Spring Actuator     | Infos système et état applicatif   | Erreurs de démarrage, état HS        |
| Docker              | Logs d’exécution conteneurisée     | Crashs, redémarrages, out of memory  |
| LDAP                | Gestion des utilisateurs           | Utilisateur inactif, accès refusé    |
| API (REST)          | Communication frontend-backend     | Erreurs HTTP 404, 500, timeout       |
| Réseau              | Connexions, transferts             | Timeout, refus de connexion          |


⸻

4. Problèmes actuels identifiés
	•	L’analyse des logs est manuelle et chronophage
	•	Risques d’erreurs d’interprétation
	•	Retards dans la détection de bugs fréquents
	•	Logs parfois difficiles à lire ou à classer
	•	Absence de catégorisation normalisée des erreurs

⸻

5. Besoins métiers exprimés
	•	Lecture automatique des fichiers de logs
	•	Classification immédiate des erreurs
	•	Réduction du temps de diagnostic
	•	Amélioration de la fiabilité des tests QA
	•	Interface simple pour les développeurs et testeurs

⸻

6. Outils existants dans l’industrie

| Outil               | Fonction                                 | Pourquoi non retenu                         |
|---------------------|------------------------------------------|---------------------------------------------|
| ELK Stack           | Centralisation et visualisation des logs | Trop complexe et lourd pour un POC          |
| Graylog             | Agrégation des logs avec dashboards      | Nécessite une architecture serveur          |
| Sentry / Datadog    | Monitoring DevOps & alertes temps réel   | Moins adapté à la classification IA         |
| Splunk              | Observabilité complète                   | Solution propriétaire, coût élevé           |


⸻

7. Proposition de catégorisation initiale des erreurs

Cette première classification est basée sur l’observation de logs simulés et sur la structure prévue de l’application GAP. Elle sera validée et enrichie par la suite.

| Catégorie           | Description                                             |
|---------------------|---------------------------------------------------------|
| api                 | Erreurs dans les appels REST (code 500, endpoint manquant) |
| ldap                | Problèmes d’authentification, utilisateur inactif       |
| authentification    | Identifiants invalides, mot de passe incorrect          |
| docker              | Crash de conteneur, redémarrage                         |
| réseau              | Timeout, erreur de connectivité                         |
| base_de_donnees     | Erreurs SQL, accès refusé, requêtes échouées            |
| autres              | Non classable ou bruit                                  |



⸻

8. Outils choisis pour le projet IA

| Outil            | Rôle                                  | Justification                                         |
|------------------|----------------------------------------|-------------------------------------------------------|
| Python           | Langage de développement principal     | Flexible, riche en bibliothèques IA                  |
| Pandas           | Traitement de données logs             | Très adapté aux fichiers tabulaires                 |
| Scikit-learn     | Machine learning (SGD, vectorisation)  | Compatible apprentissage incrémental                |
| Streamlit        | Interface utilisateur                  | Intuitif, rapide à déployer                         |
| SpaCy / NLTK     | Traitement du texte                    | Pour enrichir le prétraitement                      |
| Joblib           | Sauvegarde des modèles entraînés       | Léger et bien intégré à Scikit-learn                |
| GitLab           | Versionnage, collaboration             | Suivi clair du code et des versions                 |
| Markdown / Confluence | Documentation continue            | Pour garder une traçabilité technique complète       |


⸻

9. Conclusion

Cette étude de l’existant permet de :
	•	Cadrer le périmètre fonctionnel et technique du projet
	•	Identifier les sources de logs à exploiter
	•	Proposer une première taxonomie des erreurs à classer
	•	Justifier les choix technologiques et leur adaptation au contexte de Zenity

Elle servira de base pour demander :
	•	Les extraits de logs réels
	•	La validation des catégories
	•	Et pour guider la conception de l’IA et de l’interface

⸻