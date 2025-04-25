
⸻

Étude de l’existant – Projet IA : Classification intelligente des erreurs de test et de déploiement

⸻

1. Clarification des objectifs de l’étude

L’objectif du projet est de concevoir une IA capable d’assister les développeurs et testeurs dans la compréhension automatique des erreurs rencontrées durant les tests et déploiements d’une application.

L’outil visé :
	•	Lit un message d’erreur
	•	Classifie automatiquement le type d’erreur
	•	Explique humainement l’erreur détectée
	•	Propose une solution possible
	•	Apprend en continu à partir des corrections manuelles
	•	Historise les erreurs récurrentes

Le projet cible la phase de post-test et de déploiement CI/CD, deux étapes souvent négligées dans les outils IA traditionnels.

⸻

2. Étude de marché

L’analyse du marché montre que :
	•	Les erreurs de test sont encore souvent analysées manuellement
	•	Peu d’outils assistent l’explication ou la compréhension des erreurs
	•	Les solutions existantes (Launchable, Testim, Sentry) se concentrent sur :
	•	La génération de tests
	•	La priorisation des tests
	•	La surveillance des erreurs en production

Aucune solution ne traite les messages d’erreurs pendant les tests manuels ou les déploiements CI/CD, ce qui confirme la pertinence du projet.

⸻

3. Analyse de l’environnement de l’entreprise


| Élément              | Détail                                                   |
|----------------------|-----------------------------------------------------------|
| Entreprise           | KORI Asset Management (Bali) – projet réalisé chez Zenity |
| Application          | OF INTRA – plateforme RH interne                          |
| Frontend             | Next.js                                                   |
| Backend              | Spring Boot                                               |
| Base de données      | PostgreSQL                                                |
| Authentification     | Serveur LDAP (QA, Dev, Production)                        |
| Conteneurisation     | Docker                                                    |
| CI/CD                | Présent (outil non précisé)                               |


Les tests sont manuels, les erreurs visibles apparaissent :
	•	En console navigateur (API)
	•	En alertes visuelles sur l’interface

⸻

4. Analyse qualitative
	•	Les erreurs sont détectées manuellement (console ou navigateur)
	•	L’analyse repose sur l’intuition des testeurs
	•	Les messages sont parfois copiés manuellement dans des documents
	•	Il n’existe aucune base de données d’erreurs
	•	Les erreurs sont récurrentes, souvent reconnues de mémoire
	•	L’équipe serait intéressée par un outil capable de :
	•	Classer
	•	Expliquer
	•	Aider à corriger une erreur, de façon pédagogique

⸻

5. Analyse quantitative

⚠️ Données à confirmer plus tard par l’équipe

	•	Les tests ne sont pas encore tous réalisés
	•	Le volume de tests n’est pas encore chiffré
	•	Les types d’erreurs fréquents ne sont pas encore documentés
	•	Les erreurs sont stockées manuellement, sans outil centralisé

Il est recommandé de tenir un journal des tests et erreurs dès la première phase de validation pour alimenter l’apprentissage du modèle IA.

⸻

6. Analyse de la concurrence


| Outil ou solution   | Fonctionnalité principale                                           |
|---------------------|---------------------------------------------------------------------|
| Launchable          | Prédiction des tests les plus susceptibles d’échouer               |
| Testim (Tricentis)  | Génération et stabilisation des tests via IA                       |
| Diffblue Cover      | Génération automatique de tests unitaires en Java via IA           |
| Sentry              | Détection d’anomalies et d’erreurs en production                   |
| Rookout             | Débogage en temps réel via insertion de logs dynamiques            |

Aucun outil ne cible l’analyse post-test manuel ni l’interprétation intelligente de messages d’erreur CI/CD.

⸻

7. Repérage des concurrents à analyser

3 outils ont été sélectionnés pour comparaison approfondie :
	•	Sentry
	•	Launchable
	•	Testim

⸻

8. Profil des concurrents retenus


| Critère                  | Sentry                        | Launchable                     | Testim (Tricentis)            |
|--------------------------|-------------------------------|---------------------------------|-------------------------------|
| Objectif principal       | Surveillance des erreurs prod | Prédiction de tests à lancer   | Génération de tests UI        |
| Type d’IA                | Regroupement automatique      | ML sur historique              | Apprentissage comportemental  |
| Cible métier             | Devs, DevOps                  | CI/CD                          | QA, Frontend Dev              |
| Phase ciblée             | Post-production               | Avant test                     | Pré-test                      |
| Explication fournie      | Oui (stacktrace, contexte)    | Non                            | Non                           |
| Interaction humaine      | Lecture, feedback             | Suivi statistiques             | Ajustement manuel             |
| Lien avec ton projet     | Phase post-erreur (similaire) | Complémentaire (en amont)      | Apprentissage comme modèle    |
| Modèle économique        | Freemium / SaaS               | Produit pro                    | Commercial                    |


⸻

9. Analyse des informations collectées
	•	Le besoin d’aide à la compréhension des erreurs est réel
	•	Il existe peu ou pas d’outil pour la phase post-test manuel
	•	Le projet peut aussi cibler les logs CI/CD, comme ceux produits lors des échecs de déploiement GitLab

Extension pertinente ajoutée :

L’IA peut également analyser les erreurs survenues lors du déploiement CI/CD, très techniques et souvent mal comprises :
	•	Échecs de build
	•	Problèmes de dépendance
	•	Erreurs réseau ou LDAP
	•	Plantages backend

⸻

10. Enjeux pédagogiques et évolutifs

L’objectif est de créer une IA qui :
	•	Classifie les erreurs de manière fiable
	•	Explique de manière compréhensible ce qui s’est passé
	•	Suggère des solutions
	•	Apprend continuellement
	•	S’ouvre à l’analyse de logs backend et CI/CD

⸻

11. Recommandations pour la suite
	•	Lancer une phase de collecte de messages réels (tests + CI/CD)
	•	Annoter manuellement un jeu de données de départ
	•	Structurer un pipeline IA évolutif
	•	Ajouter une fonction de correction manuelle (feedback utilisateur)
	•	Envisager l’extension vers les logs techniques GitLab (CI)

| Étape recommandée                                 | Objectif                                                             |
|---------------------------------------------------|----------------------------------------------------------------------|
| Collecte des messages d’erreur                    | Créer un vrai jeu de données réaliste                                |
| Annotation manuelle                               | Permettre l’apprentissage supervisé                                  |
| Entraînement IA progressif                        | Améliorer le modèle au fur et à mesure                               |
| Feedback utilisateur                              | Affiner le modèle avec les corrections humaines                      |
| Intégration des logs CI/CD                        | Étendre l’outil aux erreurs de déploiement GitLab                    |
| Ajout d’une catégorie "erreur inconnue"           | Gérer les messages ambigus ou nouveaux                               |

⸻
