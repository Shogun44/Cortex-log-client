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