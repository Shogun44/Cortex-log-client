import csv
from datetime import datetime 
HISTORIQUE_PATH = "historique_erreurs.csv"
def ajouter_erreur_dans_historique(message, type_pred, origine="manuel", modele="SGD"):
    horodatage = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(HISTORIQUE_PATH, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([horodatage, message, type_pred, origine, modele])