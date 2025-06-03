
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import re 
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from io import BytesIO
import os
import textwrap
import json
import unicodedata
import numpy as np

# Charger le modèle BERT léger
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Chargement de la base locale d'erreurs connues ===
def charger_base_erreurs():
    chemin = os.path.join(os.path.dirname(__file__), "erreurs_connues.json")
    if not os.path.exists(chemin):
        return []

    with open(chemin, "r", encoding="utf-8") as f:
        return json.load(f)
def remove_emojis(text):
    return re.sub(r'[^\x00-\x7F]+','',text) #supprime tous les caractères non-ASCII

# === Trouver l'explication la plus proche ===
def expliquer_par_similarity(ligne, base_erreurs, model):
    if not base_erreurs:
        return "Erreur inconnue", "Aucune base d’erreurs n’a été chargée."

    emb_ligne = model.encode([ligne])[0]
    exemples = [err["exemple"] for err in base_erreurs]
    emb_exemples = model.encode(exemples)

    similarities = np.dot(emb_exemples, emb_ligne) / (
        np.linalg.norm(emb_exemples, axis=1) * np.linalg.norm(emb_ligne)
    )

    index_best = int(np.argmax(similarities))
    best = base_erreurs[index_best]

    return best["titre"], best["explication"]

# === Regroupement sémantique des lignes ===
def cluster_erreurs(lignes, n_clusters=4):
    embeddings = model.encode(lignes)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)
    clusters = {}
    for i, label in enumerate(labels):
        clusters.setdefault(label, []).append(lignes[i])
    return list(clusters.values())

# === Génération du rapport final avec titres et explications ===
def generer_rapport_clusters(clusters, lignes_initiales, lignes_filtrees):
    base_erreurs = charger_base_erreurs()
    rapport = f"📊 Rapport d’analyse – CortexLog\n\n"
    rapport += f"Nombre total de lignes analysées : {len(lignes_initiales)}\n"
    rapport += f"Nombre de lignes critiques retenues : {len(lignes_filtrees)}\n"
    rapport += f"Nombre de groupes détectés : {len(clusters)}\n"
    rapport += "\n---\n"
    
    for i, cluster in enumerate(clusters):
        titre, explication = expliquer_par_similarity(cluster[0], base_erreurs, model)
        rapport += f"🔹 Groupe {i+1} – {len(cluster)} message(s)\n"
        rapport += f"📌 **{titre}**\n"
        rapport += f"🧠 {explication}\n"
        for ligne in cluster[:10]:  # Afficher 10 lignes max par groupe
            rapport += f" - {ligne.strip()}\n"
        rapport += "\n"
    return rapport.encode('utf-8', errors='ignore').decode('utf-8')

def generer_pdf(rapport_txt):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4

    # Charger une police mono-espacée pour lisibilité (comme un terminal)
    font_path = os.path.join("fonts", "DejaVuSansMono.ttf")
    pdfmetrics.registerFont(TTFont('DejaVuMono', font_path))
    c.setFont("DejaVuMono", 10)

    x = 20 * mm
    y = height - 20 * mm

    for ligne in rapport_txt.splitlines():
        # Découpe longue ligne (même sans espace) pour éviter plantage
        ligne = re.sub(r'([^\s]{40})', r'\1 ',ligne)
        morceaux = textwrap.wrap(ligne, width=110, break_long_words=False)
        for morceau in morceaux:
            c.drawString(x, y, morceau)
            y -= 12
            if y < 20 * mm:
                c.showPage()
                c.setFont("DejaVuMono", 10)
                y = height - 20 * mm

        # Ajouter une ligne vide après chaque message
        y -= 6
        if y < 20 * mm:
            c.showPage()
            c.setFont("DejaVuMono", 10)
            y = height - 20 * mm

    c.save()
    buffer.seek(0)
    return buffer