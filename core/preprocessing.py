import re
import spacy
import nltk
from nltk.corpus import stopwords

# Télécharger les stopwords si jamais ils ne sont pas encore installés
try:
    stopwords_fr = set(stopwords.words("french"))
except LookupError:
    nltk.download("stopwords")
    stopwords_fr = set(stopwords.words("french"))

# Charger le modèle spaCy
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "fr_core_news_sm"])
    nlp = spacy.load("fr_core_news_sm")

def nettoyer_message(texte):
    texte = texte.lower()
    texte = re.sub(r"[^a-zA-Z\s]", " ", texte)
    doc = nlp(texte)
    tokens = [
        token.lemma_ for token in doc
        if token.lemma_ not in stopwords_fr and not token.is_punct
    ]
    return " ".join(tokens)