import smtplib
import re
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import streamlit as st
import streamlit.components.v1 as components
import base64
import os
import tempfile
import io
from semantic_model import cluster_erreurs, generer_rapport_clusters, generer_pdf

# === CONFIG
st.set_page_config(page_title="CortexLog", layout="centered",page_icon="🧠")
def est_email_valide(email):
    return re.match(r"[^@]+@[^@]+\.[^@]+", email)
def envoyer_par_mail(destinataire, pdf_bytes):
    expediteur = "andyjoan004@gmail.com"  # ←  adresse Gmail
    mot_de_passe = "bwgnhgyyulmexnsg"  # ←  mot de passe app Gmail

    msg = MIMEMultipart()
    msg['From'] = expediteur
    msg['To'] = destinataire
    msg['Subject'] = "Rapport CortexLog"

    body = MIMEText("Bonjour,\n\nVeuillez trouver ci-joint le rapport CortexLog généré automatiquement.\n\nCordialement,\nL’équipe CortexLog", 'plain')
    msg.attach(body)

    piece_jointe = MIMEApplication(pdf_bytes.read(), Name="rapport_logs.pdf")
    piece_jointe['Content-Disposition'] = 'attachment; filename="rapport_logs.pdf"'
    msg.attach(piece_jointe)

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(expediteur, mot_de_passe)
        server.send_message(msg)
        server.quit()
        return True
    except Exception as e:
        st.error(f"Erreur lors de l'envoi de l'email : {e}")
        return False


# === FOND
def set_background():
    if os.path.exists("asset/350.png"):
        with open("asset/350.png", "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """, unsafe_allow_html=True)
set_background()

# === STYLES UI
st.markdown("""
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
div[data-testid="stToolbar"]{visibilty: hidden; height: 0px;}

html, body, .stApp {
    margin: 0 !important;
    padding: 0 !important;
    height: 100% !important;
    overflow: hidden;
}
.css-18e3rh9 { padding: 1rem !important; }
.block-container { padding : 0rem 1rem 1rem 1rem !important; }

.logo-container {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-bottom: 30px;
    text-shadow: 1px 1px 2px #000;
}
.card {
    background-color: rgba(255, 255, 255, 0.15);
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 0px 12px rgba(0,0,0,0.2);
    backdrop-filter: blur(4px);
}
.aide-fixed {
    position: fixed;
    bottom: 0px;
    right: 0px;
    z-index: 9999;
    padding: 10px;
            
}
</style>
""", unsafe_allow_html=True)

# === LOGO CORTEXLOG
if os.path.exists("logo222.png"):
    with open("logo222.png", "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode()
    st.markdown(f"""
    <div class="logo-container">
        <img src="data:image/png;base64,{logo_b64}" width="60" style="margin-right: 15px;">
        <span style="font-size:32px;font-weight:bold;color:white;">CortexLog</span>
    </div>
    """, unsafe_allow_html=True)

# === ZONE IMPORT FILE (.log / .txt)
rapport_txt = ""
uploaded_lines = []

st.markdown("### 📂 Importer un fichier de logs (.log ou .txt)")

fichier = st.file_uploader("Déposez ici ou cliquez pour importer", type=["log", "txt"])
if fichier:
    content = fichier.read().decode("utf-8")
    uploaded_lines = [line.strip() for line in content.splitlines() if line.strip()]
    st.success(f"✅ {len(uploaded_lines)} lignes importées.")
   

# === ANALYSE DES LOGS + FILTRAGE INTELLIGENT

    # Étape A : filtrage automatique
    niveaux_critiques = ("ERROR", "CRITICAL", "WARN")
    lignes_filtrees = [l for l in uploaded_lines if any(n in l for n in niveaux_critiques)]

    # Étape B : fallback si aucun message critique
    if len(lignes_filtrees) < 2:
        st.warning("✅ Aucun message d’erreur critique trouvé (ERROR, WARN, CRITICAL).")
        inclure_info_debug = st.checkbox("Inclure aussi les messages INFO et DEBUG ?")

        if inclure_info_debug:
            lignes_filtrees = uploaded_lines
            st.info("🟢 Analyse lancée avec tous les messages (INFO, DEBUG inclus).")
        else:
            st.stop()
    else:
        inclure_info_debug = False

    # === Lancer analyse avec explication intelligente
    nb_clusters = min(4, len(lignes_filtrees))
    clusters = cluster_erreurs(lignes_filtrees, n_clusters=nb_clusters)
    rapport_txt = generer_rapport_clusters(clusters, uploaded_lines, lignes_filtrees)


# === AFFICHAGE RAPPORT + TÉLÉCHARGEMENT
if rapport_txt:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.text_area("🧠 Rapport d’analyse", rapport_txt, height=300, label_visibility="visible")
    st.markdown("</div>", unsafe_allow_html=True)

    # Export TXT
    st.download_button("⬇️ Télécharger le rapport (.txt)", data=rapport_txt, file_name="rapport_logs.txt", mime="text/plain")

    pdf_buffer = generer_pdf(rapport_txt)

    st.download_button("⬇️ Télécharger le rapport (.pdf)", data=pdf_buffer, file_name="rapport_logs.pdf", mime="application/pdf")
    st.markdown("---")
    st.markdown("### 📤 Envoyer le rapport par mail")

    email = st.text_input("Adresse e-mail du destinataire")

    if st.button("📤 Envoyer le rapport") and email and pdf_buffer:
        if not email:
            st.warning("veuillez saisir une adresse email.")
        elif not est_email_valide(email):
            st.warning(" l'adresse email n'est pas valide.")
        elif rapport_txt and pdf_buffer:
             pdf_buffer.seek(0)  # Très important pour lire depuis le début
             if envoyer_par_mail(email, pdf_buffer):
                st.success("✅ Rapport envoyé avec succès à " + email)
# === BOUTON AIDE EN BAS À DROITE
if os.path.exists("asset/Information-button.png"):
    with open("asset/Information-button.png", "rb") as f:
        aide_b64 = base64.b64encode(f.read()).decode()
    components.html(f"""
    <div class="aide-fixed">
        <img src="data:image/png;base64,{aide_b64}" width="50"
             onclick="alert('📂 Importez un fichier .log ou .txt\\n🧠 Analyse automatique des erreurs\\n⬇️ Téléchargez le rapport en .txt ou .pdf')"
             style="cursor:pointer;">
    </div>
    """, height=80)