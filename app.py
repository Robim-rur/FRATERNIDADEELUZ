import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Orientação Espiritual", layout="centered")

st.title("📖 Conselhos Espirituais")
st.write("Digite seu problema e receba uma orientação baseada em ensinamentos espirituais.")

# =========================
# CARREGAR BASE
# =========================
with open("base.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# =========================
# MODELO LOCAL
# =========================
model = SentenceTransformer('all-MiniLM-L6-v2')

# =========================
# GERAR EMBEDDINGS
# =========================
temas = [item["tema"] for item in data]
embeddings = model.encode(temas)

# =========================
# FUNÇÃO DE BUSCA
# =========================
def buscar_resposta(pergunta):
    pergunta_emb = model.encode([pergunta])[0]
    
    similaridades = np.dot(embeddings, pergunta_emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(pergunta_emb)
    )
    
    idx = np.argmax(similaridades)
    return data[idx]

# =========================
# INPUT
# =========================
pergunta = st.text_area("🧠 Qual problema você está enfrentando?")

if st.button("Buscar orientação"):
    if pergunta.strip() == "":
        st.warning("Digite algo primeiro.")
    else:
        resultado = buscar_resposta(pergunta)
        
        st.markdown("### ✨ Orientação:")
        st.success(resultado["mensagem"])
        
        st.markdown(f"📚 Livro: **{resultado['livro']}**")
        st.markdown(f"📅 Ano: **{resultado['ano']}**")
