import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Orientação Espiritual V7", layout="centered")

st.title("📖 Orientação Espiritual V7")
st.write("Mensagem espiritual em texto contínuo, baseada em princípios kardecistas.")

# =========================
# BASE
# =========================
with open("base.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

data = [item for item in raw_data if item.get("contexto")]

texts = [item["contexto"] for item in data]
embeddings = model.encode(texts)

# =========================
# BUSCA
# =========================
def buscar(pergunta):
    emb = model.encode([pergunta])[0]

    sim = np.dot(embeddings, emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(emb)
    )

    idx = np.argmax(sim)
    return data[idx]

# =========================
# GERAÇÃO DE TEXTO FINAL (ESSÊNCIA DO V7)
# =========================
def gerar_texto(item, pergunta):

    intro = (
        "Diante da situação que você está vivendo, é natural que surjam sentimentos de angústia e incerteza, "
        "especialmente quando envolve responsabilidades importantes e o medo do futuro."
    )

    corpo = item.get("mensagem", "")

    fechamento = (
        "Segundo a visão espiritualista presente nas obras de Emmanuel, André Luiz e nos ensinamentos transmitidos por "
        "Chico Xavier e Divaldo Franco, as dificuldades não representam abandono, mas oportunidades de fortalecimento interior, "
        "reorganização emocional e aprendizado gradual. Ainda que a dor seja real, ela não é permanente, e pode ser transformada "
        "pela fé ativa, pela serenidade e pela ação responsável no presente."
    )

    return f"{intro}\n\n{corpo}\n\n{fechamento}"

# =========================
# INPUT
# =========================
pergunta = st.text_area("🧠 Descreva sua situação")

if st.button("Receber orientação"):
    if not pergunta.strip():
        st.warning("Digite sua situação.")
    else:
        item = buscar(pergunta)

        resposta = gerar_texto(item, pergunta)

        st.markdown("## 📖 Orientação espiritual")
        st.success(resposta)

        st.caption(f"Base conceitual: {item.get('livro','')} ({item.get('ano','')})")
