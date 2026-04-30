import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Orientação Espiritual V3", layout="centered")

st.title("📖 Orientação Espiritual V3")
st.write("Descreva sua situação e receba uma leitura emocional + reflexão + direcionamento.")

# =========================
# BASE
# =========================
with open("base.json", "r", encoding="utf-8") as f:
    data = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [item["contexto"] for item in data]
embeddings = model.encode(texts)

# =========================
# BUSCA SEMÂNTICA
# =========================
def buscar(pergunta):
    emb = model.encode([pergunta])[0]

    sim = np.dot(embeddings, emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(emb)
    )

    idx = np.argsort(sim)[-3:][::-1]
    return [data[i] for i in idx]

# =========================
# GERADOR DE RESPOSTA V3
# =========================
def gerar_resposta(item, pergunta):
    return {
        "emocional": item["emocional"],
        "reflexao": item["mensagem"],
        "direcionamento": item["direcionamento"],
        "livro": item["livro"],
        "ano": item["ano"]
    }

# =========================
# INPUT
# =========================
pergunta = st.text_area("🧠 O que está acontecendo na sua vida?")

if st.button("Analisar situação"):
    if not pergunta.strip():
        st.warning("Digite sua situação.")
    else:
        resultados = buscar(pergunta)

        for r in resultados:
            resposta = gerar_resposta(r, pergunta)

            st.markdown("## 💛 Leitura emocional")
            st.info(resposta["emocional"])

            st.markdown("## 📖 Reflexão")
            st.success(resposta["reflexao"])

            st.markdown("## 🧭 Direcionamento prático")
            st.warning(resposta["direcionamento"])

            st.markdown(f"📚 Livro: **{resposta['livro']}**")
            st.markdown(f"📅 Ano: **{resposta['ano']}**")

            st.markdown("---")
