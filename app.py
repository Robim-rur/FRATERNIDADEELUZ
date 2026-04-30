import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Orientação Espiritual V5", layout="centered")

st.title("📖 Orientação Espiritual V5")
st.write("Sistema emocional inteligente com interpretação contextual avançada.")

# =========================
# CARREGAMENTO SEGURO
# =========================
with open("base.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# LIMPEZA ROBUSTA
# =========================
data = []

for item in raw_data:
    contexto = item.get("contexto", "").strip()

    if not contexto:
        continue

    data.append(item)

texts = [item["contexto"] for item in data]
embeddings = model.encode(texts)

# =========================
# DETECÇÃO SIMPLES DE EMOÇÃO
# =========================
def detectar_emocao(texto):
    texto = texto.lower()

    if any(w in texto for w in ["medo", "ansiedade", "preocupação"]):
        return "ansiedade"
    if any(w in texto for w in ["triste", "depress", "vazio"]):
        return "tristeza"
    if any(w in texto for w in ["raiva", "revolta", "injustiça"]):
        return "raiva"
    if any(w in texto for w in ["traição", "infidelidade"]):
        return "dor afetiva"

    return "neutro"

# =========================
# BUSCA INTELIGENTE
# =========================
def buscar(pergunta):
    emb = model.encode([pergunta])[0]

    sim = np.dot(embeddings, emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(emb)
    )

    top_idx = np.argsort(sim)[-3:][::-1]

    return [data[i] for i in top_idx]

# =========================
# INPUT
# =========================
pergunta = st.text_area("🧠 Descreva sua situação")

if st.button("Analisar"):
    if not pergunta.strip():
        st.warning("Digite algo.")
    else:
        emocao = detectar_emocao(pergunta)
        resultados = buscar(pergunta)

        st.info(f"🎯 Emoção detectada: {emocao}")

        for r in resultados:

            st.markdown("## 💛 Leitura emocional")
            st.info(r.get("emocional", ""))

            st.markdown("## 📖 Reflexão")
            st.success(r.get("mensagem", ""))

            st.markdown("## 🧭 Direcionamento")
            st.warning(r.get("direcionamento", ""))

            st.markdown(f"📚 Livro: **{r.get('livro','')}**")
            st.markdown(f"📅 Ano: **{r.get('ano','')}**")

            st.markdown("---")
