import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Orientação Espiritual V4", layout="centered")

st.title("📖 Orientação Espiritual V4 (Blindado)")
st.write("Sistema robusto com validação automática de dados.")

# =========================
# CARREGAR BASE COM SEGURANÇA
# =========================
try:
    with open("base.json", "r", encoding="utf-8") as f:
        raw_data = json.load(f)
except Exception as e:
    st.error("Erro ao carregar base.json")
    st.stop()

# =========================
# VALIDAÇÃO + LIMPEZA (CORE DO V4)
# =========================
data = []

for i, item in enumerate(raw_data):
    contexto = item.get("contexto", "").strip()

    if not contexto:
        st.warning(f"Item {i} ignorado: sem 'contexto'")
        continue

    clean_item = {
        "contexto": contexto,
        "emocional": item.get("emocional", "Não informado."),
        "mensagem": item.get("mensagem", "Sem mensagem disponível."),
        "direcionamento": item.get("direcionamento", "Reflita com calma sobre sua situação."),
        "livro": item.get("livro", "Desconhecido"),
        "ano": item.get("ano", 0)
    }

    data.append(clean_item)

if len(data) == 0:
    st.error("Base inválida: nenhum dado utilizável foi encontrado.")
    st.stop()

# =========================
# MODELO IA
# =========================
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
        resultados = buscar(pergunta)

        for r in resultados:

            st.markdown("## 💛 Leitura emocional")
            st.info(r["emocional"])

            st.markdown("## 📖 Reflexão")
            st.success(r["mensagem"])

            st.markdown("## 🧭 Direcionamento")
            st.warning(r["direcionamento"])

            st.markdown(f"📚 Livro: **{r['livro']}**")
            st.markdown(f"📅 Ano: **{r['ano']}**")

            st.markdown("---")
