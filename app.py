import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="Orientação Espiritual V6", layout="centered")

st.title("📖 Orientação Espiritual V6")
st.write("Sistema de orientação com foco em crise emocional real.")

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
# DETECÇÃO DE CRISE
# =========================
def detectar_crise(texto):
    t = texto.lower()

    if any(w in t for w in ["desesper", "sem dinheiro", "emprego", "filho", "contas", "aluguel"]):
        return "crise_material"

    if any(w in t for w in ["traição", "separação", "termino"]):
        return "crise_afetiva"

    if any(w in t for w in ["triste", "depress", "vazio"]):
        return "crise_emocional"

    return "geral"

# =========================
# BUSCA INTELIGENTE
# =========================
def buscar(pergunta):
    emb = model.encode([pergunta])[0]

    sim = np.dot(embeddings, emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(emb)
    )

    idx_sorted = np.argsort(sim)[::-1]

    return [data[i] for i in idx_sorted]

# =========================
# INPUT
# =========================
pergunta = st.text_area("🧠 Conte sua situação")

if st.button("Analisar"):
    if not pergunta.strip():
        st.warning("Digite sua situação.")
    else:

        crise = detectar_crise(pergunta)
        resultados = buscar(pergunta)

        principal = resultados[0]
        secundario = resultados[1] if len(resultados) > 1 else None

        # =========================
        # INTRO HUMANA (ESSENCIAL)
        # =========================
        if crise == "crise_material":
            st.error("Perder o emprego enquanto se tem responsabilidade familiar é uma das situações mais desafiadoras emocionalmente, pois envolve medo, pressão e urgência prática.")

        elif crise == "crise_afetiva":
            st.error("Situações de ruptura afetiva geram impacto profundo na estrutura emocional e no senso de estabilidade pessoal.")

        else:
            st.info("Sua situação envolve um processo emocional que exige compreensão e reorganização interna.")

        # =========================
        # RESPOSTA PRINCIPAL
        # =========================
        st.markdown("## 💛 Direcionamento principal")
        st.success(principal["mensagem"])

        st.markdown("## 🧭 Orientação prática")
        st.warning(principal["direcionamento"])

        st.markdown(f"📚 Livro: **{principal['livro']}**")

        # =========================
        # COMPLEMENTO (SE RELEVANTE)
        # =========================
        if secundario:
            st.markdown("---")
            st.markdown("## 📖 Complemento")
            st.info(secundario["mensagem"])
