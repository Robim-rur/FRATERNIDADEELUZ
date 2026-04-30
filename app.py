import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Reflexões e Direcionamento", layout="centered")

st.title("💬 Reflexões e Direcionamento Pessoal")

# =========================
# MEMÓRIA DE CONVERSA
# =========================
if "chat" not in st.session_state:
    st.session_state.chat = []

if "contexto_acumulado" not in st.session_state:
    st.session_state.contexto_acumulado = ""

# =========================
# BASE
# =========================
with open("base.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

data = [
    item for item in raw_data
    if item.get("contexto") and item.get("mensagem")
]

texts = [item["contexto"] for item in data]
embeddings = model.encode(texts)

# =========================
# BUSCA COM CONTEXTO ACUMULADO
# =========================
def buscar(pergunta, contexto_total):

    entrada = pergunta + " " + contexto_total

    emb = model.encode([entrada])[0]

    sim = np.dot(embeddings, emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(emb)
    )

    idx = np.argmax(sim)

    return data[idx]

# =========================
# RESPOSTA CONTÍNUA (IMPORTANTE)
# =========================
def gerar_resposta(pergunta, item, contexto_total):

    corpo = item["mensagem"]

    # DETECTA EVOLUÇÃO DA CRISE
    if "comer" in pergunta.lower() or "fome" in pergunta.lower():
        continuidade = (
            "Quando a dificuldade atinge necessidades básicas, o foco precisa ser ainda mais imediato e prático, "
            "priorizando soluções de curto prazo e apoio externo sempre que possível."
        )
    else:
        continuidade = ""

    fechamento = (
        "Se quiser, continue me contando — vou te ajudando a organizar isso passo a passo, sem pressa e sem sobrecarga."
    )

    return f"{corpo} {continuidade}\n\n{fechamento}"

# =========================
# CHAT
# =========================
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =========================
# INPUT
# =========================
user_input = st.chat_input("Me conte um pouco do que está acontecendo com você atualmente.")

if user_input:

    st.session_state.chat.append({"role": "user", "content": user_input})

    # 🔥 atualiza contexto acumulado
    st.session_state.contexto_acumulado += " " + user_input

    item = buscar(user_input, st.session_state.contexto_acumulado)

    resposta = gerar_resposta(user_input, item, st.session_state.contexto_acumulado)

    st.session_state.chat.append({"role": "assistant", "content": resposta})

    st.rerun()
