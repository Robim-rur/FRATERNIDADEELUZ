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
# CHAT
# =========================
if "chat" not in st.session_state:
    st.session_state.chat = []

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
# 🔥 RESPOSTA LIMPA (SEM CAMADAS)
# =========================
def gerar_resposta(pergunta, item):

    base = item["mensagem"]

    # 🔥 aqui entra o ajuste mais importante de todos:
    # transforma em fala única, sem blocos repetidos

    resposta_final = f"{base} Se quiser, me conta mais um pouco do que está acontecendo, para eu te ajudar melhor a organizar isso."

    return resposta_final

# =========================
# CHAT
# =========================
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =========================
# INPUT
# =========================
user_input = st.chat_input("Me conte o que está acontecendo com você...")

if user_input:

    st.session_state.chat.append({"role": "user", "content": user_input})

    item = buscar(user_input)

    resposta = gerar_resposta(user_input, item)

    st.session_state.chat.append({"role": "assistant", "content": resposta})

    st.rerun()
