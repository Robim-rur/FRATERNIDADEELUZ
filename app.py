import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG CHAT
# =========================
st.set_page_config(page_title="Reflexões", layout="centered")

st.title("💬 Reflexões e Direcionamento")

# =========================
# MEMÓRIA DE CHAT (SESSION)
# =========================
if "chat" not in st.session_state:
    st.session_state.chat = []

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
# RESPOSTA HUMANA (SEM FORMALIDADE)
# =========================
def gerar_resposta(pergunta, item):

    corpo = item.get("mensagem", "")

    fechamento = (
        "Se quiser, me conta um pouco mais — às vezes entender melhor o contexto ajuda a organizar melhor os próximos passos."
    )

    return f"{corpo}\n\n{fechamento}"

# =========================
# MOSTRAR CHAT
# =========================
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =========================
# INPUT ESTILO CHAT
# =========================
user_input = st.chat_input("Digite aqui...")

if user_input:

    # usuário
    st.session_state.chat.append({"role": "user", "content": user_input})

    item = buscar(user_input)
    resposta = gerar_resposta(user_input, item)

    # assistente
    st.session_state.chat.append({"role": "assistant", "content": resposta})

    st.rerun()
