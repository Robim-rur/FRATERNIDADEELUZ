import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Reflexões e Direcionamento", layout="centered")

st.title("💬 Reflexões e Direcionamento Pessoal")
st.write("Converse livremente e receba uma reflexão útil e coerente.")

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

# FILTRO MAIS ESTRITO (evita lixo silencioso)
data = [
    item for item in raw_data
    if item.get("contexto")
    and item.get("mensagem")
    and len(item["mensagem"].strip()) > 30
]

texts = [item["contexto"] for item in data]
embeddings = model.encode(texts)

# =========================
# BUSCA COM LIMIAR DE CONFIANÇA (ESSENCIAL)
# =========================
def buscar(pergunta):

    emb = model.encode([pergunta])[0]

    sim = np.dot(embeddings, emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(emb)
    )

    best_idx = np.argmax(sim)
    best_score = sim[best_idx]

    # 🔥 AQUI ESTÁ A CORREÇÃO REAL
    # se a confiança for baixa → NÃO inventa resposta falsa
    if best_score < 0.25:
        return None

    return data[best_idx]

# =========================
# RESPOSTA
# =========================
def gerar(pergunta, item):

    if not item:
        return (
            "Entendi o que você está passando. Às vezes, quando tudo parece pesado ao mesmo tempo, "
            "o mais importante é focar no próximo passo possível, mesmo que pequeno. "
            "Se quiser, me conte um pouco mais da sua situação para eu te ajudar melhor."
        )

    corpo = item["mensagem"]

    fechamento = (
        "Se quiser, me conta mais detalhes — isso ajuda a organizar melhor o que você está vivendo e encontrar caminhos mais claros."
    )

    return f"{corpo}\n\n{fechamento}"

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

    item = buscar(user_input)
    resposta = gerar(user_input, item)

    st.session_state.chat.append({"role": "assistant", "content": resposta})

    st.rerun()
