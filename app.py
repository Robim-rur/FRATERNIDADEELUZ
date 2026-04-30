import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Reflexões e Direcionamento", layout="centered")

st.title("💬 Reflexões e Direcionamento Pessoal")
st.write("Converse livremente e receba uma reflexão clara e útil.")

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
    if item.get("contexto") and item.get("mensagem") and len(item["mensagem"].strip()) > 20
]

texts = [item["contexto"] for item in data]
embeddings = model.encode(texts)

# =========================
# BUSCA REAL (CORRIGIDA)
# =========================
def buscar(pergunta):
    emb = model.encode([pergunta])[0]

    sim = np.dot(embeddings, emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(emb)
    )

    top_idx = np.argsort(sim)[-5:][::-1]

    for idx in top_idx:
        item = data[idx]
        if item.get("mensagem") and len(item["mensagem"].strip()) > 20:
            return item

    return None

# =========================
# LIMPEZA
# =========================
def limpar(texto):
    frase_ruim = "O crescimento interior exige compreensão dos próprios erros sem aprisionamento no passado."
    return texto.replace(frase_ruim, "").strip()

# =========================
# RESPOSTA FINAL
# =========================
def gerar(pergunta, item):

    if not item:
        return (
            "Entendo que você está passando por um momento difícil. Quando tudo parece pesado ao mesmo tempo, "
            "o mais importante é focar no próximo passo possível, mesmo que pequeno. Se quiser, me conta mais um pouco do que está acontecendo."
        )

    corpo = limpar(item["mensagem"])

    fechamento = (
        "Se quiser, me conta mais detalhes — isso ajuda a organizar melhor a situação e pensar em caminhos mais claros para você."
    )

    return f"{corpo}\n\n{fechamento}"

# =========================
# CHAT DISPLAY
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
