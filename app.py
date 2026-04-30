import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import random

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

data = [item for item in raw_data if item.get("contexto") and item.get("mensagem")]

texts = [item["contexto"] for item in data]
embeddings = model.encode(texts)

# =========================
# CLASSIFICAÇÃO SIMPLES DE INTENÇÃO
# =========================
def classificar(texto):

    t = texto.lower()

    if any(x in t for x in ["emprego", "dinheiro", "trabalho"]):
        return "financeiro"

    if any(x in t for x in ["fome", "comer", "filho", "casa"]):
        return "familiar"

    if any(x in t for x in ["desespero", "ansioso", "triste"]):
        return "emocional"

    return "geral"

# =========================
# BUSCA MELHORADA
# =========================
def buscar(pergunta):

    emb = model.encode([pergunta])[0]

    sim = np.dot(embeddings, emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(emb)
    )

    top_idx = np.argsort(sim)[-7:][::-1]

    return data[random.choice(top_idx)]

# =========================
# GERADOR DE RESPOSTA EVOLUÍDO
# =========================
def gerar_resposta(pergunta, item):

    tipo = classificar(pergunta)

    base = item["mensagem"]

    if tipo == "financeiro":
        ajuste = "O mais importante agora é focar em soluções práticas e imediatas, sem se sobrecarregar com tudo de uma vez."

    elif tipo == "familiar":
        ajuste = "Quando envolve família, o impacto emocional é maior, mas o próximo passo precisa ser simples e possível dentro da sua realidade."

    elif tipo == "emocional":
        ajuste = "Nesse momento, o mais importante é reduzir a pressão interna e organizar um passo de cada vez."

    else:
        ajuste = "Vamos organizar isso de forma simples e clara para você conseguir enxergar melhor a situação."

    finais = [
        "Se quiser, me conta mais detalhes.",
        "Se quiser, posso te ajudar a organizar melhor isso.",
        "Se quiser, seguimos passo a passo juntos.",
        "Se quiser, me diga como isso está acontecendo hoje."
    ]

    return f"{base} {ajuste} {random.choice(finais)}"

# =========================
# CHAT DISPLAY
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
