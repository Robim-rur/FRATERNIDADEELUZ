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
# MEMÓRIA
# =========================
if "chat" not in st.session_state:
    st.session_state.chat = []

if "fase" not in st.session_state:
    st.session_state.fase = "inicio"

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
# CLASSIFICAR FASE
# =========================
def detectar_fase(texto):

    t = texto.lower()

    if any(w in t for w in ["o que devo fazer", "como resolver", "agora", "?"]):
        return "acao"

    if any(w in t for w in ["desespero", "fome", "sem dinheiro", "urgente"]):
        return "crise"

    return "relato"

# =========================
# BUSCA INTELIGENTE
# =========================
def buscar(pergunta):

    emb = model.encode([pergunta])[0]

    sim = np.dot(embeddings, emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(emb)
    )

    idx = np.argmax(sim)

    return data[idx]

# =========================
# RESPOSTA POR FASE (AQUI ESTÁ A SOLUÇÃO REAL)
# =========================
def gerar_resposta(pergunta, item, fase):

    corpo = item["mensagem"]

    if fase == "relato":
        retorno = (
            "Entendi o que você está passando. Vamos organizar isso com calma para ficar mais claro o próximo passo."
        )

    elif fase == "acao":
        retorno = (
            "Agora o mais importante é transformar isso em ações práticas e possíveis no curto prazo. Vamos focar no que pode ser feito primeiro."
        )

    elif fase == "crise":
        retorno = (
            "Quando a situação envolve urgência, o foco precisa ser totalmente em soluções imediatas e apoio concreto no curto prazo."
        )

    else:
        retorno = ""

    fechamento = "Se quiser, continue me contando — vou te acompanhando nisso passo a passo."

    # 🔥 EVITA REPETIÇÃO: não repete mensagem igual sempre
    if st.session_state.chat and retorno in st.session_state.chat[-1]["content"]:
        retorno = "Vamos continuar a partir disso de forma mais prática para sua situação."

    return f"{corpo}\n\n{retorno}\n\n{fechamento}"

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

    fase = detectar_fase(user_input)
    st.session_state.fase = fase

    item = buscar(user_input)

    resposta = gerar_resposta(user_input, item, fase)

    st.session_state.chat.append({"role": "assistant", "content": resposta})

    st.rerun()
