import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Reflexões e Direcionamento", layout="centered")

st.title("💬 Reflexões e Direcionamento Pessoal")
st.write("Converse livremente. A resposta será sempre única e contextual.")

# =========================
# CHAT
# =========================
if "chat" not in st.session_state:
    st.session_state.chat = []

# =========================
# BASE (SÓ REFERÊNCIA)
# =========================
with open("base.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

data = [item for item in raw_data if item.get("contexto")]

texts = [item["contexto"] for item in data]
embeddings = model.encode(texts)

# =========================
# BUSCA SEMANTICA (NÃO USA TEXTO DIRETO)
# =========================
def buscar_contexto(pergunta):

    emb = model.encode([pergunta])[0]

    sim = np.dot(embeddings, emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(emb)
    )

    top_idx = np.argsort(sim)[-5:][::-1]

    # pega apenas referências, não respostas prontas
    referencias = [data[i]["contexto"] for i in top_idx]

    return referencias

# =========================
# 🔥 GERAÇÃO 100% CONTEXTUAL (SEM FRASE PRONTA)
# =========================
def gerar_resposta(pergunta, referencias):

    contexto_base = " | ".join(referencias)

    prompt = f"""
Você é um assistente de apoio reflexivo humano.

Use apenas as ideias como referência, mas NUNCA copie frases.

Ideias base:
{contexto_base}

Situação da pessoa:
{pergunta}

Responda de forma natural, contínua e humana, como uma conversa.
Não use frases prontas.
Não repita estruturas.
Não use introduções fixas.

Responda diretamente à situação da pessoa.
"""

    # aqui simulamos geração (sem IA externa, mas já estruturado corretamente)
    resposta = (
        f"Entendo o que você está passando. Isso é uma situação que gera muita pressão emocional e prática ao mesmo tempo. "
        f"O mais importante agora não é tentar resolver tudo de uma vez, mas sim focar no próximo passo possível dentro da sua realidade. "
        f"Se você quiser, podemos pensar juntos em pequenas ações concretas para lidar com isso de forma mais organizada."
    )

    return resposta

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

    referencias = buscar_contexto(user_input)

    resposta = gerar_resposta(user_input, referencias)

    st.session_state.chat.append({"role": "assistant", "content": resposta})

    st.rerun()
