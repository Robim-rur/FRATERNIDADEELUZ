import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# CONFIGURAÇÃO GERAL
# =========================
st.set_page_config(page_title="Reflexões e Direcionamento", layout="centered")

st.title("💬 Reflexões e Direcionamento Pessoal")
st.write("Converse livremente. Vou te ajudar a organizar seus pensamentos de forma clara e prática.")

# =========================
# HISTÓRICO DE CHAT
# =========================
if "chat" not in st.session_state:
    st.session_state.chat = []

# =========================
# CARREGAR BASE
# =========================
with open("base.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

# filtra apenas itens válidos
data = [item for item in raw_data if item.get("contexto") and item.get("mensagem")]

texts = [item["contexto"] for item in data]
embeddings = model.encode(texts)

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
# LIMPEZA DE RUÍDO (CORREÇÃO REAL)
# =========================
def limpar_texto(texto):
    # remove frase problemática específica caso exista na base
    frase_ruim = "O crescimento interior exige compreensão dos próprios erros sem aprisionamento no passado."
    return texto.replace(frase_ruim, "").strip()

# =========================
# GERAÇÃO FINAL (SEM RUÍDO)
# =========================
def gerar_resposta(pergunta, item):

    corpo = limpar_texto(item.get("mensagem", ""))

    fechamento = (
        "Se quiser, pode continuar me contando — entender melhor o contexto ajuda a organizar melhor os próximos passos de forma mais clara."
    )

    return f"{corpo}\n\n{fechamento}"

# =========================
# EXIBIR CHAT
# =========================
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =========================
# INPUT ESTILO HUMANO
# =========================
user_input = st.chat_input("Me conte um pouco do que está acontecendo com você atualmente.")

if user_input:

    # salva mensagem do usuário
    st.session_state.chat.append({"role": "user", "content": user_input})

    # busca resposta
    item = buscar(user_input)
    resposta = gerar_resposta(user_input, item)

    # salva resposta
    st.session_state.chat.append({"role": "assistant", "content": resposta})

    st.rerun()
