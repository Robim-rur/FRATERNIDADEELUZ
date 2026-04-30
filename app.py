import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Reflexões e Direcionamento", layout="centered")

st.title("💬 Reflexões e Direcionamento Pessoal")
st.write("Converse livremente. Vou te ajudar a organizar seus pensamentos de forma clara e prática.")

# =========================
# CHAT MEMORY
# =========================
if "chat" not in st.session_state:
    st.session_state.chat = []

# =========================
# LOAD BASE
# =========================
with open("base.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

# FILTRA APENAS ITENS COMPLETOS (CORREÇÃO PRINCIPAL)
data = []
for item in raw_data:
    if item.get("contexto") and item.get("mensagem"):
        if len(item["mensagem"].strip()) > 10:
            data.append(item)

texts = [item["contexto"] for item in data]
embeddings = model.encode(texts)

# =========================
# BUSCA SEGURA
# =========================
def buscar(pergunta):
    emb = model.encode([pergunta])[0]

    sim = np.dot(embeddings, emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(emb)
    )

    idx_sorted = np.argsort(sim)[::-1]

    # pega TOP 3 para segurança
    for idx in idx_sorted[:3]:
        item = data[idx]
        if item.get("mensagem") and len(item["mensagem"].strip()) > 10:
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
            "Entendi o que você está passando. Às vezes, quando tudo parece difícil ao mesmo tempo, "
            "o mais importante é focar no próximo passo possível, sem tentar resolver tudo de uma vez.\n\n"
            "Se quiser, me conta um pouco mais da sua situação para eu te ajudar melhor."
        )

    corpo = limpar(item["mensagem"])

    if not corpo:
        corpo = "Entendi sua situação. Vamos olhar isso com calma e clareza para encontrar o melhor próximo passo."

    fechamento = (
        "Se quiser, pode continuar me contando — isso ajuda a organizar melhor o que você está vivendo e pensar em soluções mais claras."
    )

    return f"{corpo}\n\n{fechamento}"

# =========================
# DISPLAY CHAT
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
