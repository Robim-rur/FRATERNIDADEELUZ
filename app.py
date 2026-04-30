import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# CONFIGURAÇÃO
# =========================
st.set_page_config(page_title="Reflexões e Direcionamento Pessoal", layout="centered")

st.title("📖 Reflexões e Direcionamento Pessoal")
st.write("Descreva sua situação e receba uma reflexão clara, humana e prática para te ajudar a organizar seus pensamentos.")

# =========================
# CARREGAR BASE
# =========================
with open("base.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

# =========================
# LIMPEZA DA BASE (ANTI-RUÍDO)
# =========================
data = []

for item in raw_data:
    contexto = item.get("contexto", "").strip()
    mensagem = item.get("mensagem", "").strip()

    if contexto and mensagem:
        data.append(item)

texts = [item["contexto"] for item in data]
embeddings = model.encode(texts)

# =========================
# DETECÇÃO DE TIPO DE SITUAÇÃO
# =========================
def detectar_tipo(texto):
    t = texto.lower()

    if any(w in t for w in ["emprego", "dinheiro", "desempregado", "contas", "aluguel"]):
        return "crise_material"

    if any(w in t for w in ["traição", "separação", "relacionamento", "casamento"]):
        return "crise_afetiva"

    if any(w in t for w in ["triste", "desespero", "vazio", "ansioso"]):
        return "crise_emocional"

    return "geral"

# =========================
# BUSCA INTELIGENTE (SEM RUÍDO)
# =========================
def buscar_melhor(pergunta):
    emb = model.encode([pergunta])[0]

    sim = np.dot(embeddings, emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(emb)
    )

    # pega só o melhor resultado (zero poluição)
    idx = np.argmax(sim)

    return data[idx]

# =========================
# GERAÇÃO DE TEXTO LIMPO (SEM FRAGMENTOS)
# =========================
def gerar_texto(item, pergunta):

    abertura = (
        "Diante da situação que você está enfrentando, é natural que surjam sentimentos de preocupação, insegurança e sobrecarga emocional, "
        "especialmente quando há responsabilidades importantes envolvidas."
    )

    corpo = item.get("mensagem", "")

    fechamento = (
        "Em momentos como este, o mais importante é organizar o pensamento, focar no que pode ser feito no presente e evitar decisões tomadas apenas pela emoção imediata. "
        "Com o tempo, situações difíceis tendem a se reorganizar quando enfrentadas com calma, clareza e ação prática."
    )

    return f"{abertura} {corpo} {fechamento}"

# =========================
# INPUT
# =========================
pergunta = st.text_area("🧠 Descreva sua situação")

if st.button("Receber análise"):

    if not pergunta.strip():
        st.warning("Por favor, descreva sua situação.")
    else:

        tipo = detectar_tipo(pergunta)
        item = buscar_melhor(pergunta)

        resposta = gerar_texto(item, pergunta)

        st.markdown("## 📖 Reflexão e Direcionamento")
        st.success(resposta)
