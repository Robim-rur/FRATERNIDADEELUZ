import streamlit as st
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Reflexões e Direcionamento V9", layout="centered")

st.title("📖 Reflexões e Direcionamento Pessoal V9")
st.write("Sistema com filtragem de contexto para respostas mais coerentes e sem ruído.")

# =========================
# CARREGAR BASE
# =========================
with open("base.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")

data = [item for item in raw_data if item.get("contexto")]

texts = [item["contexto"] for item in data]
embeddings = model.encode(texts)

# =========================
# CLASSIFICAÇÃO DE CONTEXTO (ESSENCIAL DO V9)
# =========================
def classificar_contexto(texto):
    t = texto.lower()

    if any(w in t for w in ["emprego", "dinheiro", "contas", "aluguel", "desempregado"]):
        return "material"

    if any(w in t for w in ["traição", "separação", "casamento", "relacionamento"]):
        return "afetivo"

    if any(w in t for w in ["triste", "desespero", "ansioso", "vazio"]):
        return "emocional"

    return "geral"

# =========================
# BUSCA COM FILTRO DE CONTEXO (CORE DO V9)
# =========================
def buscar_filtrado(pergunta, tipo):
    emb = model.encode([pergunta])[0]

    sim = np.dot(embeddings, emb) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(emb)
    )

    candidatos = []

    for i, item in enumerate(data):
        contexto = item["contexto"].lower()

        # FILTRO OBRIGATÓRIO POR TIPO
        if tipo == "material" and any(w in contexto for w in ["emprego", "dinheiro", "contas", "trabalho"]):
            candidatos.append((sim[i], item))

        elif tipo == "afetivo" and any(w in contexto for w in ["traição", "relacionamento", "casamento", "amor"]):
            candidatos.append((sim[i], item))

        elif tipo == "emocional" and any(w in contexto for w in ["triste", "desespero", "ansiedade", "vazio"]):
            candidatos.append((sim[i], item))

        elif tipo == "geral":
            candidatos.append((sim[i], item))

    # fallback se filtro ficar vazio
    if not candidatos:
        candidatos = list(zip(sim, data))

    # pega o melhor dentro do grupo correto
    candidatos.sort(reverse=True, key=lambda x: x[0])

    return candidatos[0][1]

# =========================
# TEXTO FINAL LIMPO
# =========================
def gerar_resposta(item):
    intro = (
        "Diante da situação que você está vivendo, é compreensível que surjam preocupações e sensação de pressão emocional, "
        "especialmente quando há responsabilidades importantes envolvidas."
    )

    corpo = item.get("mensagem", "")

    fechamento = (
        "Nessas situações, o mais importante é organizar os próximos passos de forma prática, sem se sobrecarregar com tudo de uma vez. "
        "Com o tempo, ações consistentes tendem a reorganizar a realidade de maneira mais estável e clara."
    )

    return f"{intro} {corpo} {fechamento}"

# =========================
# INPUT
# =========================
pergunta = st.text_area("🧠 Descreva sua situação")

if st.button("Analisar"):

    if not pergunta.strip():
        st.warning("Digite sua situação.")
    else:

        tipo = classificar_contexto(pergunta)
        item = buscar_filtrado(pergunta, tipo)

        resposta = gerar_resposta(item)

        st.markdown("## 📖 Reflexão e Direcionamento")
        st.success(resposta)
