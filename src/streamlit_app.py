import os, streamlit as st
from pinecone import Pinecone
from dotenv import load_dotenv

from src.config import SETTINGS, require_env
from src.tfidf import load_vectorizer
from src.retriever import PineconeHybridRetriever
from src.rag_chain import make_llm, answer

load_dotenv()
require_env()

st.set_page_config(page_title="UBC Grades RAG")
st.title("UBC Grades Chatbot (Hybrid: Gemini + TF-IDF)")

# Pinecone client + index
pc = Pinecone(api_key=SETTINGS.PINECONE_API_KEY)
index = pc.Index(SETTINGS.HYBRID_INDEX_NAME)

# Load TF-IDF vectorizer persisted by the indexing step
vectorizer = load_vectorizer()

# Controls
alpha = st.sidebar.slider("Hybrid α (semantic ↔ lexical)", 0.0, 1.0, 0.6, 0.05)
top_k = st.sidebar.slider("Top-K", 5, 30, 10, 1)
dept = st.sidebar.text_input("Filter dept (e.g., CPSC)")
year_min = st.sidebar.number_input("Min year", value=2018)

flt = {"dept": {"$eq": dept}} if dept else None
if year_min:
    flt = (flt or {}) | {"year": {"$gte": int(year_min)}}

retriever = PineconeHybridRetriever(index, vectorizer, alpha=alpha, top_k=top_k, flt=flt)
llm = make_llm()

if "chat" not in st.session_state: st.session_state.chat = []
q = st.chat_input("Ask about courses, terms, instructors…")
if q:
    st.session_state.chat.append(("user", q))
    with st.spinner("Thinking…"):
        a = answer(llm, retriever, q)
    st.session_state.chat.append(("assistant", a))

for role, msg in st.session_state.chat:
    with st.chat_message(role):
        st.write(msg)
