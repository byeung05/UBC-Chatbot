"""
Prompt + Gemini chat model composed with the hybrid retriever.
"""

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from .config import SETTINGS

def make_llm():
    return ChatGoogleGenerativeAI(
        model="models/gemini-1.5-flash",   # or "models/gemini-1.5-pro"
        google_api_key=SETTINGS.GEMINI_API_KEY,
        temperature=0.2
    )

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant for UBC grades. Use what you know from the data in the database. "
     "If the user requires you to make calculations, provide the sources and figures behind the calculations. If it is impossible to calculate with the given context, tell the user that"),
    ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer with relevant course/term citations.")
])

def format_docs(docs, max_docs=10):
    lines = []
    for d in docs[:max_docs]:
        c = d.metadata.get("course", "")
        t = d.metadata.get("term", d.metadata.get("year", ""))
        lines.append(f"- [{c} {t}] {d.page_content[:550]}")
    return "\n".join(lines)

def answer(llm, retriever, question: str) -> str:
    docs = retriever.invoke(question)
    msgs = PROMPT.format_messages(question=question, context=format_docs(docs))
    return llm.invoke(msgs).content
