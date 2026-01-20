import os
import requests
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --------------------
# LLM (Groq via HTTP)
# --------------------
try:
    import streamlit as st
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found")

def llm(prompt: str) -> str:
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1024
        }
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# --------------------
# Embeddings
# --------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --------------------
# Vector store builder
# --------------------
DATA_DIR = "data"
PERSIST_DIR = "chroma_db"

def build_vector_store():
    docs = []

    for file in os.listdir(DATA_DIR):
        if file.lower().endswith(".pdf"):
            loader = PyMuPDFLoader(os.path.join(DATA_DIR, file))
            docs.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vectordb.persist()
    return vectordb

vectordb = build_vector_store()

# --------------------
# Retriever
# --------------------
class RAGRetriever:
    def retrieve(self, query, top_k=3):
        docs = vectordb.similarity_search(query, k=top_k)
        return [
            {
                "content": d.page_content,
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page", "N/A")
            }
            for d in docs
        ]

rag_retriever = RAGRetriever()

# --------------------
# RAG function (your rag_simple, fixed)
# --------------------
def rag_advanced(query, top_k=3):
    results = rag_retriever.retrieve(query, top_k=top_k)

    if not results:
        return "I couldn't find this information in the uploaded documents."

    context = "\n\n".join(r["content"] for r in results)

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question:
{query}

Answer:
"""

    return llm(prompt)
