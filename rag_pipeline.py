# =============================
# Environment setup
# =============================
import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_PATH)

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY not found in environment")

# =============================
# LLM (Groq)
# =============================
import requests

# Support Streamlit Cloud secrets + local env
try:
    import streamlit as st
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
except Exception:
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not found")

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1024,
    }

    response = requests.post(GROQ_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


print("LLM INITIALIZED")

# =============================
# Embeddings
# =============================
from langchain_community.embeddings import HuggingFaceEmbeddings

class EmbeddingManager:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def get(self):
        return self.embeddings

embedding_manager = EmbeddingManager()
print("EMBEDDINGS READY")

# =============================
# TEMP STOP HERE
# =============================
# DO NOT add VectorStore / RAG yet
# First confirm this file runs cleanly
