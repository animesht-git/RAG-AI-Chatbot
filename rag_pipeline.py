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
from groq import Groq
import os

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


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
