# rag_pipeline.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# =============================
# Load environment variables
# =============================
load_dotenv()

# =============================
# LLM
# =============================
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant",
    temperature=0.2,
    max_tokens=1024
)

# =============================
# IMPORT your existing classes
# (paste them here EXACTLY as is)
# =============================
# EmbeddingManager
# VectorStore
# RAGRetriever
# rag_advanced
# rag_simple

# =============================
# Initialize objects
# =============================
embedding_manager = EmbeddingManager()
vector_store = VectorStore()
rag_retriever = RAGRetriever(vector_store, embedding_manager)
