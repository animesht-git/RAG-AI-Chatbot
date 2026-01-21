import os
import uuid
import requests
import numpy as np
import chromadb
import tempfile
import streamlit as st


from typing import List, Any

from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings


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
            "Content-Type": "application/json",
        },
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 1024,
        },
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

class EmbeddingManager:
    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def embed_documents(self, docs: List[Document]) -> np.ndarray:
        texts = [doc.page_content for doc in docs]
        return np.array(self.model.embed_documents(texts))

    def embed_query(self, query: str) -> np.ndarray:
        return np.array(self.model.embed_query(query))


# Vector Store 
class VectorStore:
    def __init__(
        self,
        collection_name: str = "pdf_documents",
    ):
        self.collection_name = collection_name
        self.persist_directory = os.path.join(
    os.getcwd(), "vector_store"
)

        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "PDF + manual document embeddings for RAG"},
        )
        print(f"Vector store initialized: {self.collection.count()} documents")

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        if not documents or len(embeddings) == 0:
            raise ValueError("Cannot add empty documents or embeddings")

        ids = []
        texts = []
        metadatas = []
        embedding_list = []

        for i, (doc, emb) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
            embedding_list.append(emb.tolist())

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embedding_list,
        )

        print(f"Added {len(documents)} documents to vector store")
#Ingestion Pipeline
DATA_DIR = "data/pdf"

def ingest_documents(vector_store: VectorStore, embedder: EmbeddingManager):
    documents: List[Document] = []

    # ---- Load PDFs + Word files ----
    if os.path.exists(DATA_DIR):
     for root, _, files in os.walk(DATA_DIR):
        for file in files:
            file_path = os.path.join(root, file)


            if file.lower().endswith(".pdf"):
                loader = PyMuPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                print(f"Loaded {len(docs)} pages from PDF: {file}")

            elif file.lower().endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                print(f"Loaded Word document: {file}")

    # ---- Manual fallback document ----
    documents.append(
        Document(
            page_content="main page content i will be using to create RAG",
            metadata={
                "source": "example.txt",
                "page": 1,
                "author": "Animesh",
                "date_created": "2026-01-16",
            },
        )
    )

    if not documents:
        raise RuntimeError("No documents found for ingestion")

    # ---- Clean + normalize ----
    cleaned_documents: List[Document] = []

    for doc in documents:
        text = doc.page_content.replace("\n", " ").replace("\t", " ")
        source = doc.metadata.get("source", "unknown")

        text = f"Document Name: {source}\n{text}"

        if len(text.strip()) < 10:
            continue

        cleaned_documents.append(
            Document(
                page_content=text,
                metadata=doc.metadata,
            )
        )

    print("TOTAL CLEANED DOCUMENTS:", len(cleaned_documents))

    # ---- Chunking ----
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(cleaned_documents)

    print("TOTAL CHUNKS:", len(chunks))

    # ---- Embeddings ----
    embeddings = embedder.embed_documents(chunks)

    # ---- Store ----
    vector_store.add_documents(chunks, embeddings)



class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedder: EmbeddingManager):
        self.collection = vector_store.collection
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 3):
        query_embedding = self.embedder.embed_query(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas"],
        )

        retrieved = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            retrieved.append(
                {
                    "content": doc,
                    "source": meta.get("source", "unknown"),
                    "page": meta.get("page", "N/A"),
                }
            )

        return retrieved


@st.cache_resource
def get_embedding_manager():
    return EmbeddingManager()

embedding_manager = get_embedding_manager()

@st.cache_resource
def get_vector_store():
    return VectorStore()

vector_store = get_vector_store()
print("VECTOR STORE COUNT:", vector_store.collection.count())


ingest_documents(vector_store, embedding_manager)


rag_retriever = RAGRetriever(vector_store, embedding_manager)


def rag_advanced(query: str) -> str:
    results = rag_retriever.retrieve(query, top_k=5)

    print("RETRIEVED CHUNKS COUNT:", len(results))
    for i, r in enumerate(results):
        print(f"\n--- CHUNK {i+1} ---")
        print(r["content"][:500])

    if not results:
        return "NO DOCUMENTS RETRIEVED"

    context = "\n\n".join(r["content"] for r in results)

    prompt = f"""
You are answering questions using retrieved document excerpts.

If the question asks generally about a document (e.g. "what is mentioned"),
summarize the relevant information found in the context.

Do NOT hallucinate. Use only the context.

Context:
{context}

Question:
{query}

Answer:
"""

    return llm(prompt)
