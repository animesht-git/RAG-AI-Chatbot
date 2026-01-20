import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
import streamlit as st

# Import your existing RAG components
# These should already be initialized exactly as in your notebook
from rag_pipeline import rag_advanced, rag_retriever, llm


# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="RAG PDF Assistant",
    page_icon="📄",
    layout="wide"
)

# -------------------------------
# Custom CSS (Professional look)
# -------------------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .chat-title {
        font-size: 2rem;
        font-weight: 700;
    }
    .confidence-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        background-color: #eef2ff;
    }
</style>
""", unsafe_allow_html=True)
