import streamlit as st
from rag_pipeline import rag_advanced

st.set_page_config(
    page_title="RAG AI Chatbot",
    page_icon="🚀",
    layout="centered"
)

# --- CSS to center content ---
st.markdown(
    """
    <style>
    .center-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 70vh;
    }
    .search-box {
        width: 100%;
        max-width: 600px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Centered layout ---
st.markdown('<div class="center-container">', unsafe_allow_html=True)

st.markdown("🧠 UST AI Assistant")
st.markdown("Ask anything and get an AI-powered answer")

query = st.text_input(
    "",
    placeholder="Type your question here...",
    key="search",
    label_visibility="collapsed"
)

st.markdown('</div>', unsafe_allow_html=True)

# --- Answer section ---
if query:
    with st.spinner("Thinking..."):
        answer = rag_advanced(query)

    st.markdown("---")
    st.markdown("### 📄 Answer")
    st.write(answer)
