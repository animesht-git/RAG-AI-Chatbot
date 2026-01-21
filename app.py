import streamlit as st
from rag_pipeline import rag_advanced

st.set_page_config(
    page_title="RAG AI Chatbot",
    page_icon="🚀",
    layout="centered"
)

# --- CSS ---
st.markdown(
    """
    <style>
    .center-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-start;
        margin-top: 15px;
    }

    .title {
        font-size: 28px;
        font-weight: 600;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 16px;
        color: #666;
        margin-bottom: 20px;
    }

    .search-box {
        width: 100%;
        max-width: 600px;
    }

    .bottom-bg {
    margin-top: 40px;
    height: 70vh;     
    overflow: hidden;
}

    </style>
    """,
    unsafe_allow_html=True
)

# --- CENTER CONTENT ---
st.markdown('<div class="center-container">', unsafe_allow_html=True)

# 🔹 BIG CENTER IMAGE (BUILDING)
st.image(
    "static/ust building.png",
    width=520   
)


# 🔹 TITLE TEXT 
st.markdown('<div class="title">🧠 UST AI Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask anything and get an AI-powered answer</div>', unsafe_allow_html=True)

# 🔹 SEARCH BAR 
query = st.text_input(
    "",
    placeholder="Hey! How may I assist you today?",
    key="search",
    label_visibility="collapsed"
)

st.markdown('</div>', unsafe_allow_html=True)

# --- ANSWER  ---
if query:
    with st.spinner("Cooking..."):
        answer = rag_advanced(query)

    st.markdown("---")
    st.markdown("### 📄 Answer")
    st.write(answer)

# --- BOTTOM BACKGROUND IMAGE---
st.markdown('<div class="bottom-bg">', unsafe_allow_html=True)
st.image(
    "static/background.png",
    width="stretch"
)


st.markdown('</div>', unsafe_allow_html=True)
