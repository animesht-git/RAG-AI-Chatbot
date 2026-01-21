import streamlit as st
from rag_pipeline import rag_advanced

st.set_page_config(
    page_title="RAG AI Chatbot",
    layout="wide"
)

# ---------- SMALL TOP GAP ----------
st.markdown("<br>", unsafe_allow_html=True)

# ---------- CENTER IMAGE (BUILDING) ----------
col_left, col_center, col_right = st.columns([3, 4, 3])

with col_center:
    st.image(
        "static/ust building.png",
        use_container_width=True
    )

# ---------- VERY SMALL GAP ----------
st.markdown("<br>", unsafe_allow_html=True)

# ---------- SEARCH BAR ----------
q_left, q_center, q_right = st.columns([3, 6, 3])

with q_center:
    query = st.text_input(
        "",
        placeholder="Ask anything from your documents (Training Calendar, Manuals, PDFs...)",
        label_visibility="collapsed"
    )

# ---------- ANSWER ----------
if query:
    with st.spinner("Searching documents..."):
        answer = rag_advanced(query)
        st.markdown("### Answer")
        st.write(answer)

# ---------- BOTTOM BACKGROUND IMAGE ----------
st.markdown("<br><br>", unsafe_allow_html=True)

st.image(
    "static/background.jpg",
    use_container_width=True
)
