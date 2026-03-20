from pathlib import Path

import streamlit as st

from api.main import build_knowledge_base, chat

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Company RAG Chatbot")

st.title("Company Assistant (Advanced RAG)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "knowledge_base_ready" not in st.session_state:
    st.session_state.knowledge_base_ready = False

st.subheader("Upload PDFs")
uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.button("Process Files"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        saved_paths = []
        for uploaded_file in uploaded_files:
            target_path = UPLOAD_DIR / uploaded_file.name
            target_path.write_bytes(uploaded_file.getbuffer())
            saved_paths.append(str(target_path))

        with st.spinner("Building knowledge base..."):
            doc_count, chunk_count = build_knowledge_base(saved_paths)

        st.session_state.messages = []
        st.session_state.knowledge_base_ready = True
        st.success(
            f"Processed {len(uploaded_files)} file(s), {doc_count} page(s), and {chunk_count} chunk(s)."
        )

user_input = st.text_input("Ask your question:")

if st.button("Send"):
    if not st.session_state.knowledge_base_ready:
        st.warning("Upload and process at least one PDF first.")
    elif user_input:
        response = chat(user_input)
        st.session_state.messages.append(("You", user_input))
        st.session_state.messages.append(("Bot", response))

for role, msg in st.session_state.messages:
    if role == "You":
        st.markdown(f"**You:** {msg}")
    else:
        st.markdown(f"**Bot:** {msg}")
