import streamlit as st
from api.main import chat

st.set_page_config(page_title="Company RAG Chatbot")

st.title("🏢 Company Assistant (Advanced RAG)")

if "messages" not in st.session_state:
    st.session_state.messages = []

user_input = st.text_input("Ask your question:")

if st.button("Send"):
    if user_input:
        response = chat(user_input)

        st.session_state.messages.append(("You", user_input))
        st.session_state.messages.append(("Bot", response))

for role, msg in st.session_state.messages:
    if role == "You":
        st.markdown(f"**🧑 {msg}**")
    else:
        st.markdown(f"**🤖 {msg}**")