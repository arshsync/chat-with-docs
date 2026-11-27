import streamlit as st
from rag import add_document, answer_question, clear_my_data, has_documents

st.title("Chat with your documents")

import uuid

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
session_id = st.session_state.session_id

# session_state keeps values between reruns (Streamlit reruns the file on every click)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "indexed" not in st.session_state:
    st.session_state.indexed = []


if st.button("Delete My data"):
    clear_my_data(session_id)
    st.session_state.indexed = []
    st.session_state.messages = []
    st.rerun()

with st.sidebar:
    uploaded = st.file_uploader("Upload a PDF or DOCX", type=["pdf", "docx"])
    if uploaded and uploaded.name not in st.session_state.indexed:
        with st.spinner("Reading and indexing the file..."):
            count = add_document(uploaded, session_id)
        st.session_state.indexed.append(uploaded.name)
        if count == 0:
            st.error("No text found in this file.")
        else:
            st.success(f"Indexed {count} chunks")

if not has_documents(session_id):
    st.info("Please upload a PDF or DOCX file from the sidebar to start chatting.")
    st.stop()

# show old messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# chat_input shows the text box at the bottom and returns the text after Enter
question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = answer_question(question, session_id)
        st.write(answer)
        with st.expander("Sources"):
            for s in sources:
                st.caption(f"{s.payload['source']} | score {s.score:.2f}")
                st.write(s.payload["text"])

    st.session_state.messages.append({"role": "assistant", "content": answer})