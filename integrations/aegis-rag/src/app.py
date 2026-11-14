import asyncio
import os
import sys

import requests
import streamlit as st

# --- Windows Asyncio Fix for Streamlit ---
# Prevents spammy "WinError 10054" ConnectionResetErrors when the browser reloads or disconnects
if sys.platform == 'win32':
    try:
        # Modern Python 3.14+ way to globally swap the default loop type
        # No policies, no deprecation warnings, entirely future-proof
        asyncio.DefaultEventLoop = asyncio.SelectorEventLoop
    except Exception:
        pass

# Configuration
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Aegis Agent", page_icon="🛡️", layout="centered")

st.markdown("<h1 style='text-align: center; color: #4A90E2;'>🛡️ Aegis Omnisearch Agent</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; opacity: 0.7;'>Ask me anything. I can search my memory and compute math.</p>",
    unsafe_allow_html=True)
st.markdown("---")

# --- SIDEBAR FOR AUTH & FILE UPLOADS ---
with st.sidebar:
    st.header("🔑 Authentication")
    api_key = st.text_input("Gemini API Key", type="password",
                            help="Required to chat. Get a free key at aistudio.google.com")
    st.caption("Your key is never stored. It is used statelessly.")

    st.markdown("---")

    st.header("📚 Add Knowledge")
    st.markdown("Upload documents to teach Aegis new information.")
    uploaded_file = st.file_uploader("Upload a TXT or PDF document", type=["txt", "pdf"])

    if st.button("Learn Document") and uploaded_file is not None:
        with st.spinner("Reading, chunking, and embedding document..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                res = requests.post(f"{API_URL}/upload", files=files)
                if res.status_code == 200:
                    st.success(res.json().get("message", "Success!"))
                else:
                    st.error(f"Error: {res.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API. Is it running?")

# --- MAIN CHAT INTERFACE ---
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am Aegis. Please enter your Gemini API Key in the sidebar to begin."}
    ]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Enter your request here..."):

    if not api_key:
        st.warning("⚠️ Please enter your Gemini API Key in the sidebar first.")
    else:
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Aegis is reasoning, searching, and acting..."):
                try:
                    payload = {
                        "message": prompt,
                        "api_key": api_key
                    }
                    response = requests.post(f"{API_URL}/chat", json=payload)

                    if response.status_code == 200:
                        answer = response.json()["answer"]
                        st.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        error_msg = f"API Error: {response.text}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})

                except requests.exceptions.ConnectionError:
                    error_msg = "Could not connect to the API. Is the FastAPI backend running?"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
