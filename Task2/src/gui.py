# src/gui.py

import streamlit as st
import sys
import os
import time

# --- PATH CORRECTION TO FIX IMPORTERROR ---
# This block allows Streamlit to find the 'src' package when run from the root directory.
current_dir = os.path.dirname(__file__)
# Go up one directory from 'src' to the project root ('Task2')
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# --- CORRECTED IMPORT ---
# Use the absolute import path from the project root.
from src.chatbot import MedicalChatbot 


@st.cache_resource
def load_chatbot():
    """Loads and caches the chatbot to prevent re-initialization on every run."""
    try:
        # The TfidfRetriever initialization is resource-intensive
        return MedicalChatbot()
    except FileNotFoundError:
        st.error("Model or data files not found! Please run the medqa_training.ipynb notebook first to create models/ and data/ files.")
        return None

# --- Streamlit App ---
st.set_page_config(page_title="MedQuAD Q&A Chatbot", layout="wide")

st.title("👨‍⚕️ Medical Q&A Chatbot")
st.markdown("Ask a medical question and the bot will retrieve the best answer from the MedQuAD knowledge base.")

chatbot = load_chatbot()

if chatbot:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a medical question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get bot response
        with st.chat_message("assistant"):
            with st.spinner("Searching MedQuAD knowledge base..."):
                time.sleep(1) # Simulate processing time
                response, entities = chatbot.get_response(prompt)
                
                # Display the main response
                st.markdown(response)
                
                # Display NER info
                if entities:
                    with st.expander("Entity Recognition Report"):
                        st.text(entities)

            # Add bot response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})