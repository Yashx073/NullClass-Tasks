import os
from dotenv import load_dotenv
import streamlit as st
from utils.response_generator import generate_response
import google.generativeai as genai

# Load environment variables
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not set in .env file!")

# Configure Gemini AI
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")  # Optional for advanced responses

# Streamlit UI
st.set_page_config(page_title="Sentiment-Aware Chatbot", layout="wide")
st.title("💬 Sentiment-Aware Chatbot")

# Chat history
if "history" not in st.session_state:
    st.session_state.history = []

# User input
user_input = st.text_input("You:")

if user_input:
    # Detect sentiment and generate response
    sentiment_response = generate_response(user_input)
    
    # Optional: Enhance response with Gemini AI
    try:
        gemini_response = model.generate_content(f"{user_input}\nSentiment aware context: {sentiment_response}")
        final_response = f"{sentiment_response}\n\n💡 Gemini says: {gemini_response.text}"
    except Exception:
        final_response = sentiment_response  # fallback if API fails
    
    # Save chat history
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", final_response))

# Display chat
for speaker, message in st.session_state.history:
    if speaker == "You":
        st.markdown(f"**You:** {message}")
    else:
        st.markdown(f"**Bot:** {message}")
