import streamlit as st
from utils.embedder import Embedder
from utils.search import retrieve_papers
from utils.summarizer import summarize_text  # optional if you want local summaries
import google.generativeai as genai
import os
from dotenv import load_dotenv

# -----------------------------
# Load API key
# -----------------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY not set in .env file!")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# -----------------------------
# Load knowledge base
# -----------------------------
embedder = Embedder()
vectors, documents = embedder.load_vectors()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📚 ArXiv Domain Expert Chatbot - Computer Science")

query = st.text_input("Enter your question about computer science papers:")

if query:
    # Retrieve top 3 papers
    top_papers = retrieve_papers(query, embedder, top_k=3)
    
    if not top_papers:
        st.warning("⚠️ No relevant papers found for your query.")
    else:
        st.subheader("🔍 Top Relevant Papers")
        for idx, paper in enumerate(top_papers, 1):
            st.write(f"{idx}. {paper[:300]}...")  # Show snippet

        # Summarize & explain using Gemini
        combined_text = "\n\n".join(top_papers)
        prompt = f"Summarize and explain the following content:\n\n{combined_text}"
        
        try:
            response = model.generate_content(prompt)
            st.subheader("💡 Bot Explanation")
            st.write(response.text)
        except Exception as e:
            st.error(f"Error generating explanation: {str(e)}")
