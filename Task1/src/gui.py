"""src/gui.py

Simple Streamlit GUI to paste a document and get an extractive summary.
"""

import streamlit as st
from summarizer import ExtractiveSummarizer

# Page settings
st.set_page_config(page_title="Extractive Summarizer", layout="centered")
st.title("📝 Extractive Summarizer")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    num_sentences = st.slider("Number of sentences", 1, 10, 3)

# Main text area
text = st.text_area("Paste your document here (or load from sample)", height=300)

# Summarize button
if st.button("Summarize"):
    if not text.strip():
        st.warning("⚠️ Please paste some text first.")
    else:
        summarizer = ExtractiveSummarizer()
        summary = summarizer.summarize(text, num_sentences=num_sentences)
        st.subheader("📌 Summary")
        st.write(summary)
