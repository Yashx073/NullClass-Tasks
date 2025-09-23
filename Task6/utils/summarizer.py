def summarize_text(text, max_words=50):
    # Simple truncation; can integrate LLM summarization
    return " ".join(text.split()[:max_words]) + "..."
