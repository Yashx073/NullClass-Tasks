# Task4 - Dynamic Knowledge Base Chatbot

## Description
This project implements a multi-modal chatbot that dynamically updates its knowledge base using new information sources. It integrates Gemini 1.5-flash for language understanding and a vector database for context-aware responses.

## Structure
- `app.py` → Main chatbot Gradio interface.
- `notebook.ipynb` → Demo and experiments.
- `data/` → Sources for knowledge base updates.
- `knowledge_base/` → Saved vector embeddings.
- `models/` → Optional ML/NLP model weights.
- `utils/` → Helper scripts to update knowledge base.

## How to Run
1. Set your `GOOGLE_API_KEY` in `.env`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Run `python app.py` to launch the chatbot.
