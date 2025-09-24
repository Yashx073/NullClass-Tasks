🧠 AI Chatbot Projects Repository

This repository contains multiple AI-powered chatbot and NLP projects developed as part of my internship. Each task builds on advanced natural language processing (NLP), information retrieval, and AI integration techniques.

📌 Project Tasks
Task 1 – Extractive Summarization

Goal: Implement an extractive summarization tool.

Approach: Uses NLP techniques to select and combine the most important sentences from longer documents.

Outcome: Generates concise and meaningful summaries for quick understanding.

Task 2 – Medical Q&A Chatbot (MedQuAD Dataset)

Goal: Develop a specialized medical chatbot using the MedQuAD dataset
.

Features:

Retrieval-based Q&A system.

Basic medical entity recognition (symptoms, diseases, treatments).

User interface via Streamlit for medical question answering.

Outcome: A chatbot capable of providing reliable medical answers.

Task 3 – Multi-Modal Chatbot

Goal: Extend chatbot capabilities to handle text + image inputs.

Technologies: Google Gemini (1.5 Flash/Pro).

Features:

Understands user queries with both text and images.

Generates context-aware responses.

Can describe, analyze, or generate image content.

Outcome: A chatbot that seamlessly integrates visual and textual conversation.

Task 4 – Dynamic Knowledge Base Expansion

Goal: Implement a system to dynamically expand the chatbot’s knowledge base.

Approach:

Uses vector databases (TF-IDF embeddings).

Periodically updates from specified sources.

Outcome: A chatbot that continuously improves by incorporating new information.

Task 5 – Multilingual Chatbot

Goal: Extend chatbot to support multiple languages.

Features:

Detects user’s input language automatically.

Provides culturally appropriate responses.

Supports at least 3+ additional languages beyond English.

Outcome: A truly multilingual AI assistant with enhanced language understanding.

Task 6 – Domain Expert Chatbot (ArXiv Papers)

Goal: Create a domain-specific chatbot trained on a subset of the arXiv dataset
.

Focus Area: Computer Science research papers.

Features:

Retrieval of relevant research papers.

Advanced summarization of academic text.

Concept explanation with an open-source LLM + Gemini.

Implemented with Streamlit for searching, summarizing, and visualizing papers.

Outcome: A chatbot capable of discussing complex research topics and providing meaningful explanations.

Task 7 – Sentiment-Aware Chatbot

Goal: Integrate sentiment analysis into chatbot conversations.

Features:

Detects positive, negative, or neutral sentiment in user input.

Responds in an emotionally appropriate manner.

Improves user satisfaction by adapting tone and style.

Outcome: A chatbot with emotional intelligence, enhancing customer interaction quality.

⚙️ Technologies Used

Python

Streamlit / Gradio for UI

scikit-learn (TF-IDF, similarity)

Google Generative AI (Gemini)

Pandas, NumPy, Matplotlib

NLTK / SpaCy for NLP

Sentiment Analysis Models (VADER/TextBlob/Transformers)

🚀 How to Run

Clone the repository:

git clone https://github.com/your-username/ai-chatbot-projects.git
cd ai-chatbot-projects


Create a virtual environment & install dependencies:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Set up your .env file with API keys (e.g., Google Gemini):

GOOGLE_API_KEY=your_api_key_here


Run individual projects:

Task 1: python Task1/summarizer.py

Task 2: streamlit run Task2/app.py

Task 3: python Task3/app.py

Task 4: python utils/update_kb.py then python Task4/app.py

Task 5: python Task5/app.py

Task 6: streamlit run Task6/app.py

Task 7: streamlit run Task7/app.py

📊 Learning Outcomes

Built end-to-end AI chatbots for real-world use cases.

Gained experience with NLP, information retrieval, embeddings, and summarization.

Learned multi-modal AI, multilingual support, and sentiment-aware design.

Hands-on with Google Gemini API, Streamlit, and vector databases.
