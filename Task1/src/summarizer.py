"""src/summarizer.py

Extractive summarizer implementation using TF-IDF + cosine similarity sentence ranking.
"""

import os
import re
import nltk
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download punkt tokenizer for sentence splitting
nltk.download("punkt", quiet=True)


def clean_sentence(sent: str) -> str:
    """Clean up whitespace and formatting in a sentence."""
    sent = re.sub(r"\s+", " ", sent).strip()
    return sent


class ExtractiveSummarizer:
    def __init__(self, vectorizer: TfidfVectorizer = None,
                 model_path: str = "Task1/models/vectorizer.pkl"):
        """
        Initialize summarizer.
        If vectorizer is not provided, try loading from model_path.
        """
        if vectorizer is not None:
            self.vectorizer = vectorizer
        elif os.path.exists(model_path):
            self.vectorizer = joblib.load(model_path)
            print(f"[INFO] Loaded vectorizer from {model_path}")
        else:
            self.vectorizer = None
            print("[WARN] No pre-trained vectorizer found, will fit a new one.")

    def fit_vectorizer(self, sentences):
        """Fit a TF-IDF vectorizer on given sentences."""
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(sentences)
        return self.vectorizer

    def score_sentences(self, sentences):
        """Score sentences using cosine similarity matrix (TextRank-like)."""
        if self.vectorizer is None:
            self.fit_vectorizer(sentences)

        # Transform sentences
        tfidf_matrix = self.vectorizer.transform(sentences)
        sim_matrix = cosine_similarity(tfidf_matrix)

        # Sentence scores: sum of similarity values (excluding self-similarity)
        scores = sim_matrix.sum(axis=1) - 1  # subtract self similarity
        return scores

    def summarize(self, text: str, num_sentences: int = 3) -> str:
        """Summarize the text by selecting top N sentences."""
        if not text.strip():
            return ""

        # Split into sentences
        sentences = nltk.sent_tokenize(text)
        cleaned = [clean_sentence(s) for s in sentences]

        # Score
        scores = self.score_sentences(cleaned)

        # Pick top N sentence indices
        ranked_indices = np.argsort(scores)[::-1][:num_sentences]

        # Keep original order
        ranked_indices.sort()
        summary = " ".join([sentences[i] for i in ranked_indices])
        return summary
