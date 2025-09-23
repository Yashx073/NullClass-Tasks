from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def retrieve_papers(query, embedder, top_k=3):
    if not embedder.documents or embedder.vectors is None:
        return []

    # Auto-fit vectorizer if needed
    if not hasattr(embedder.vectorizer, "vocabulary_"):
        embedder.vectorizer.fit(embedder.documents)

    query_vec = embedder.transform([query])

    # Ensure X and Y dimensions match
    if query_vec.shape[1] != embedder.vectors.shape[1]:
        embedder.vectorizer.fit(embedder.documents + [query])
        query_vec = embedder.transform([query])

    sims = cosine_similarity(query_vec, embedder.vectors).flatten()
    top_indices = sims.argsort()[-top_k:][::-1]
    return [embedder.documents[i] for i in top_indices]
