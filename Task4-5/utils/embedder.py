import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

VECTORS_PATH = "knowledge_base/vectors.pkl"

class Embedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.vectors = []
        self.documents = []

    def fit(self, documents):
        """Fit the vectorizer and save vectors."""
        self.documents = documents
        self.vectors = self.vectorizer.fit_transform(documents)
        self.save_vectors()

    def transform(self, new_docs):
        """Transform new documents; fit vectorizer if empty."""
        if not hasattr(self.vectorizer, 'vocabulary_') or not self.documents:
            # Fit the vectorizer if not yet fitted
            self.fit(new_docs)
            return self.vectors
        else:
            return self.vectorizer.transform(new_docs)

    def save_vectors(self):
        os.makedirs(os.path.dirname(VECTORS_PATH), exist_ok=True)
        with open(VECTORS_PATH, "wb") as f:
            pickle.dump({"vectors": self.vectors, "documents": self.documents}, f)

    def load_vectors(self):
        """Load vectors safely, handle empty or missing file."""
        if os.path.exists(VECTORS_PATH) and os.path.getsize(VECTORS_PATH) > 0:
            with open(VECTORS_PATH, "rb") as f:
                try:
                    data = pickle.load(f)
                    self.vectors = data.get("vectors", [])
                    self.documents = data.get("documents", [])
                except EOFError:
                    self.vectors = []
                    self.documents = []
        else:
            self.vectors = []
            self.documents = []
        return self.vectors, self.documents
