import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

class Embedder:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.documents = []
        self.vectors = None

    def fit(self, documents):
        """Fit TF-IDF vectorizer on documents and save vectors."""
        self.documents = documents
        self.vectors = self.vectorizer.fit_transform(documents)
        os.makedirs("knowledge_base", exist_ok=True)
        with open("knowledge_base/vectors.pkl", "wb") as f:
            pickle.dump({"vectors": self.vectors, "documents": self.documents}, f)
        print(f"✅ Knowledge base saved with {len(documents)} documents.")

    def transform(self, new_docs):
        """Transform new documents into TF-IDF vectors (auto-fit if needed)."""
        if not hasattr(self.vectorizer, "vocabulary_"):
            if self.documents:
                self.vectorizer.fit(self.documents)
            else:
                raise ValueError("No documents available to fit the vectorizer!")
        return self.vectorizer.transform(new_docs)

    def load_vectors(self, path="knowledge_base/vectors.pkl"):
        """Load vectors from pickle safely."""
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            print("⚠️ vectors.pkl is empty. Returning empty vectors/documents.")
            self.vectors = None
            self.documents = []
            return self.vectors, self.documents

        with open(path, "rb") as f:
            data = pickle.load(f)
        self.vectors = data.get("vectors", None)
        self.documents = data.get("documents", [])
        print(f"✅ Loaded {len(self.documents)} documents from vectors.pkl")
        return self.vectors, self.documents
