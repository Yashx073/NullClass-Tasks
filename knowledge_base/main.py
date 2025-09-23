import pickle
import os

class Embedder:
    def __init__(self):
        self.vectorizer = None
        self.vectors = None
        self.documents = []

    def fit(self, documents):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer()
        self.vectors = self.vectorizer.fit_transform(documents)
        self.documents = documents

    def transform(self, new_docs):
        return self.vectorizer.transform(new_docs)

    def save_vectors(self, path="knowledge_base/vectors.pkl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump((self.vectors, self.documents), f)
        print(f"✅ Knowledge base saved with {len(self.documents)} documents!")

    def load_vectors(self, path="knowledge_base/vectors.pkl"):
        import pickle
        if not os.path.exists(path):
            return None, []
        with open(path, "rb") as f:
            self.vectors, self.documents = pickle.load(f)
        return self.vectors, self.documents
