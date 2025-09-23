import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

class TfidfRetriever:
    def __init__(self, model_path='./models/tfidf_vectorizer.pkl', 
                 vectors_path='./models/answer_vectors.npy', 
                 data_path='./data/medqa_knowledge_base.csv'):
        
        self.model_path = model_path
        self.vectors_path = vectors_path
        self.data_path = data_path
        
        self.vectorizer = None
        self.answer_vectors = None
        self.knowledge_base = None
        
        self._load_resources()

    def _load_resources(self):
        """Loads the saved vectorizer, vectors, and knowledge base."""
        print("Loading Retrieval Resources...")
        if not all(os.path.exists(p) for p in [self.model_path, self.vectors_path, self.data_path]):
            raise FileNotFoundError("One or more required model/data files not found. Run the notebook first.")

        with open(self.model_path, 'rb') as f:
            self.vectorizer = pickle.load(f)

        self.answer_vectors = np.load(self.vectors_path)
        
        self.knowledge_base = pd.read_csv(self.data_path)
        print("Resources Loaded.")

    def retrieve_answer(self, query: str, top_k: int = 1):
        """Finds the most relevant answer(s) using cosine similarity."""
        if not self.vectorizer or self.answer_vectors is None:
            raise RuntimeError("Retriever resources not loaded.")
            
        # 1. Vectorize the input query
        query_vector = self.vectorizer.transform([query])
        
        # 2. Calculate cosine similarity between query and all answers
        similarities = cosine_similarity(query_vector, self.answer_vectors)[0]
        
        # 3. Get the indices of the top-k most similar answers
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 4. Extract the results
        results = []
        for i in top_indices:
            result = {
                'score': similarities[i],
                'question': self.knowledge_base.iloc[i]['question'],
                'answer': self.knowledge_base.iloc[i]['answer'],
                'source': self.knowledge_base.iloc[i]['source']
            }
            results.append(result)
            
        return results

if __name__ == '__main__':
    # This part will fail unless you've run the notebook and saved the files
    try:
        retriever = TfidfRetriever()
        test_query = "What is type 2 diabetes?"
        answer = retriever.retrieve_answer(test_query)
        print("\nTest Retrieval:")
        print(f"Query: {test_query}")
        print(f"Best Answer Score: {answer[0]['score']:.4f}")
        print(f"Answer: {answer[0]['answer'][:150]}...")
    except FileNotFoundError as e:
        print(f"\nError: {e}")