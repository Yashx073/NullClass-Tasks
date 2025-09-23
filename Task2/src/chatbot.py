from .retriever import TfidfRetriever
from .ner import get_medical_entities

class MedicalChatbot:
    def __init__(self):
        # Initialize the retrieval mechanism
        self.retriever = TfidfRetriever()
        print("Chatbot Initialized.")

    def get_response(self, query: str):
        """Generates a comprehensive response to a medical query."""
        
        # 1. Entity Recognition
        entities = get_medical_entities(query)
        entity_report = f"Detected concepts: {', '.join(entities.get('Concepts', []))}"
        
        # 2. Retrieval
        retrieval_results = self.retriever.retrieve_answer(query, top_k=1)
        
        if not retrieval_results or retrieval_results[0]['score'] < 0.2:
            # Low similarity threshold (adjust as needed)
            return "I'm sorry, I couldn't find a relevant answer in my knowledge base. Please try rephrasing your question.", None
        
        best_match = retrieval_results[0]
        
        # 3. Construct the Response
        response = f"""
        **Answer:** {best_match['answer']}
        
        **Confidence Score:** {best_match['score']:.2f}
        
        **Source:** {best_match['source']}
        """
        
        return response, entity_report

if __name__ == '__main__':
    # This will fail unless you've run the notebook and saved the files
    try:
        bot = MedicalChatbot()
        test_query = "Tell me about the causes of stomach ulcers."
        response, entities = bot.get_response(test_query)
        print("\n--- CHATBOT RESPONSE ---")
        print(response)
        print(f"\nNER Report: {entities}")
    except FileNotFoundError as e:
        print(f"\nError: {e}")