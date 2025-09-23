import os
from embedder import Embedder

DATA_FOLDER = "data"

def update_knowledge_base():
    embedder = Embedder()
    embedder.load_vectors()

    # Read all txt files in data/
    new_docs = []
    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".txt"):
            with open(os.path.join(DATA_FOLDER, file), "r", encoding="utf-8") as f:
                new_docs.append(f.read().strip())

    if new_docs:
        print(f"Updating knowledge base with {len(new_docs)} new documents...")
        all_docs = embedder.documents + new_docs
        embedder.fit(all_docs)
        print("Knowledge base updated!")
    else:
        print("No new documents to add.")

if __name__ == "__main__":
    update_knowledge_base()
