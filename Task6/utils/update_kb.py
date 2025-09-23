import pandas as pd
from utils.embedder import Embedder

def update_knowledge_base(csv_file="data/arxiv_cs_subset.csv"):
    reader = pd.read_csv(csv_file, chunksize=500)  # process 500 rows at a time
    embedder = Embedder()
    all_docs = []

    for chunk in reader:
        docs = chunk['abstract'].tolist()
        all_docs.extend(docs)

    embedder.fit(all_docs)
    print(f"Knowledge base updated with {len(all_docs)} papers!")
