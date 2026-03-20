import faiss
import pickle
from pathlib import Path

from ingestion.embedder import embed_text

STORE_PATH = Path("faiss_store.pkl")

class VectorStore:
    def __init__(self):
        self.index = None
        self.docs = []

    def build(self, documents):
        texts = [doc.page_content for doc in documents]
        embeddings = embed_text(texts)

        self.index = faiss.IndexFlatL2(len(embeddings[0]))
        self.index.add(embeddings)

        self.docs = documents

        with STORE_PATH.open("wb") as f:
            pickle.dump((self.index, self.docs), f)

    def load(self):
        with STORE_PATH.open("rb") as f:
            self.index, self.docs = pickle.load(f)

    def exists(self):
        return STORE_PATH.exists()

    def search(self, query, k=5):
        query_vec = embed_text([query])
        distances, indices = self.index.search(query_vec, k)
        if distances[0][0] > 1.5:
            return []  # not relevant
        return [self.docs[i] for i in indices[0]]
