import faiss
import pickle
from ingestion.embedder import embed_text

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

        with open("faiss_store.pkl", "wb") as f:
            pickle.dump((self.index, self.docs), f)

    def load(self):
        with open("faiss_store.pkl", "rb") as f:
            self.index, self.docs = pickle.load(f)

    def search(self, query, k=5):
        query_vec = embed_text([query])
        distances, indices = self.index.search(query_vec, k)
        return [self.docs[i] for i in indices[0]]