from ingestion.vector_store import VectorStore

class VectorRetriever:
    def __init__(self):
        self.store = VectorStore()
        self.store.load()

    def search(self, query, k=5):
        return self.store.search(query, k)