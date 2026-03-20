from ingestion.vector_store import VectorStore

class VectorRetriever:
    def __init__(self, store=None):
        self.store = store or VectorStore()
        if self.store.index is None:
            self.store.load()

    def search(self, query, k=5):
        return self.store.search(query, k)
