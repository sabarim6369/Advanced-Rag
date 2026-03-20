from rank_bm25 import BM25Okapi

class KeywordRetriever:
    def __init__(self, documents):
        self.texts = [doc.page_content.split() for doc in documents]
        self.docs = documents
        self.bm25 = BM25Okapi(self.texts)

    def search(self, query, k=5):
        scores = self.bm25.get_scores(query.split())
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.docs[i] for i in top_k]