from retrieval.vector_retriever import VectorRetriever
from retrieval.keyword_retriever import KeywordRetriever

class HybridRetriever:
    def __init__(self, documents):
        self.vector = VectorRetriever()
        self.keyword = KeywordRetriever(documents)

    def search(self, query):
        v_docs = self.vector.search(query, k=5)
        k_docs = self.keyword.search(query, k=5)

        combined = v_docs + k_docs

        # remove duplicates
        seen = set()
        unique_docs = []
        for doc in combined:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        return unique_docs