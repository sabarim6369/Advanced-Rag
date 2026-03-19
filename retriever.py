"""
Hybrid Retriever Module for Advanced RAG System
Combines semantic search (FAISS) and keyword search (BM25) for better retrieval
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict

# LangChain imports
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# BM25 implementation
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logging.warning("rank_bm25 not available. BM25 retrieval will be disabled.")

# Local imports
from config import config
from utils import TextProcessor, ConfidenceCalculator

# Configure logging
logger = logging.getLogger(__name__)

class SemanticRetriever:
    """Handles semantic search using FAISS"""
    
    def __init__(self, vector_store: FAISS, embedding_generator):
        """Initialize semantic retriever"""
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
    
    def retrieve(self, query: str, top_k: int = config.TOP_K_RETRIEVAL) -> List[Tuple[Document, float]]:
        """Retrieve documents using semantic search"""
        if not self.vector_store:
            logger.error("No vector store available for semantic retrieval")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_generator.generate_query_embedding(query)
            if not query_embedding:
                logger.error("Failed to generate query embedding")
                return []
            
            # Search vector store
            results = self.vector_store.similarity_search_with_score(
                query, 
                k=top_k
            )
            
            # Normalize scores (FAISS returns distance, lower is better)
            normalized_results = []
            for doc, score in results:
                # Convert distance to similarity score (0-1)
                similarity_score = 1 / (1 + score)
                normalized_results.append((doc, similarity_score))
            
            logger.info(f"Semantic retrieval returned {len(normalized_results)} results")
            return normalized_results
            
        except Exception as e:
            logger.error(f"Error in semantic retrieval: {str(e)}")
            return []

class KeywordRetriever:
    """Handles keyword-based search using BM25"""
    
    def __init__(self):
        """Initialize keyword retriever"""
        self.bm25_index = None
        self.documents = []
        self.tokenized_docs = []
    
    def build_index(self, documents: List[Document]) -> bool:
        """Build BM25 index from documents"""
        if not BM25_AVAILABLE:
            logger.warning("BM25 not available, skipping keyword index build")
            return False
        
        if not documents:
            logger.error("No documents provided for BM25 indexing")
            return False
        
        try:
            self.documents = documents
            
            # Tokenize documents
            self.tokenized_docs = []
            for doc in documents:
                # Extract keywords from document content
                keywords = TextProcessor.extract_keywords(doc.page_content, max_keywords=50)
                self.tokenized_docs.append(keywords)
            
            # Build BM25 index
            self.bm25_index = BM25Okapi(self.tokenized_docs)
            logger.info(f"Successfully built BM25 index with {len(documents)} documents")
            return True
            
        except Exception as e:
            logger.error(f"Error building BM25 index: {str(e)}")
            return False
    
    def retrieve(self, query: str, top_k: int = config.TOP_K_RETRIEVAL) -> List[Tuple[Document, float]]:
        """Retrieve documents using keyword search"""
        if not self.bm25_index or not self.documents:
            logger.error("BM25 index not built or no documents available")
            return []
        
        try:
            # Extract keywords from query
            query_keywords = TextProcessor.extract_keywords(query, max_keywords=10)
            if not query_keywords:
                logger.warning("No keywords extracted from query")
                return []
            
            # Get BM25 scores
            bm25_scores = self.bm25_index.get_scores(query_keywords)
            
            # Get top-k results
            top_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
            
            results = []
            for idx in top_indices:
                if bm25_scores[idx] > 0:  # Only include documents with positive scores
                    doc = self.documents[idx]
                    # Normalize BM25 score to 0-1 range (simple normalization)
                    normalized_score = min(bm25_scores[idx] / 10.0, 1.0)
                    results.append((doc, normalized_score))
            
            logger.info(f"Keyword retrieval returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in keyword retrieval: {str(e)}")
            return []

class HybridRetriever:
    """Combines semantic and keyword retrieval for better results"""
    
    def __init__(self, vector_store: FAISS, embedding_generator):
        """Initialize hybrid retriever"""
        self.semantic_retriever = SemanticRetriever(vector_store, embedding_generator)
        self.keyword_retriever = KeywordRetriever()
        self.documents_indexed = False
        
        # Build keyword index if documents are available
        self._build_keyword_index()
    
    def _build_keyword_index(self) -> None:
        """Build keyword index from vector store documents"""
        try:
            # Get all documents from vector store
            if hasattr(self.semantic_retriever.vector_store, 'docstore'):
                all_docs = list(self.semantic_retriever.vector_store.docstore._dict.values())
                if all_docs:
                    success = self.keyword_retriever.build_index(all_docs)
                    self.documents_indexed = success
                    if success:
                        logger.info("Keyword index built successfully")
                    else:
                        logger.warning("Failed to build keyword index")
        except Exception as e:
            logger.error(f"Error building keyword index: {str(e)}")
    
    def retrieve(self, query: str, top_k: int = config.TOP_K_RETRIEVAL, 
                 semantic_weight: float = 0.6, keyword_weight: float = 0.4) -> List[Tuple[Document, float]]:
        """
        Retrieve documents using hybrid approach
        
        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic search results
            keyword_weight: Weight for keyword search results
        """
        logger.info(f"Starting hybrid retrieval for query: {query[:100]}...")
        
        # Get semantic results
        semantic_results = self.semantic_retriever.retrieve(query, top_k)
        
        # Get keyword results
        keyword_results = []
        if self.documents_indexed:
            keyword_results = self.keyword_retriever.retrieve(query, top_k)
        
        # Combine and re-rank results
        combined_results = self._combine_results(
            semantic_results, 
            keyword_results,
            semantic_weight,
            keyword_weight
        )
        
        # Return top-k results
        final_results = combined_results[:top_k]
        logger.info(f"Hybrid retrieval returned {len(final_results)} final results")
        
        return final_results
    
    def _combine_results(self, semantic_results: List[Tuple[Document, float]], 
                        keyword_results: List[Tuple[Document, float]],
                        semantic_weight: float, keyword_weight: float) -> List[Tuple[Document, float]]:
        """Combine semantic and keyword results with weighted scoring"""
        
        # Create a dictionary to store combined scores by document content
        combined_scores = defaultdict(float)
        doc_map = {}
        
        # Process semantic results
        for doc, score in semantic_results:
            doc_key = self._get_document_key(doc)
            combined_scores[doc_key] += score * semantic_weight
            doc_map[doc_key] = doc
        
        # Process keyword results
        for doc, score in keyword_results:
            doc_key = self._get_document_key(doc)
            combined_scores[doc_key] += score * keyword_weight
            if doc_key not in doc_map:
                doc_map[doc_key] = doc
        
        # Sort by combined scores
        sorted_results = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Convert back to list of (Document, score) tuples
        final_results = []
        for doc_key, combined_score in sorted_results:
            doc = doc_map[doc_key]
            # Add combined score to metadata
            doc.metadata['hybrid_score'] = combined_score
            final_results.append((doc, combined_score))
        
        return final_results
    
    def _get_document_key(self, doc: Document) -> str:
        """Generate a unique key for document deduplication"""
        # Use content hash and metadata for uniqueness
        content = doc.page_content[:200]  # First 200 chars for uniqueness
        source = doc.metadata.get('source', '')
        chunk_id = doc.metadata.get('chunk_id', '')
        return f"{hash(content)}_{source}_{chunk_id}"
    
    def get_retrieval_stats(self, query: str) -> Dict[str, Any]:
        """Get detailed statistics about retrieval performance"""
        # Get individual retrieval results
        semantic_results = self.semantic_retriever.retrieve(query, config.TOP_K_RETRIEVAL)
        keyword_results = []
        if self.documents_indexed:
            keyword_results = self.keyword_retriever.retrieve(query, config.TOP_K_RETRIEVAL)
        
        # Get hybrid results
        hybrid_results = self.retrieve(query, config.TOP_K_RETRIEVAL)
        
        # Calculate confidence scores
        hybrid_scores = [score for _, score in hybrid_results]
        confidence_data = ConfidenceCalculator.calculate_similarity_confidence(hybrid_scores)
        
        return {
            "query": query,
            "semantic_results_count": len(semantic_results),
            "keyword_results_count": len(keyword_results),
            "hybrid_results_count": len(hybrid_results),
            "keyword_index_available": self.documents_indexed,
            "confidence_data": confidence_data,
            "top_semantic_scores": [score for _, score in semantic_results[:3]],
            "top_keyword_scores": [score for _, score in keyword_results[:3]],
            "top_hybrid_scores": hybrid_scores[:3]
        }
    
    def update_vector_store(self, vector_store: FAISS) -> None:
        """Update the vector store and rebuild keyword index"""
        self.semantic_retriever.vector_store = vector_store
        self._build_keyword_index()
        logger.info("Vector store updated and keyword index rebuilt")

class AdaptiveRetriever:
    """Adaptive retriever that adjusts strategy based on query type and performance"""
    
    def __init__(self, hybrid_retriever: HybridRetriever):
        """Initialize adaptive retriever"""
        self.hybrid_retriever = hybrid_retriever
        self.query_history = []
    
    def retrieve(self, query: str, top_k: int = config.TOP_K_RETRIEVAL) -> List[Tuple[Document, float]]:
        """Retrieve with adaptive strategy"""
        # Analyze query type
        query_type = self._analyze_query_type(query)
        
        # Adjust weights based on query type
        if query_type == "factual":
            # Favor semantic search for factual queries
            semantic_weight, keyword_weight = 0.7, 0.3
        elif query_type == "keyword_heavy":
            # Favor keyword search for queries with specific terms
            semantic_weight, keyword_weight = 0.4, 0.6
        else:
            # Use balanced approach for general queries
            semantic_weight, keyword_weight = 0.6, 0.4
        
        logger.info(f"Query type: {query_type}, weights - semantic: {semantic_weight}, keyword: {keyword_weight}")
        
        # Perform retrieval with adjusted weights
        results = self.hybrid_retriever.retrieve(
            query, 
            top_k, 
            semantic_weight, 
            keyword_weight
        )
        
        # Store query performance
        self._record_query_performance(query, query_type, results)
        
        return results
    
    def _analyze_query_type(self, query: str) -> str:
        """Analyze query to determine optimal retrieval strategy"""
        query_lower = query.lower()
        
        # Check for factual question patterns
        factual_patterns = ["what is", "what are", "how does", "why is", "when was", "who is"]
        if any(pattern in query_lower for pattern in factual_patterns):
            return "factual"
        
        # Check for keyword-heavy queries (specific terms, names, etc.)
        words = query_lower.split()
        if len(words) <= 4 or any(word.isdigit() for word in words):
            return "keyword_heavy"
        
        return "general"
    
    def _record_query_performance(self, query: str, query_type: str, results: List[Tuple[Document, float]]) -> None:
        """Record query performance for learning"""
        if len(results) > 0:
            avg_score = sum(score for _, score in results) / len(results)
            self.query_history.append({
                "query": query,
                "type": query_type,
                "results_count": len(results),
                "avg_score": avg_score
            })
            
            # Keep only recent history (last 100 queries)
            if len(self.query_history) > 100:
                self.query_history = self.query_history[-100:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.query_history:
            return {"message": "No query history available"}
        
        # Calculate stats by query type
        type_stats = defaultdict(list)
        for record in self.query_history:
            type_stats[record["type"]].append(record["avg_score"])
        
        performance_summary = {}
        for query_type, scores in type_stats.items():
            performance_summary[query_type] = {
                "count": len(scores),
                "avg_score": sum(scores) / len(scores),
                "max_score": max(scores),
                "min_score": min(scores)
            }
        
        return {
            "total_queries": len(self.query_history),
            "performance_by_type": performance_summary,
            "recent_queries": self.query_history[-5:]
        }
