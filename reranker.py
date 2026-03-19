"""
Reranker Module for Advanced RAG System
Uses cross-encoder models to re-rank retrieved documents for better relevance
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# LangChain imports
from langchain_core.documents import Document

# Sentence Transformers for cross-encoder
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logging.warning("sentence-transformers not available. Cross-encoder reranking will be disabled.")

# Local imports
from config import config
from utils import ConfidenceCalculator

# Configure logging
logger = logging.getLogger(__name__)

class CrossEncoderReranker:
    """Reranker using cross-encoder models for better relevance scoring"""
    
    def __init__(self, model_name: str = config.RERANKER_MODEL, batch_size: int = config.RERANKER_BATCH_SIZE):
        """Initialize cross-encoder reranker"""
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the cross-encoder model"""
        if not CROSS_ENCODER_AVAILABLE:
            logger.error("sentence-transformers not available for cross-encoder reranking")
            return
        
        try:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading cross-encoder model: {str(e)}")
            self.model = None
    
    def rerank(self, query: str, documents: List[Document], 
               top_k: int = config.TOP_K_RERANK) -> List[Tuple[Document, float]]:
        """
        Rerank documents based on their relevance to the query
        
        Args:
            query: The search query
            documents: List of documents to rerank
            top_k: Number of top documents to return
            
        Returns:
            List of (Document, relevance_score) tuples
        """
        if not self.model:
            logger.error("Cross-encoder model not available for reranking")
            return [(doc, 0.0) for doc in documents[:top_k]]
        
        if not documents:
            logger.warning("No documents provided for reranking")
            return []
        
        try:
            # Prepare query-document pairs
            query_doc_pairs = []
            for doc in documents:
                # Use a portion of the document content for scoring
                content = doc.page_content[:500]  # Limit content length for efficiency
                query_doc_pairs.append([query, content])
            
            # Get relevance scores in batches
            all_scores = []
            for i in range(0, len(query_doc_pairs), self.batch_size):
                batch_pairs = query_doc_pairs[i:i + self.batch_size]
                batch_scores = self.model.predict(batch_pairs)
                all_scores.extend(batch_scores)
            
            # Combine documents with scores
            doc_scores = list(zip(documents, all_scores))
            
            # Sort by relevance score (descending)
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Return top-k results
            top_results = doc_scores[:top_k]
            
            # Add rerank scores to metadata
            for doc, score in top_results:
                doc.metadata['rerank_score'] = float(score)
                doc.metadata['rerank_model'] = self.model_name
            
            logger.info(f"Reranked {len(documents)} documents, returning top {len(top_results)}")
            return top_results
            
        except Exception as e:
            logger.error(f"Error during reranking: {str(e)}")
            # Fallback to original order
            return [(doc, 0.0) for doc in documents[:top_k]]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the reranker model"""
        return {
            "model_name": self.model_name,
            "model_loaded": self.model is not None,
            "batch_size": self.batch_size,
            "cross_encoder_available": CROSS_ENCODER_AVAILABLE
        }

class DiversityReranker:
    """Reranker that promotes diversity in selected documents"""
    
    def __init__(self, diversity_threshold: float = 0.8):
        """Initialize diversity reranker"""
        self.diversity_threshold = diversity_threshold
    
    def rerank(self, query: str, documents: List[Document], 
               base_scores: List[float], top_k: int = config.TOP_K_RERANK) -> List[Tuple[Document, float]]:
        """
        Rerank documents considering both relevance and diversity
        
        Args:
            query: The search query
            documents: List of documents to rerank
            base_scores: Base relevance scores for documents
            top_k: Number of top documents to return
            
        Returns:
            List of (Document, adjusted_score) tuples
        """
        if not documents or len(documents) != len(base_scores):
            logger.warning("Invalid input for diversity reranking")
            return []
        
        try:
            # Sort by base scores initially
            sorted_docs = sorted(zip(documents, base_scores), key=lambda x: x[1], reverse=True)
            
            selected_docs = []
            selected_scores = []
            
            for doc, base_score in sorted_docs:
                if len(selected_docs) >= top_k:
                    break
                
                # Check diversity against already selected documents
                is_diverse = True
                diversity_penalty = 0.0
                
                for selected_doc, _ in selected_docs:
                    similarity = self._calculate_content_similarity(doc.page_content, selected_doc.page_content)
                    if similarity > self.diversity_threshold:
                        is_diverse = False
                        diversity_penalty += 0.1  # Penalize similar content
                        break
                
                # Adjust score based on diversity
                adjusted_score = base_score - diversity_penalty
                
                if is_diverse or len(selected_docs) < top_k // 2:  # Always include some top results
                    selected_docs.append((doc, adjusted_score))
                    selected_scores.append(adjusted_score)
                    doc.metadata['diversity_score'] = adjusted_score
                    doc.metadata['diversity_penalty'] = diversity_penalty
            
            # Sort final results by adjusted scores
            selected_docs.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"Diversity reranking: selected {len(selected_docs)} diverse documents")
            return selected_docs
            
        except Exception as e:
            logger.error(f"Error during diversity reranking: {str(e)}")
            return list(zip(documents, base_scores))[:top_k]
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate content similarity using simple word overlap"""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0

class HybridReranker:
    """Combines multiple reranking strategies for optimal results"""
    
    def __init__(self, use_cross_encoder: bool = True, use_diversity: bool = True):
        """Initialize hybrid reranker"""
        self.use_cross_encoder = use_cross_encoder
        self.use_diversity = use_diversity
        
        # Initialize individual rerankers
        self.cross_encoder = None
        if use_cross_encoder and CROSS_ENCODER_AVAILABLE:
            self.cross_encoder = CrossEncoderReranker()
        
        self.diversity_reranker = None
        if use_diversity:
            self.diversity_reranker = DiversityReranker()
    
    def rerank(self, query: str, documents: List[Document], 
               initial_scores: List[float], top_k: int = config.TOP_K_RERANK) -> List[Tuple[Document, float]]:
        """
        Apply hybrid reranking combining multiple strategies
        
        Args:
            query: The search query
            documents: List of documents to rerank
            initial_scores: Initial relevance scores
            top_k: Number of top documents to return
            
        Returns:
            List of (Document, final_score) tuples
        """
        if not documents:
            logger.warning("No documents provided for hybrid reranking")
            return []
        
        logger.info(f"Starting hybrid reranking for {len(documents)} documents")
        
        try:
            # Step 1: Cross-encoder reranking (if available)
            if self.use_cross_encoder and self.cross_encoder and self.cross_encoder.model:
                cross_encoder_results = self.cross_encoder.rerank(query, documents, len(documents))
                reranked_docs = [doc for doc, _ in cross_encoder_results]
                cross_encoder_scores = [score for _, score in cross_encoder_results]
                logger.info("Applied cross-encoder reranking")
            else:
                reranked_docs = documents
                cross_encoder_scores = initial_scores
                logger.info("Skipped cross-encoder reranking (not available)")
            
            # Step 2: Diversity reranking (if enabled)
            if self.use_diversity and self.diversity_reranker:
                final_results = self.diversity_reranker.rerank(
                    query, reranked_docs, cross_encoder_scores, top_k
                )
                logger.info("Applied diversity reranking")
            else:
                # Just use cross-encoder results
                final_results = list(zip(reranked_docs, cross_encoder_scores))[:top_k]
                logger.info("Skipped diversity reranking")
            
            # Add final ranking metadata
            for i, (doc, score) in enumerate(final_results):
                doc.metadata['final_rank'] = i + 1
                doc.metadata['final_score'] = float(score)
            
            logger.info(f"Hybrid reranking completed: returning top {len(final_results)} documents")
            return final_results
            
        except Exception as e:
            logger.error(f"Error during hybrid reranking: {str(e)}")
            # Fallback to original ordering
            return list(zip(documents, initial_scores))[:top_k]
    
    def get_reranker_config(self) -> Dict[str, Any]:
        """Get reranker configuration"""
        config_info = {
            "use_cross_encoder": self.use_cross_encoder,
            "use_diversity": self.use_diversity,
            "cross_encoder_available": CROSS_ENCODER_AVAILABLE
        }
        
        if self.cross_encoder:
            config_info["cross_encoder_info"] = self.cross_encoder.get_model_info()
        
        return config_info

class RerankingPipeline:
    """Complete reranking pipeline with confidence scoring"""
    
    def __init__(self, use_cross_encoder: bool = True, use_diversity: bool = True):
        """Initialize reranking pipeline"""
        self.hybrid_reranker = HybridReranker(use_cross_encoder, use_diversity)
        self.confidence_calculator = ConfidenceCalculator()
    
    def process(self, query: str, documents: List[Document], 
                initial_scores: List[float], top_k: int = config.TOP_K_RERANK) -> Dict[str, Any]:
        """
        Process documents through complete reranking pipeline
        
        Args:
            query: The search query
            documents: List of retrieved documents
            initial_scores: Initial retrieval scores
            top_k: Number of final documents to return
            
        Returns:
            Dictionary containing reranking results and metadata
        """
        logger.info(f"Processing {len(documents)} documents through reranking pipeline")
        
        # Rerank documents
        reranked_results = self.hybrid_reranker.rerank(query, documents, initial_scores, top_k)
        
        # Extract final scores
        final_scores = [score for _, score in reranked_results]
        
        # Calculate confidence metrics
        confidence_data = self.confidence_calculator.calculate_similarity_confidence(
            final_scores, 
            threshold=config.CONFIDENCE_THRESHOLD
        )
        
        # Prepare results
        result_docs = [doc for doc, _ in reranked_results]
        
        # Create comprehensive result
        pipeline_result = {
            "documents": result_docs,
            "scores": final_scores,
            "count": len(result_docs),
            "confidence_data": confidence_data,
            "reranker_config": self.hybrid_reranker.get_reranker_config(),
            "query": query,
            "original_count": len(documents)
        }
        
        # Add decision based on confidence
        pipeline_result["should_use_context"] = confidence_data["confidence"] in ["high", "medium"]
        pipeline_result["recommendation"] = self._get_recommendation(confidence_data)
        
        logger.info(f"Reranking pipeline completed: {len(result_docs)} documents, confidence: {confidence_data['confidence']}")
        return pipeline_result
    
    def _get_recommendation(self, confidence_data: Dict[str, Any]) -> str:
        """Get recommendation based on confidence data"""
        confidence = confidence_data["confidence"]
        avg_score = confidence_data["avg_score"]
        
        if confidence == "high":
            return "High confidence results - use retrieved context for answer generation"
        elif confidence == "medium":
            return "Medium confidence - consider query rewriting or additional context"
        else:
            return "Low confidence - may need to retry retrieval or fallback to general knowledge"
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics and configuration"""
        return {
            "reranker_config": self.hybrid_reranker.get_reranker_config(),
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "top_k_rerank": config.TOP_K_RERANK
        }
