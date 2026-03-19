"""
Main RAG Pipeline for Advanced RAG System
Coordinates all components for retrieval-augmented generation
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime

# LangChain imports
from langchain_core.documents import Document

# Local imports
from config import config
from ingestion import DocumentIngestionPipeline, EmbeddingGenerator
from retriever import HybridRetriever, AdaptiveRetriever
from reranker import RerankingPipeline
from guardrails import SecurityGuard
from utils import QueryRewriter, ConfidenceCalculator, setup_logging

# Configure logging
logger = logging.getLogger(__name__)

class GroqLLM:
    """Groq LLM wrapper for text generation"""
    
    def __init__(self, api_key: str = None, model: str = None):
        """Initialize Groq LLM"""
        self.api_key = api_key or config.GROQ_API_KEY
        self.model = model or config.GROQ_MODEL
        self.base_url = config.GROQ_BASE_URL
        
        if not self.api_key:
            raise ValueError("Groq API key is required")
        
        # Try to import Groq
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            logger.info("Groq client initialized successfully")
        except ImportError:
            logger.error("Groq library not installed. Install with: pip install groq")
            raise
    
    def generate_response(self, prompt: str, max_tokens: int = 1000, 
                        temperature: float = 0.1) -> str:
        """
        Generate response using Groq API
        
        Args:
            prompt: The prompt to send to the LLM
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated response text
        """
        try:
            logger.info(f"Generating response with {self.model}")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            generated_text = response.choices[0].message.content
            logger.info(f"Generated {len(generated_text)} characters")
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error generating response with Groq: {str(e)}")
            return "I apologize, but I encountered an error while generating a response."
    
    def rewrite_query(self, original_query: str) -> str:
        """
        Rewrite query for better retrieval
        
        Args:
            original_query: Original user query
            
        Returns:
            Rewritten query
        """
        rewrite_prompt = f"""
        Rewrite the following query to be more effective for document retrieval.
        Make it more specific, clear, and focused on the key information needed.
        
        Original query: {original_query}
        
        Rewritten query (just the query, no explanation):
        """
        
        try:
            rewritten = self.generate_response(rewrite_prompt, max_tokens=200, temperature=0.3)
            return rewritten.strip()
        except Exception as e:
            logger.error(f"Error rewriting query: {str(e)}")
            return original_query

class RAGPipeline:
    """Main RAG pipeline coordinating all components"""
    
    def __init__(self, use_openai_embeddings: bool = False):
        """Initialize RAG pipeline"""
        logger.info("Initializing Advanced RAG Pipeline")
        
        # Initialize components
        self.ingestion_pipeline = DocumentIngestionPipeline(use_openai_embeddings)
        self.embedding_generator = self.ingestion_pipeline.embedding_generator
        self.security_guard = SecurityGuard()
        self.query_rewriter = QueryRewriter()
        self.confidence_calculator = ConfidenceCalculator()
        
        # Initialize retriever (will be set when vector store is available)
        self.hybrid_retriever = None
        self.adaptive_retriever = None
        
        # Initialize reranker
        self.reranking_pipeline = RerankingPipeline(
            use_cross_encoder=True,
            use_diversity=True
        )
        
        # Initialize LLM
        self.llm = GroqLLM()
        
        # Pipeline statistics
        self.query_history = []
        self.pipeline_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "blocked_queries": 0,
            "low_confidence_queries": 0,
            "avg_response_time": 0.0
        }
        
        # Try to load existing vector store
        self._initialize_retriever()
    
    def _initialize_retriever(self) -> None:
        """Initialize retriever with existing vector store"""
        vector_store = self.ingestion_pipeline.vector_store_manager.get_vector_store()
        if vector_store:
            self.hybrid_retriever = HybridRetriever(vector_store, self.embedding_generator)
            self.adaptive_retriever = AdaptiveRetriever(self.hybrid_retriever)
            logger.info("Retriever initialized with existing vector store")
        else:
            logger.warning("No vector store available - need to ingest documents first")
    
    def ingest_document(self, file_path: str) -> Dict[str, Any]:
        """
        Ingest a document into the system
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with ingestion results
        """
        logger.info(f"Ingesting document: {file_path}")
        
        result = self.ingestion_pipeline.ingest_file(file_path)
        
        if result["success"]:
            # Reinitialize retriever with updated vector store
            self._initialize_retriever()
            logger.info("Document ingestion completed successfully")
        else:
            logger.error(f"Document ingestion failed: {result['error']}")
        
        return result
    
    def process_query(self, query: str, use_adaptive: bool = True) -> Dict[str, Any]:
        """
        Process a user query through the complete RAG pipeline
        
        Args:
            query: User query
            use_adaptive: Whether to use adaptive retrieval
            
        Returns:
            Dictionary with query results and metadata
        """
        start_time = datetime.now()
        logger.info(f"Processing query: {query[:100]}...")
        
        # Update pipeline stats
        self.pipeline_stats["total_queries"] += 1
        
        # Step 1: Security check
        security_analysis = self.security_guard.process_query(query)
        if security_analysis["should_block"]:
            self.pipeline_stats["blocked_queries"] += 1
            return {
                "success": False,
                "answer": "I cannot process this request due to security concerns.",
                "security_analysis": security_analysis,
                "confidence": 0.0,
                "sources": [],
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
        
        sanitized_query = security_analysis["final_query"]
        
        # Step 2: Initial retrieval attempt
        retrieval_result = self._perform_retrieval(sanitized_query, use_adaptive)
        
        # Step 3: Check confidence and decide on retry
        confidence_data = retrieval_result.get("confidence_data", {})
        should_retry = confidence_data.get("confidence") == "low"
        
        if should_retry:
            logger.info("Low confidence detected, attempting query rewrite")
            
            # Step 4: Query rewrite and retry
            rewritten_query = self.llm.rewrite_query(sanitized_query)
            logger.info(f"Rewritten query: {rewritten_query}")
            
            retry_result = self._perform_retrieval(rewritten_query, use_adaptive)
            
            # Use better result between original and retry
            if (retry_result.get("confidence_data", {}).get("avg_score", 0) > 
                confidence_data.get("avg_score", 0)):
                retrieval_result = retry_result
                logger.info("Retry with rewritten query was better")
            else:
                logger.info("Original query was better, keeping original result")
        
        # Step 5: Final confidence check
        final_confidence = retrieval_result.get("confidence_data", {}).get("confidence", "low")
        
        if final_confidence == "low":
            self.pipeline_stats["low_confidence_queries"] += 1
            answer = "I don't have enough information to answer this question based on the available documents."
            sources = []
        else:
            # Step 6: Generate answer
            answer, sources = self._generate_answer(
                sanitized_query, 
                retrieval_result["documents"]
            )
            self.pipeline_stats["successful_queries"] += 1
        
        # Step 7: Validate response
        response_validation = self.security_guard.validate_response(answer)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        self._update_avg_response_time(processing_time)
        
        # Log query
        self._log_query(query, retrieval_result, processing_time)
        
        return {
            "success": True,
            "answer": answer,
            "sources": sources,
            "confidence_data": retrieval_result.get("confidence_data", {}),
            "security_analysis": security_analysis,
            "response_validation": response_validation,
            "retrieval_stats": retrieval_result.get("retrieval_stats", {}),
            "processing_time": processing_time,
            "query_rewritten": should_retry and final_confidence != "low"
        }
    
    def _perform_retrieval(self, query: str, use_adaptive: bool = True) -> Dict[str, Any]:
        """Perform document retrieval"""
        if not self.hybrid_retriever:
            logger.error("No retriever available - no documents ingested")
            return {
                "documents": [],
                "confidence_data": {"confidence": "low", "avg_score": 0.0},
                "retrieval_stats": {"error": "No retriever available"}
            }
        
        try:
            # Choose retriever
            retriever = self.adaptive_retriever if use_adaptive else self.hybrid_retriever
            
            # Perform retrieval
            retrieved_docs = retriever.retrieve(query, top_k=config.TOP_K_RETRIEVAL)
            
            if not retrieved_docs:
                logger.warning("No documents retrieved")
                return {
                    "documents": [],
                    "confidence_data": {"confidence": "low", "avg_score": 0.0},
                    "retrieval_stats": {"results_count": 0}
                }
            
            # Extract documents and scores
            documents = [doc for doc, _ in retrieved_docs]
            initial_scores = [score for _, score in retrieved_docs]
            
            # Rerank documents
            reranking_result = self.reranking_pipeline.process(
                query, documents, initial_scores, config.TOP_K_RERANK
            )
            
            # Get retrieval statistics
            if use_adaptive and self.adaptive_retriever:
                retrieval_stats = self.adaptive_retriever.hybrid_retriever.get_retrieval_stats(query)
                retrieval_stats.update(self.adaptive_retriever.get_performance_stats())
            else:
                retrieval_stats = self.hybrid_retriever.get_retrieval_stats(query)
            
            return {
                "documents": reranking_result["documents"],
                "confidence_data": reranking_result["confidence_data"],
                "retrieval_stats": retrieval_stats,
                "reranker_config": reranking_result["reranker_config"]
            }
            
        except Exception as e:
            logger.error(f"Error during retrieval: {str(e)}")
            return {
                "documents": [],
                "confidence_data": {"confidence": "low", "avg_score": 0.0},
                "retrieval_stats": {"error": str(e)}
            }
    
    def _generate_answer(self, query: str, documents: List[Document]) -> Tuple[str, List[Dict[str, Any]]]:
        """Generate answer based on retrieved documents"""
        if not documents:
            return "I don't have enough information to answer this question.", []
        
        # Prepare context
        context_parts = []
        sources = []
        
        for i, doc in enumerate(documents):
            context_parts.append(f"Document {i+1}: {doc.page_content}")
            
            # Prepare source information
            source_info = {
                "doc_id": i + 1,
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "score": doc.metadata.get("final_score", doc.metadata.get("hybrid_score", 0.0)),
                "source": doc.metadata.get("source", "Unknown"),
                "chunk_id": doc.metadata.get("chunk_id", f"chunk_{i}")
            }
            sources.append(source_info)
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""
        Answer ONLY using the provided context. If the answer is not found in the context, say "I don't know".
        Do not use any external knowledge or make up information.
        
        Context:
        {context}
        
        Question:
        {query}
        
        Answer:
        """
        
        try:
            answer = self.llm.generate_response(prompt, max_tokens=800, temperature=0.1)
            return answer, sources
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I apologize, but I encountered an error while generating an answer.", sources
    
    def _log_query(self, query: str, retrieval_result: Dict[str, Any], processing_time: float) -> None:
        """Log query for monitoring"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query[:100],  # Truncate for privacy
            "processing_time": processing_time,
            "retrieved_docs": len(retrieval_result.get("documents", [])),
            "confidence": retrieval_result.get("confidence_data", {}).get("confidence", "unknown"),
            "avg_score": retrieval_result.get("confidence_data", {}).get("avg_score", 0.0)
        }
        
        self.query_history.append(log_entry)
        
        # Keep only recent history
        if len(self.query_history) > 1000:
            self.query_history = self.query_history[-1000:]
    
    def _update_avg_response_time(self, processing_time: float) -> None:
        """Update average response time"""
        total_queries = self.pipeline_stats["total_queries"]
        current_avg = self.pipeline_stats["avg_response_time"]
        
        # Calculate new average
        new_avg = ((current_avg * (total_queries - 1)) + processing_time) / total_queries
        self.pipeline_stats["avg_response_time"] = new_avg
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        # Get ingestion stats
        ingestion_stats = self.ingestion_pipeline.get_pipeline_stats()
        
        # Get security stats
        security_stats = self.security_guard.get_security_stats()
        
        # Get retriever stats
        retriever_stats = {}
        if self.adaptive_retriever:
            retriever_stats = self.adaptive_retriever.get_performance_stats()
        elif self.hybrid_retriever:
            # Basic stats without adaptive retriever
            retriever_stats = {
                "vector_store_stats": ingestion_stats.get("vector_store_stats", {}),
                "message": "Adaptive retriever not available"
            }
        
        return {
            "pipeline_stats": self.pipeline_stats,
            "ingestion_stats": ingestion_stats,
            "security_stats": security_stats,
            "retriever_stats": retriever_stats,
            "reranker_stats": self.reranking_pipeline.get_pipeline_stats(),
            "config": {
                "chunk_size": config.CHUNK_SIZE,
                "chunk_overlap": config.CHUNK_OVERLAP,
                "top_k_retrieval": config.TOP_K_RETRIEVAL,
                "top_k_rerank": config.TOP_K_RERANK,
                "confidence_threshold": config.CONFIDENCE_THRESHOLD,
                "groq_model": config.GROQ_MODEL
            }
        }
    
    def clear_all_data(self) -> Dict[str, Any]:
        """Clear all data from the system"""
        logger.info("Clearing all system data")
        
        try:
            # Clear vector store
            success = self.ingestion_pipeline.clear_vector_store()
            
            # Reset retriever
            self.hybrid_retriever = None
            self.adaptive_retriever = None
            
            # Clear query history
            self.query_history = []
            
            # Reset stats
            self.pipeline_stats = {
                "total_queries": 0,
                "successful_queries": 0,
                "blocked_queries": 0,
                "low_confidence_queries": 0,
                "avg_response_time": 0.0
            }
            
            # Clear security log
            self.security_guard.reset_security_log()
            
            return {
                "success": success,
                "message": "All data cleared successfully" if success else "Error clearing data"
            }
            
        except Exception as e:
            logger.error(f"Error clearing data: {str(e)}")
            return {
                "success": False,
                "message": f"Error clearing data: {str(e)}"
            }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check"""
        health_status = {
            "overall_status": "healthy",
            "components": {},
            "issues": []
        }
        
        # Check Groq API
        try:
            test_response = self.llm.generate_response("Hello", max_tokens=10)
            health_status["components"]["groq_api"] = "healthy"
        except Exception as e:
            health_status["components"]["groq_api"] = "unhealthy"
            health_status["issues"].append(f"Groq API error: {str(e)}")
            health_status["overall_status"] = "degraded"
        
        # Check vector store
        vector_store = self.ingestion_pipeline.vector_store_manager.get_vector_store()
        if vector_store:
            health_status["components"]["vector_store"] = "healthy"
            health_status["components"]["document_count"] = vector_store.index.ntotal if hasattr(vector_store.index, 'ntotal') else "unknown"
        else:
            health_status["components"]["vector_store"] = "not_initialized"
            health_status["issues"].append("No vector store initialized")
            health_status["overall_status"] = "degraded"
        
        # Check retriever
        if self.hybrid_retriever:
            health_status["components"]["retriever"] = "healthy"
        else:
            health_status["components"]["retriever"] = "not_initialized"
            health_status["issues"].append("No retriever initialized")
        
        # Check reranker
        reranker_config = self.reranking_pipeline.get_pipeline_stats()
        if reranker_config.get("reranker_config", {}).get("cross_encoder_available"):
            health_status["components"]["reranker"] = "healthy"
        else:
            health_status["components"]["reranker"] = "limited"
            health_status["issues"].append("Cross-encoder reranker not available")
        
        return health_status
