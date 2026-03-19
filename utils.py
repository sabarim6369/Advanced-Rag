"""
Utility functions for Advanced RAG System
Contains helper functions for text processing, chunking, and common operations
"""

import re
import logging
from typing import List, Dict, Any, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextProcessor:
    """Utility class for text processing operations"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\"\'\/]', '', text)
        
        # Remove multiple punctuation
        text = re.sub(r'[\.]{2,}', '.', text)
        text = re.sub(r'[\!]{2,}', '!', text)
        text = re.sub(r'[\?]{2,}', '?', text)
        
        return text.strip()
    
    @staticmethod
    def normalize_query(query: str) -> str:
        """Normalize user query for better retrieval"""
        if not query:
            return ""
        
        # Convert to lowercase
        query = query.lower()
        
        # Remove question words at the end
        query = re.sub(r'\b(what|how|when|where|why|who|which|whose)\?*$', '', query)
        
        # Remove common stop words that don't add semantic value
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}
        words = query.split()
        words = [word for word in words if word not in stop_words]
        
        return ' '.join(words).strip()
    
    @staticmethod
    def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text for BM25 retrieval"""
        if not text:
            return []
        
        # Simple keyword extraction based on word frequency
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'that', 'this', 'it', 'from', 'be', 'are', 'been', 'was', 'were', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'has', 'have', 'had', 'do', 'does', 'did'}
        words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count word frequency
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_words[:max_keywords]]
        
        return keywords

class DocumentChunker:
    """Utility class for document chunking"""
    
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        """Initialize chunker with specified parameters"""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        if not documents:
            return []
        
        chunks = []
        for i, doc in enumerate(documents):
            # Clean the document content first
            cleaned_content = TextProcessor.clean_text(doc.page_content)
            
            if not cleaned_content:
                logger.warning(f"Document {i} has no content after cleaning")
                continue
            
            # Create a temporary document with cleaned content
            temp_doc = Document(
                page_content=cleaned_content,
                metadata=doc.metadata
            )
            
            # Split the document
            doc_chunks = self.text_splitter.split_documents([temp_doc])
            
            # Add chunk metadata
            for j, chunk in enumerate(doc_chunks):
                chunk.metadata.update({
                    "chunk_id": f"doc_{i}_chunk_{j}",
                    "source_doc": i,
                    "chunk_index": j,
                    "total_chunks": len(doc_chunks)
                })
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def get_chunk_stats(self, chunks: List[Document]) -> Dict[str, Any]:
        """Get statistics about chunked documents"""
        if not chunks:
            return {"total_chunks": 0, "avg_chunk_size": 0, "min_chunk_size": 0, "max_chunk_size": 0}
        
        chunk_sizes = [len(chunk.page_content) for chunk in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
            "min_chunk_size": min(chunk_sizes),
            "max_chunk_size": max(chunk_sizes)
        }

class ConfidenceCalculator:
    """Utility class for calculating confidence scores"""
    
    @staticmethod
    def calculate_similarity_confidence(scores: List[float], threshold: float = 0.7) -> Dict[str, Any]:
        """Calculate confidence based on similarity scores"""
        if not scores:
            return {
                "avg_score": 0.0,
                "max_score": 0.0,
                "min_score": 0.0,
                "confidence": "low",
                "above_threshold": 0,
                "total_results": 0
            }
        
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        min_score = min(scores)
        above_threshold = sum(1 for score in scores if score >= threshold)
        
        # Determine confidence level
        if avg_score >= threshold and above_threshold >= len(scores) * 0.5:
            confidence = "high"
        elif avg_score >= threshold * 0.7 and above_threshold >= len(scores) * 0.3:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "avg_score": avg_score,
            "max_score": max_score,
            "min_score": min_score,
            "confidence": confidence,
            "above_threshold": above_threshold,
            "total_results": len(scores)
        }
    
    @staticmethod
    def should_retry_retrieval(confidence_data: Dict[str, Any]) -> bool:
        """Determine if retrieval should be retried based on confidence"""
        return confidence_data["confidence"] in ["low"]

class QueryRewriter:
    """Utility class for query rewriting"""
    
    @staticmethod
    def extract_key_terms(query: str) -> List[str]:
        """Extract key terms from query for rewriting"""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', query.lower())
        
        # Remove stop words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by', 'what', 'how', 'when', 'where', 'why', 'who', 'which', 'whose'}
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms
    
    @staticmethod
    def generate_alternative_queries(query: str, max_alternatives: int = 3) -> List[str]:
        """Generate alternative query formulations"""
        key_terms = QueryRewriter.extract_key_terms(query)
        
        if not key_terms:
            return [query]
        
        alternatives = [query]  # Original query
        
        # Alternative 1: Focus on key terms
        if len(key_terms) >= 2:
            alternatives.append(" ".join(key_terms[:3]))
        
        # Alternative 2: Expand with related concepts
        if len(key_terms) >= 1:
            alternatives.append(f"information about {key_terms[0]}")
        
        # Alternative 3: Question form
        if not query.endswith("?"):
            alternatives.append(f"What is {query}?")
        
        return alternatives[:max_alternatives]

def setup_logging(log_level: str = "INFO", log_format: str = None) -> None:
    """Setup logging configuration"""
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("rag_system.log")
        ]
    )

def validate_file_type(filename: str) -> bool:
    """Validate if file type is supported"""
    supported_extensions = {'.pdf', '.txt', '.md', '.docx', '.doc'}
    return any(filename.lower().endswith(ext) for ext in supported_extensions)

def format_source_display(chunks: List[Document]) -> List[Dict[str, str]]:
    """Format chunks for display in Streamlit"""
    formatted_sources = []
    
    for i, chunk in enumerate(chunks):
        source_info = {
            "chunk_id": chunk.metadata.get("chunk_id", f"chunk_{i}"),
            "source": chunk.metadata.get("source", "Unknown"),
            "content": chunk.page_content[:200] + "..." if len(chunk.page_content) > 200 else chunk.page_content,
            "score": chunk.metadata.get("score", 0.0)
        }
        formatted_sources.append(source_info)
    
    return formatted_sources
