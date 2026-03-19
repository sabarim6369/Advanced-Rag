"""
Configuration module for Advanced RAG System
Contains API keys, constants, and system settings
"""

import os
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration class for the Advanced RAG system"""
    
    # API Keys (loaded from environment variables)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    
    # Groq Configuration
    GROQ_MODEL = "llama3-8b-8192"
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    
    # Embedding Configuration
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION = 384
    
    # Document Processing
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 150
    MAX_FILE_SIZE_MB = 50
    
    # Retrieval Configuration
    FAISS_INDEX_TYPE = "IndexFlatIP"
    TOP_K_RETRIEVAL = 10
    TOP_K_RERANK = 5
    CONFIDENCE_THRESHOLD = 0.7
    
    # Reranking Configuration
    RERANKER_MODEL = "sentence-transformers/ms-marco-MiniLM-L-6-v2"
    RERANKER_BATCH_SIZE = 32
    
    # File Storage
    DATA_DIR = "data"
    VECTOR_STORE_DIR = "data/vector_store"
    UPLOADS_DIR = "data/uploads"
    
    # Logging Configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Security Configuration
    MAX_QUERY_LENGTH = 1000
    BLOCKED_PATTERNS = [
        "ignore previous instructions",
        "ignore all instructions",
        "reveal system prompt",
        "show me your prompt",
        "what are your instructions",
        "system prompt",
        "developer mode",
        "jailbreak",
        "override"
    ]
    
    # Streamlit Configuration
    STREAMLIT_TITLE = "Advanced RAG System"
    STREAMLIT_LAYOUT = "centered"
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        status = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        if not cls.GROQ_API_KEY:
            status["errors"].append("GROQ_API_KEY is required")
            status["valid"] = False
        
        if not cls.OPENAI_API_KEY:
            status["warnings"].append("OPENAI_API_KEY not set - using HuggingFace embeddings")
        
        return status
    
    @classmethod
    def get_groq_config(cls) -> Dict[str, str]:
        """Get Groq API configuration"""
        return {
            "api_key": cls.GROQ_API_KEY,
            "base_url": cls.GROQ_BASE_URL,
            "model": cls.GROQ_MODEL
        }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.DATA_DIR,
            cls.VECTOR_STORE_DIR,
            cls.UPLOADS_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

# Global configuration instance
config = Config()
