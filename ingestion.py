"""
Document Ingestion Module for Advanced RAG System
Handles document loading, processing, and vector store creation
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

# LangChain imports
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# Local imports
from config import config
from utils import DocumentChunker, validate_file_type, setup_logging

# Configure logging
logger = logging.getLogger(__name__)

class DocumentLoader:
    """Handles loading of different document types"""
    
    @staticmethod
    def load_pdf(file_path: str) -> List[Document]:
        """Load PDF document using PyPDFLoader"""
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            logger.info(f"Successfully loaded PDF: {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            return []
    
    @staticmethod
    def load_text(file_path: str) -> List[Document]:
        """Load text document"""
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents = loader.load()
            logger.info(f"Successfully loaded text file: {file_path}")
            return documents
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                loader = TextLoader(file_path, encoding='latin-1')
                documents = loader.load()
                logger.info(f"Successfully loaded text file with latin-1 encoding: {file_path}")
                return documents
            except Exception as e:
                logger.error(f"Error loading text file {file_path}: {str(e)}")
                return []
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            return []
    
    @staticmethod
    def load_markdown(file_path: str) -> List[Document]:
        """Load Markdown document"""
        try:
            loader = UnstructuredMarkdownLoader(file_path)
            documents = loader.load()
            logger.info(f"Successfully loaded Markdown file: {file_path}")
            return documents
        except Exception as e:
            logger.error(f"Error loading Markdown file {file_path}: {str(e)}")
            return []
    
    @classmethod
    def load_document(cls, file_path: str) -> List[Document]:
        """Load document based on file type"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
        
        if not validate_file_type(file_path):
            logger.error(f"Unsupported file type: {file_path}")
            return []
        
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.pdf':
            return cls.load_pdf(file_path)
        elif file_extension in ['.txt', '.text']:
            return cls.load_text(file_path)
        elif file_extension in ['.md', '.markdown']:
            return cls.load_markdown(file_path)
        else:
            logger.error(f"Unsupported file extension: {file_extension}")
            return []

class EmbeddingGenerator:
    """Handles embedding generation"""
    
    def __init__(self, use_openai: bool = False):
        """Initialize embedding generator"""
        self.use_openai = use_openai
        self.embeddings = self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        try:
            if self.use_openai and config.OPENAI_API_KEY:
                logger.info("Using OpenAI embeddings")
                return OpenAIEmbeddings(
                    openai_api_key=config.OPENAI_API_KEY,
                    model="text-embedding-ada-002"
                )
            else:
                logger.info("Using HuggingFace embeddings")
                return HuggingFaceEmbeddings(
                    model_name=config.EMBEDDING_MODEL,
                    model_kwargs={'device': 'cpu'},
                    encode_kwargs={'normalize_embeddings': True}
                )
        except Exception as e:
            logger.error(f"Error initializing embeddings: {str(e)}")
            raise
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
        try:
            embedding = self.embeddings.embed_query(query)
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            return []

class VectorStoreManager:
    """Manages FAISS vector store operations"""
    
    def __init__(self, embedding_generator: EmbeddingGenerator):
        """Initialize vector store manager"""
        self.embedding_generator = embedding_generator
        self.vector_store = None
        self.index_path = config.VECTOR_STORE_DIR
    
    def create_vector_store(self, documents: List[Document]) -> Optional[FAISS]:
        """Create FAISS vector store from documents"""
        if not documents:
            logger.error("No documents provided for vector store creation")
            return None
        
        try:
            # Create vector store
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embedding_generator.embeddings
            )
            
            # Save the vector store
            self.save_vector_store()
            
            logger.info(f"Successfully created vector store with {len(documents)} documents")
            return self.vector_store
            
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")
            return None
    
    def load_vector_store(self) -> Optional[FAISS]:
        """Load existing vector store from disk"""
        try:
            if os.path.exists(self.index_path):
                self.vector_store = FAISS.load_local(
                    self.index_path,
                    self.embedding_generator.embeddings,
                    allow_dangerous_deserialization=True
                )
                logger.info("Successfully loaded existing vector store")
                return self.vector_store
            else:
                logger.info("No existing vector store found")
                return None
        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            return None
    
    def save_vector_store(self) -> bool:
        """Save vector store to disk"""
        if not self.vector_store:
            logger.error("No vector store to save")
            return False
        
        try:
            os.makedirs(self.index_path, exist_ok=True)
            self.vector_store.save_local(self.index_path)
            logger.info(f"Successfully saved vector store to {self.index_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving vector store: {str(e)}")
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """Add documents to existing vector store"""
        if not documents:
            logger.error("No documents to add")
            return False
        
        try:
            if self.vector_store is None:
                return self.create_vector_store(documents) is not None
            
            self.vector_store.add_documents(documents)
            self.save_vector_store()
            logger.info(f"Successfully added {len(documents)} documents to vector store")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def get_vector_store(self) -> Optional[FAISS]:
        """Get current vector store"""
        return self.vector_store
    
    def get_store_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        if not self.vector_store:
            return {"status": "No vector store loaded"}
        
        try:
            # Get basic stats
            index = self.vector_store.index
            doc_count = index.ntotal if hasattr(index, 'ntotal') else "Unknown"
            
            return {
                "status": "Vector store loaded",
                "document_count": doc_count,
                "embedding_dimension": config.EMBEDDING_DIMENSION,
                "index_type": type(index).__name__
            }
        except Exception as e:
            logger.error(f"Error getting vector store stats: {str(e)}")
            return {"status": "Error retrieving stats", "error": str(e)}

class DocumentIngestionPipeline:
    """Main pipeline for document ingestion"""
    
    def __init__(self, use_openai_embeddings: bool = False):
        """Initialize the ingestion pipeline"""
        self.document_loader = DocumentLoader()
        self.embedding_generator = EmbeddingGenerator(use_openai_embeddings)
        self.vector_store_manager = VectorStoreManager(self.embedding_generator)
        self.chunker = DocumentChunker(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        # Try to load existing vector store
        self.vector_store_manager.load_vector_store()
    
    def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """Ingest a single file"""
        logger.info(f"Starting ingestion for file: {file_path}")
        
        # Validate file
        if not os.path.exists(file_path):
            return {"success": False, "error": "File not found"}
        
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        if file_size > config.MAX_FILE_SIZE_MB:
            return {"success": False, "error": f"File too large. Max size: {config.MAX_FILE_SIZE_MB}MB"}
        
        try:
            # Load document
            documents = self.document_loader.load_document(file_path)
            if not documents:
                return {"success": False, "error": "Failed to load document"}
            
            # Add file metadata
            file_name = os.path.basename(file_path)
            for doc in documents:
                doc.metadata.update({
                    "source_file": file_name,
                    "file_path": file_path,
                    "file_size": file_size
                })
            
            # Chunk documents
            chunks = self.chunker.chunk_documents(documents)
            if not chunks:
                return {"success": False, "error": "Failed to chunk documents"}
            
            # Add to vector store
            success = self.vector_store_manager.add_documents(chunks)
            if not success:
                return {"success": False, "error": "Failed to add documents to vector store"}
            
            # Get stats
            chunk_stats = self.chunker.get_chunk_stats(chunks)
            store_stats = self.vector_store_manager.get_store_stats()
            
            logger.info(f"Successfully ingested file: {file_name}")
            return {
                "success": True,
                "file_name": file_name,
                "chunks_created": len(chunks),
                "chunk_stats": chunk_stats,
                "store_stats": store_stats
            }
            
        except Exception as e:
            logger.error(f"Error ingesting file {file_path}: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def ingest_directory(self, directory_path: str) -> Dict[str, Any]:
        """Ingest all supported files in a directory"""
        logger.info(f"Starting directory ingestion for: {directory_path}")
        
        if not os.path.exists(directory_path):
            return {"success": False, "error": "Directory not found"}
        
        results = {
            "success": True,
            "files_processed": 0,
            "files_failed": 0,
            "total_chunks": 0,
            "failed_files": []
        }
        
        # Get all supported files
        supported_files = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                if validate_file_type(file_path):
                    supported_files.append(file_path)
        
        logger.info(f"Found {len(supported_files)} supported files")
        
        # Process each file
        for file_path in supported_files:
            result = self.ingest_file(file_path)
            if result["success"]:
                results["files_processed"] += 1
                results["total_chunks"] += result.get("chunks_created", 0)
            else:
                results["files_failed"] += 1
                results["failed_files"].append({
                    "file": os.path.basename(file_path),
                    "error": result["error"]
                })
        
        logger.info(f"Directory ingestion completed: {results['files_processed']} processed, {results['files_failed']} failed")
        return results
    
    def clear_vector_store(self) -> bool:
        """Clear the vector store"""
        try:
            self.vector_store_manager.vector_store = None
            if os.path.exists(self.index_path):
                import shutil
                shutil.rmtree(self.index_path)
            logger.info("Vector store cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing vector store: {str(e)}")
            return False
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            "vector_store_stats": self.vector_store_manager.get_store_stats(),
            "chunker_config": {
                "chunk_size": self.chunker.chunk_size,
                "chunk_overlap": self.chunker.chunk_overlap
            },
            "embedding_model": config.EMBEDDING_MODEL,
            "using_openai": self.embedding_generator.use_openai
        }
