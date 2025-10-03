"""
Configuration settings for the Simple RAG system.
Edit these settings to customize your RAG implementation.
"""

import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class RAGConfig:
    """Configuration class for RAG system settings."""
    
    # Document Processing
    CHUNK_SIZE: int = 1000  # Size of text chunks in characters
    CHUNK_OVERLAP: int = 200  # Overlap between chunks
    MAX_CHUNKS_PER_DOCUMENT: int = 50  # Limit chunks per document
    
    # Embedding Settings
    EMBEDDING_PROVIDER: str = "ollama"  # "openai" or "ollama"
    EMBEDDING_MODEL: str = "mxbai-embed-large"  # OpenAI model or Ollama model
    EMBEDDING_DIMENSION: int = 1024  # Embedding dimension (768 for nomic, 1024 for mxbai)
    BATCH_SIZE: int = 100  # Batch size for embedding generation
    
    # Vector Database
    VECTOR_DB_TYPE: str = "chromadb"  # "chromadb" or "faiss"
    COLLECTION_NAME: str = "vietnamese_law_documents"
    SIMILARITY_THRESHOLD: float = 0.3  # Minimum similarity score (lowered for better retrieval)
    TOP_K_RESULTS: int = 5  # Number of relevant chunks to retrieve
    
    # LLM Settings
    LLM_PROVIDER: str = "ollama"  # "openai" or "ollama"
    LLM_MODEL: str = "deepseek-r1"  # OpenAI model or Ollama model name
    MAX_TOKENS: int = 1000  # Maximum tokens for response
    TEMPERATURE: float = 0.7  # Response creativity
    SYSTEM_PROMPT: str = """Bạn là một trợ lý pháp lý chuyên về luật giao thông và đất đai Việt Nam. 
    Hãy trả lời câu hỏi dựa trên thông tin được cung cấp. Nếu không có thông tin liên quan, 
    hãy nói rõ rằng bạn không thể trả lời dựa trên tài liệu hiện có."""
    
    # File Paths
    DATA_DIR: str = "data"
    RAW_DOCS_DIR: str = "data/raw"
    PROCESSED_DOCS_DIR: str = "data/processed"
    MODELS_DIR: str = "models"
    
    
    # API Keys and URLs (load from environment)
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Web Interface
    WEB_TITLE: str = "Chatbot pháp luật Việt Nam"
    WEB_DESCRIPTION: str = "Hỏi câu hỏi về pháp luật Việt Nam"
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.LLM_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            print("Warning: OPENAI_API_KEY not found in environment variables")
        elif self.LLM_PROVIDER == "ollama":
            print(f"Using Ollama with model: {self.LLM_MODEL} at {self.OLLAMA_BASE_URL}")
        
        # Create directories if they don't exist
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.RAW_DOCS_DIR, exist_ok=True)
        os.makedirs(self.PROCESSED_DOCS_DIR, exist_ok=True)
        os.makedirs(self.MODELS_DIR, exist_ok=True)

# Global configuration instance
config = RAGConfig()
