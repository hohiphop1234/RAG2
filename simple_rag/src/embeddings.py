"""
Embeddings Module

This module handles the conversion of text to vector embeddings.
It demonstrates the second step in RAG: creating numerical representations of text.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import pickle

# Embedding libraries
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

from config import config
from logging_config import get_logger

logger = get_logger(__name__)

class EmbeddingGenerator:
    """
    Generates embeddings for text chunks and queries.
    
    This class demonstrates:
    1. How to convert text to numerical vectors
    2. Different embedding models (OpenAI vs sentence-transformers)
    3. Batch processing for efficiency
    4. Caching embeddings for reuse
    """
    
    def __init__(self, model_type: str = None):
        """
        Initialize the embedding generator.
        
        Args:
            model_type: "openai", "sentence-transformers", or "ollama" (default from config)
        """
        self.model_type = model_type or config.EMBEDDING_PROVIDER
        self.model = None
        self.tokenizer = None
        self.embedding_cache = {}
        self.ollama_client = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model based on type."""
        if self.model_type == "openai":
            if not OPENAI_AVAILABLE:
                logger.warning("OpenAI not available, falling back to Ollama")
                self.model_type = "ollama"
                self._initialize_ollama()
            else:
                self._initialize_openai()
        elif self.model_type == "sentence-transformers":
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                logger.warning("Sentence-transformers not available, falling back to Ollama")
                self.model_type = "ollama"
                self._initialize_ollama()
            else:
                self._initialize_sentence_transformers()
        elif self.model_type == "ollama":
            self._initialize_ollama()
        else:
            logger.warning(f"Unsupported model type: {self.model_type}, falling back to Ollama")
            self.model_type = "ollama"
            self._initialize_ollama()
    
    def _initialize_openai(self):
        """Initialize OpenAI embedding model."""
        if not config.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        openai.api_key = config.OPENAI_API_KEY
        self.model = config.EMBEDDING_MODEL
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
        except KeyError:
            # Fallback to cl100k_base for newer models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        logger.info(f"Initialized OpenAI embedding model: {self.model}")
    
    def _initialize_sentence_transformers(self):
        """Initialize sentence-transformers model."""
        # Use a multilingual model that supports Vietnamese
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Initialized sentence-transformers model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading sentence-transformers model: {str(e)}")
            raise
    
    def _initialize_ollama(self):
        """Initialize Ollama embedding model."""
        try:
            import ollama
            self.ollama_client = ollama.Client(host=config.OLLAMA_BASE_URL)
            self.model = config.EMBEDDING_MODEL
            
            # Test the model
            test_result = self.ollama_client.embeddings(model=self.model, prompt="test")
            logger.info(f"Initialized Ollama embedding model: {self.model}")
            logger.info(f"Embedding dimension: {len(test_result['embedding'])}")
            
        except ImportError:
            raise ImportError("Ollama library not found. Install with: pip install ollama")
        except Exception as e:
            logger.error(f"Error initializing Ollama embedding model: {str(e)}")
            raise
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            List of embedding values
        """
        # Check cache first
        cache_key = f"{self.model_type}_{hash(text)}"
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        if self.model_type == "openai":
            embedding = self._get_openai_embedding(text)
        elif self.model_type == "ollama":
            embedding = self._get_ollama_embedding(text)
        else:
            embedding = self._get_sentence_transformer_embedding(text)
        
        # Cache the result
        self.embedding_cache[cache_key] = embedding
        return embedding
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Get embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if self.model_type == "openai":
            return self._get_openai_embeddings_batch(texts)
        elif self.model_type == "ollama":
            return self._get_ollama_embeddings_batch(texts)
        else:
            return self._get_sentence_transformer_embeddings_batch(texts)
    
    def _get_openai_embedding(self, text: str) -> List[float]:
        """Get embedding using OpenAI API."""
        try:
            # Truncate text if it's too long
            text = self._truncate_text_for_openai(text)
            
            response = openai.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error getting OpenAI embedding: {str(e)}")
            raise
    
    def _get_openai_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts using OpenAI API."""
        try:
            # Truncate texts and filter out empty ones
            processed_texts = []
            for text in texts:
                truncated = self._truncate_text_for_openai(text)
                if truncated.strip():
                    processed_texts.append(truncated)
            
            if not processed_texts:
                return []
            
            # Process in batches to avoid API limits
            all_embeddings = []
            batch_size = config.BATCH_SIZE
            
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                
                response = openai.embeddings.create(
                    model=self.model,
                    input=batch
                )
                
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
            
            return all_embeddings
        except Exception as e:
            logger.error(f"Error getting OpenAI embeddings batch: {str(e)}")
            raise
    
    def _get_sentence_transformer_embedding(self, text: str) -> List[float]:
        """Get embedding using sentence-transformers."""
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error getting sentence-transformer embedding: {str(e)}")
            raise
    
    def _get_sentence_transformer_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts using sentence-transformers."""
        try:
         
            processed_texts = [text for text in texts if text.strip()]
            
            if not processed_texts:
                return []
            
            embeddings = self.model.encode(processed_texts, convert_to_tensor=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error getting sentence-transformer embeddings batch: {str(e)}")
            raise
    
    def _get_ollama_embedding(self, text: str) -> List[float]:
        """Get embedding using Ollama API."""
        try:
            result = self.ollama_client.embeddings(model=self.model, prompt=text)
            return result['embedding']
        except Exception as e:
            logger.error(f"Error getting Ollama embedding: {str(e)}")
            raise
    
    def _get_ollama_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts using Ollama API."""
        try:
            # Filter out empty texts
            processed_texts = [text for text in texts if text.strip()]
            
            if not processed_texts:
                return []
            
            # Ollama doesn't have batch API, so we'll process one by one
            # This is less efficient but works with the current Ollama API
            embeddings = []
            for i, text in enumerate(processed_texts):
                if i % 10 == 0:  # Log progress every 10 items
                    logger.info(f"Processing embedding {i+1}/{len(processed_texts)}")
                
                result = self.ollama_client.embeddings(model=self.model, prompt=text)
                embeddings.append(result['embedding'])
            
            return embeddings
        except Exception as e:
            logger.error(f"Error getting Ollama embeddings batch: {str(e)}")
            raise
    
    def _truncate_text_for_openai(self, text: str) -> str:
        """Truncate text to fit OpenAI's token limit."""
        if not self.tokenizer:
            return text
        
        # OpenAI's limit is 8192 tokens for most embedding models
        max_tokens = 8000  # Leave some buffer
        
        tokens = self.tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate and decode back to text
        truncated_tokens = tokens[:max_tokens]
        return self.tokenizer.decode(truncated_tokens)
    
    def embed_documents(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add embeddings to document chunks.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            List of chunks with embeddings added
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Extract texts for batch processing
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings in batches
        embeddings = self.get_embeddings_batch(texts)
        
        # Add embeddings to chunks
        embedded_chunks = []
        for i, chunk in enumerate(chunks):
            if i < len(embeddings):
                chunk_with_embedding = chunk.copy()
                chunk_with_embedding['embedding'] = embeddings[i]
                chunk_with_embedding['embedding_model'] = self.model_type
                embedded_chunks.append(chunk_with_embedding)
            else:
                logger.warning(f"No embedding generated for chunk {i}")
        
        logger.info(f"Successfully generated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    def save_embeddings(self, embedded_chunks: List[Dict[str, Any]], 
                       output_file: str = None) -> str:
        """Save embeddings to file for later use."""
        if output_file is None:
            output_file = Path(config.MODELS_DIR) / f"embeddings_{self.model_type}.pkl"
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as pickle for efficiency
        with open(output_path, 'wb') as f:
            pickle.dump(embedded_chunks, f)
        
        logger.info(f"Saved embeddings to {output_path}")
        return str(output_path)
    
    def load_embeddings(self, input_file: str) -> List[Dict[str, Any]]:
        """Load embeddings from file."""
        input_path = Path(input_file)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Embeddings file not found: {input_file}")
        
        with open(input_path, 'rb') as f:
            embedded_chunks = pickle.load(f)
        
        logger.info(f"Loaded {len(embedded_chunks)} embedded chunks from {input_file}")
        return embedded_chunks
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        if self.model_type == "openai":
            return config.EMBEDDING_DIMENSION
        elif self.model_type == "ollama":
            return config.EMBEDDING_DIMENSION
        else:
            # Get dimension from a sample embedding
            sample_embedding = self.get_embedding("sample text")
            return len(sample_embedding)
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")

def main():
    """Example usage of the EmbeddingGenerator."""
    # Load processed chunks
    chunks_file = Path(config.PROCESSED_DOCS_DIR) / "processed_chunks.json"
    
    if not chunks_file.exists():
        logger.error("No processed chunks found. Run document_processor.py first.")
        return
    
    with open(chunks_file, 'r', encoding='utf-8') as f:
        chunks = json.load(f)
    
    # Initialize embedding generator based on config
    try:
        generator = EmbeddingGenerator()  # Will use config.EMBEDDING_PROVIDER
    except Exception as e:
        logger.warning(f"{config.EMBEDDING_PROVIDER} not available: {str(e)}")
        logger.info("Falling back to sentence-transformers...")
        try:
            generator = EmbeddingGenerator(model_type="sentence-transformers")
        except Exception as e2:
            logger.error(f"All embedding providers failed: {str(e2)}")
            raise
    
    # Generate embeddings
    embedded_chunks = generator.embed_documents(chunks)
    
    # Save embeddings
    output_file = generator.save_embeddings(embedded_chunks)
    
    print(f"Embedding generation complete!")
    print(f"Generated embeddings for {len(embedded_chunks)} chunks")
    print(f"Embedding dimension: {generator.get_embedding_dimension()}")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
