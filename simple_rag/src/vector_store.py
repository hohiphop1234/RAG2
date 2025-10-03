"""
Vector Store Module

This module handles storing and searching through document embeddings.
It demonstrates the third step in RAG: vector similarity search.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import json
import pickle

# Vector database libraries
import chromadb
from chromadb.config import Settings
import faiss

from config import config
from logging_config import get_logger

logger = get_logger(__name__)

class VectorStore:
    """
    Handles vector storage and similarity search.
    
    This class demonstrates:
    1. How to store embeddings in a vector database
    2. How to perform similarity search
    3. Different vector database options (ChromaDB vs FAISS)
    4. Index management and persistence
    """
    
    def __init__(self, db_type: str = None, collection_name: str = None):
        """
        Initialize the vector store.
        
        Args:
            db_type: "chromadb" or "faiss"
            collection_name: Name for the collection/index
        """
        self.db_type = db_type or config.VECTOR_DB_TYPE
        self.collection_name = collection_name or config.COLLECTION_NAME
        self.db = None
        self.index = None
        self.metadata = []
        
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize the vector database based on type."""
        if self.db_type == "chromadb":
            self._initialize_chromadb()
        elif self.db_type == "faiss":
            self._initialize_faiss()
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB."""
        try:
            # Create ChromaDB client
            self.db = chromadb.PersistentClient(
                path=str(Path(config.MODELS_DIR) / "chromadb"),
                settings=Settings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            try:
                self.collection = self.db.get_collection(name=self.collection_name)
                logger.info(f"Loaded existing ChromaDB collection: {self.collection_name}")
            except Exception:  # Catch any exception when collection doesn't exist
                self.collection = self.db.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Vietnamese law documents"}
                )
                logger.info(f"Created new ChromaDB collection: {self.collection_name}")
                
        except Exception as e:
            logger.error(f"Error initializing ChromaDB: {str(e)}")
            raise
    
    def _initialize_faiss(self):
        """Initialize FAISS index."""
        try:
            self.index_file = Path(config.MODELS_DIR) / f"faiss_index_{self.collection_name}.index"
            self.metadata_file = Path(config.MODELS_DIR) / f"faiss_metadata_{self.collection_name}.pkl"
            
            # Load existing index if it exists
            if self.index_file.exists() and self.metadata_file.exists():
                self.index = faiss.read_index(str(self.index_file))
                with open(self.metadata_file, 'rb') as f:
                    self.metadata = pickle.load(f)
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            else:
                logger.info("No existing FAISS index found, will create new one")
                
        except Exception as e:
            logger.error(f"Error initializing FAISS: {str(e)}")
            raise
    
    def add_documents(self, embedded_chunks: List[Dict[str, Any]]) -> bool:
        """
        Add embedded documents to the vector store.
        
        Args:
            embedded_chunks: List of chunks with embeddings
            
        Returns:
            True if successful
        """
        if not embedded_chunks:
            logger.warning("No embedded chunks provided")
            return False
        
        try:
            if self.db_type == "chromadb":
                return self._add_to_chromadb(embedded_chunks)
            else:
                return self._add_to_faiss(embedded_chunks)
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return False
    
    def _add_to_chromadb(self, embedded_chunks: List[Dict[str, Any]]) -> bool:
        """Add documents to ChromaDB."""
        try:
            # Prepare data for ChromaDB
            ids = []
            embeddings = []
            documents = []
            metadatas = []
            
            for i, chunk in enumerate(embedded_chunks):
                ids.append(f"chunk_{i}")
                embeddings.append(chunk['embedding'])
                documents.append(chunk['text'])
                
                # Prepare metadata - handle different metadata structures
                chunk_metadata = chunk.get('metadata', {})
                metadata = {
                    'source_document': chunk_metadata.get('source_document', chunk.get('source_document', 'unknown')),
                    'source_filepath': chunk_metadata.get('source_filepath', chunk.get('source_filepath', 'unknown')),
                    'chunk_size': chunk_metadata.get('chunk_size', chunk.get('chunk_size', len(chunk['text']))),
                    'embedding_model': chunk.get('embedding_model', 'unknown')
                }
                
                # Add any additional metadata fields
                for key, value in chunk_metadata.items():
                    if key not in metadata and isinstance(value, (str, int, float, bool)):
                        metadata[key] = str(value)  # ChromaDB requires string values
                
                if 'article' in chunk and chunk['article']:
                    metadata['article'] = chunk['article']
                
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(embedded_chunks)} documents to ChromaDB")
            return True
            
        except Exception as e:
            logger.error(f"Error adding to ChromaDB: {str(e)}")
            return False
    
    def _add_to_faiss(self, embedded_chunks: List[Dict[str, Any]]) -> bool:
        """Add documents to FAISS index."""
        try:
            # Extract embeddings
            embeddings = np.array([chunk['embedding'] for chunk in embedded_chunks], dtype=np.float32)
            
            # Create index if it doesn't exist
            if self.index is None:
                dimension = len(embedded_chunks[0]['embedding'])
                self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)
                logger.info(f"Created new FAISS index with dimension {dimension}")
            
            # Add embeddings to index
            self.index.add(embeddings)
            
            # Store metadata
            for chunk in embedded_chunks:
                metadata = {
                    'text': chunk['text'],
                    'source_document': chunk['source_document'],
                    'source_filepath': chunk['source_filepath'],
                    'chunk_size': chunk['chunk_size'],
                    'embedding_model': chunk.get('embedding_model', 'unknown')
                }
                
                if 'article' in chunk and chunk['article']:
                    metadata['article'] = chunk['article']
                
                self.metadata.append(metadata)
            
            # Save index and metadata
            self._save_faiss_index()
            
            logger.info(f"Added {len(embedded_chunks)} documents to FAISS index")
            return True
            
        except Exception as e:
            logger.error(f"Error adding to FAISS: {str(e)}")
            return False
    
    def similarity_search(self, query_embedding: List[float], 
                         top_k: int = None) -> List[Dict[str, Any]]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Embedding of the query
            top_k: Number of results to return
            
        Returns:
            List of similar documents with scores
        """
        top_k = top_k or config.TOP_K_RESULTS
        
        try:
            if self.db_type == "chromadb":
                return self._search_chromadb(query_embedding, top_k)
            else:
                return self._search_faiss(query_embedding, top_k)
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    def _search_chromadb(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search using ChromaDB."""
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results['ids'][0])):
                result = {
                    'text': results['documents'][0][i],
                    'score': results['distances'][0][i],
                    'metadata': results['metadatas'][0][i]
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {str(e)}")
            return []
    
    def _search_faiss(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Search using FAISS."""
        try:
            if self.index is None or self.index.ntotal == 0:
                logger.warning("FAISS index is empty")
                return []
            
            # Convert query to numpy array
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Search
            scores, indices = self.index.search(query_vector, top_k)
            
            # Format results
            formatted_results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.metadata):
                    result = {
                        'text': self.metadata[idx]['text'],
                        'score': float(score),
                        'metadata': self.metadata[idx]
                    }
                    formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching FAISS: {str(e)}")
            return []
    
    def _save_faiss_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_file))
            
            with open(self.metadata_file, 'wb') as f:
                pickle.dump(self.metadata, f)
                
            logger.info(f"Saved FAISS index and metadata")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store."""
        stats = {
            'database_type': self.db_type,
            'collection_name': self.collection_name
        }
        
        if self.db_type == "chromadb":
            try:
                count = self.collection.count()
                stats['document_count'] = count
            except:
                stats['document_count'] = 0
        else:
            stats['document_count'] = self.index.ntotal if self.index else 0
        
        return stats
    
    def clear_database(self):
        """Clear all documents from the vector store."""
        try:
            if self.db_type == "chromadb":
                # Delete and recreate collection
                self.db.delete_collection(name=self.collection_name)
                self.collection = self.db.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Vietnamese law documents"}
                )
                logger.info("Cleared ChromaDB collection")
            else:
                # Clear FAISS index
                self.index = None
                self.metadata = []
                if self.index_file.exists():
                    self.index_file.unlink()
                if self.metadata_file.exists():
                    self.metadata_file.unlink()
                logger.info("Cleared FAISS index")
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")

def main():
    """Example usage of the VectorStore."""
    # Load embedded chunks
    embeddings_file = Path(config.MODELS_DIR) / "embeddings_openai.pkl"
    
    if not embeddings_file.exists():
        embeddings_file = Path(config.MODELS_DIR) / "embeddings_sentence-transformers.pkl"
    
    if not embeddings_file.exists():
        logger.error("No embeddings found. Run embeddings.py first.")
        return
    
    with open(embeddings_file, 'rb') as f:
        embedded_chunks = pickle.load(f)
    
    # Initialize vector store
    vector_store = VectorStore()
    
    # Add documents to vector store
    success = vector_store.add_documents(embedded_chunks)
    
    if success:
        stats = vector_store.get_stats()
        print(f"Vector store setup complete!")
        print(f"Database type: {stats['database_type']}")
        print(f"Document count: {stats['document_count']}")
        
        # Test search with a sample query
        from src.embeddings import EmbeddingGenerator
        
        # Create a sample query
        sample_query = "Luật giao thông đường bộ"
        
        # Generate embedding for the query
        embedding_gen = EmbeddingGenerator()
        query_embedding = embedding_gen.get_embedding(sample_query)
        
        # Search for similar documents
        results = vector_store.similarity_search(query_embedding, top_k=3)
        
        print(f"\nSample search for: '{sample_query}'")
        print(f"Found {len(results)} results:")
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. Score: {result['score']:.4f}")
            print(f"   Source: {result['metadata']['source_document']}")
            print(f"   Text: {result['text'][:200]}...")
    else:
        print("Failed to add documents to vector store")

if __name__ == "__main__":
    main()
