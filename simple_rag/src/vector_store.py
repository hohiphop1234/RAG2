"""
Vector Store Module

This module handles storing and searching through document embeddings.
It demonstrates the third step in RAG: vector similarity search.
"""

from typing import List, Dict, Any
from pathlib import Path
import pickle

# Vector database libraries (Chroma only)
import chromadb
from chromadb.config import Settings

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
        """Initialize the vector database (Chroma-only)."""
        if self.db_type != "chromadb":
            raise NotImplementedError("Chroma-only build: set VECTOR_DB_TYPE='chromadb' in config.py")
        self._initialize_chromadb()
    
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
        """FAISS is disabled for Chroma-only configuration."""
        raise NotImplementedError("FAISS support is disabled. Set VECTOR_DB_TYPE='chromadb'.")
    
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
            return self._add_to_chromadb(embedded_chunks)
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
                # Build stable, unique ID using document name and chunk index if available
                doc_name = (chunk.get('source_document') or
                            chunk.get('metadata', {}).get('source_document') or 'unknown')
                chunk_idx = (chunk.get('metadata', {}).get('chunk_index')
                             if isinstance(chunk.get('metadata'), dict) else None)
                if chunk_idx is not None:
                    unique_id = f"{doc_name}__{chunk_idx}"
                else:
                    unique_id = f"{doc_name}__{i}"
                ids.append(unique_id)
                embeddings.append(chunk['embedding'])
                documents.append(chunk['text'])
                
                # Prepare metadata - handle different metadata structures
                chunk_metadata = chunk.get('metadata', {})
                metadata = {
                    'source_document': chunk_metadata.get('source_document', chunk.get('source_document', 'unknown')),
                    'source_filepath': chunk_metadata.get('source_filepath', chunk.get('source_filepath', 'unknown')),
                    'chunk_size': chunk_metadata.get('chunk_size', chunk.get('chunk_size', len(chunk['text']))),
                    'chunk_index': chunk_metadata.get('chunk_index'),
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
        """FAISS is disabled for Chroma-only configuration."""
        raise NotImplementedError("FAISS support is disabled. Set VECTOR_DB_TYPE='chromadb'.")
    
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
            return self._search_chromadb(query_embedding, top_k)
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
        """FAISS is disabled for Chroma-only configuration."""
        raise NotImplementedError("FAISS support is disabled. Set VECTOR_DB_TYPE='chromadb'.")
    
    def _save_faiss_index(self):
        """FAISS is disabled for Chroma-only configuration."""
        logger.warning("_save_faiss_index called but FAISS is disabled.")
    
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
            stats['document_count'] = 0
        
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
                # FAISS disabled; nothing to clear
                logger.info("FAISS is disabled; nothing to clear")
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
