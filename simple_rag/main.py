"""
Simple RAG System - Vietnamese Law Chatbot

A minimal implementation of RAG for Vietnamese traffic and land law.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import config
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.rag_pipeline import RAGPipeline
from logging_config import setup_logging

def setup_rag_system():
    """Set up the complete RAG system from scratch."""
    print("Setting up RAG system...")
    
    try:
        # Step 1: Process documents
        print("Step 1: Processing documents...")
        processor = DocumentProcessor()
        documents = processor.load_documents()
        
        if not documents:
            print("No documents found. Please add documents to data/raw/ directory.")
            return False
        
        chunks = processor.chunk_documents(documents)
        processor.save_processed_documents(chunks)
        print(f"Processed {len(documents)} documents into {len(chunks)} chunks")
        
        # Step 2: Generate embeddings
        print("Step 2: Generating embeddings...")
        try:
            embedding_gen = EmbeddingGenerator(model_type="openai")
        except ValueError:
            print("OpenAI not available, using sentence-transformers...")
            embedding_gen = EmbeddingGenerator(model_type="sentence-transformers")
        
        embedded_chunks = embedding_gen.embed_documents(chunks)
        embedding_gen.save_embeddings(embedded_chunks)
        print(f"Generated embeddings for {len(embedded_chunks)} chunks")
        
        # Step 3: Build vector store
        print("Step 3: Building vector store...")
        vector_store = VectorStore()
        success = vector_store.add_documents(embedded_chunks)
        
        if success:
            stats = vector_store.get_stats()
            print(f"Vector store built with {stats['document_count']} documents")
        else:
            print("Failed to build vector store")
            return False
        
        print("RAG system setup complete!")
        return True
        
    except Exception as e:
        print(f"Error setting up RAG system: {str(e)}")
        return False

def run_web_interface():
    """Run the Streamlit web interface."""
    import subprocess
    import sys
    
    print("Starting web interface...")
    
    try:
        # Run streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/web_interface.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("Web interface stopped by user")
    except Exception as e:
        print(f"Error running web interface: {str(e)}")

def main():
    """Main function with simple interface."""
    # Setup logging
    setup_logging()
    
    print("🇻🇳 Simple RAG System for Vietnamese Law")
    print("=" * 50)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "setup":
            success = setup_rag_system()
            sys.exit(0 if success else 1)
        
        elif command == "web":
            run_web_interface()
        
        else:
            print("Usage: python main.py [setup|web]")
    else:
        print("Usage: python main.py [setup|web]")
        print("  setup - Set up the RAG system")
        print("  web   - Run web interface")

if __name__ == "__main__":
    main()