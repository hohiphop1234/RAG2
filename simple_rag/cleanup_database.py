"""
Database Cleanup Script

This script will delete all vector database data and cached files.
Use this when you want to start fresh or clear old data.
"""

import shutil
import os
from pathlib import Path
import sys

def confirm_deletion():
    """Ask user to confirm deletion."""
    print("⚠️  WARNING: This will delete all vector database data!")
    print("\nThe following will be deleted:")
    print("  - ChromaDB vector database")
    print("  - Processed documents cache")
    print("  - Embedding cache files")
    print("  - FAISS index files (if any)")
    print()
    
    response = input("Are you sure you want to continue? (yes/no): ").strip().lower()
    return response in ['yes', 'y']

def delete_chromadb():
    """Delete ChromaDB database."""
    chromadb_path = Path("models/chromadb")
    if chromadb_path.exists():
        try:
            shutil.rmtree(chromadb_path)
            print("✅ Deleted ChromaDB database")
            return True
        except Exception as e:
            print(f"❌ Error deleting ChromaDB: {e}")
            return False
    else:
        print("ℹ️  ChromaDB database not found (already deleted?)")
        return True

def delete_processed_docs():
    """Delete processed documents cache."""
    processed_path = Path("data/processed")
    if processed_path.exists():
        try:
            shutil.rmtree(processed_path)
            print("✅ Deleted processed documents cache")
            return True
        except Exception as e:
            print(f"❌ Error deleting processed docs: {e}")
            return False
    else:
        print("ℹ️  Processed documents cache not found")
        return True

def delete_embedding_cache():
    """Delete embedding cache files."""
    models_path = Path("models")
    deleted_count = 0
    
    if models_path.exists():
        for file in models_path.glob("embeddings_*.pkl"):
            try:
                file.unlink()
                deleted_count += 1
                print(f"✅ Deleted embedding cache: {file.name}")
            except Exception as e:
                print(f"❌ Error deleting {file.name}: {e}")
    
    if deleted_count == 0:
        print("ℹ️  No embedding cache files found")
    
    return True

def delete_faiss_index():
    """Delete FAISS index files."""
    models_path = Path("models")
    deleted_count = 0
    
    if models_path.exists():
        for file in models_path.glob("faiss_*"):
            try:
                file.unlink()
                deleted_count += 1
                print(f"✅ Deleted FAISS file: {file.name}")
            except Exception as e:
                print(f"❌ Error deleting {file.name}: {e}")
    
    if deleted_count == 0:
        print("ℹ️  No FAISS index files found")
    
    return True

def delete_temp_uploads():
    """Delete temporary uploaded files."""
    temp_path = Path("temp_uploads")
    if temp_path.exists():
        for file in temp_path.iterdir():
            if file.is_file():
                try:
                    file.unlink()
                    print(f"✅ Deleted temp file: {file.name}")
                except Exception as e:
                    print(f"❌ Error deleting {file.name}: {e}")
    else:
        print("ℹ️  No temporary uploads found")
    
    return True

def main():
    """Main cleanup function."""
    print("=" * 60)
    print("🗑️  RAG System Database Cleanup Tool")
    print("=" * 60)
    print()
    
    # Check if running from correct directory
    if not Path("config.py").exists():
        print("❌ Error: Please run this script from the simple_rag directory")
        sys.exit(1)
    
    # Confirm with user
    if not confirm_deletion():
        print("\n❌ Cleanup cancelled by user")
        sys.exit(0)
    
    print("\n🔄 Starting cleanup...")
    print()
    
    # Perform cleanup operations
    success = True
    success &= delete_chromadb()
    success &= delete_processed_docs()
    success &= delete_embedding_cache()
    success &= delete_faiss_index()
    success &= delete_temp_uploads()
    
    print()
    print("=" * 60)
    if success:
        print("✅ Cleanup completed successfully!")
        print()
        print("Next steps:")
        print("  1. Add your documents to data/raw/")
        print("  2. Run: python main.py setup")
        print("  3. Or upload documents via web interface")
    else:
        print("⚠️  Cleanup completed with some errors")
        print("Check the messages above for details")
    print("=" * 60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Cleanup cancelled by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
