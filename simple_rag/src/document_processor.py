"""
Document Processing Module

This module handles loading and processing Vietnamese law documents.
It demonstrates the first step in RAG: preparing documents for embedding.
"""

import re
import json
import logging
from typing import List, Dict, Any
from pathlib import Path

# Document processing libraries
import PyPDF2
from docx import Document
from bs4 import BeautifulSoup
import unicodedata

from config import config
from logging_config import get_logger

logger = get_logger(__name__)

class DocumentProcessor:
    """
    Handles loading and processing of Vietnamese law documents.
    
    This class demonstrates:
    1. How to load different document formats (PDF, DOCX, TXT)
    2. How to clean and normalize Vietnamese text
    3. How to chunk documents for embedding
    """
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt', '.html']
    
    def load_documents(self, directory: str = None) -> List[Dict[str, Any]]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory: Path to directory containing documents
            
        Returns:
            List of document dictionaries with metadata
        """
        if directory is None:
            directory = config.RAW_DOCS_DIR
            
        documents = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            logger.warning(f"Directory {directory} does not exist")
            return documents
        
        for file_path in directory_path.iterdir():
            if file_path.suffix.lower() in self.supported_formats:
                try:
                    doc = self._load_single_document(file_path)
                    if doc:
                        documents.append(doc)
                        logger.info(f"Loaded document: {file_path.name}")
                except Exception as e:
                    logger.error(f"Error loading {file_path.name}: {str(e)}")
        
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def _load_single_document(self, file_path: Path) -> Dict[str, Any]:
        """Load a single document based on its file extension."""
        content = ""
        
        if file_path.suffix.lower() == '.pdf':
            content = self._extract_pdf_text(file_path)
        elif file_path.suffix.lower() == '.docx':
            content = self._extract_docx_text(file_path)
        elif file_path.suffix.lower() == '.txt':
            content = self._extract_txt_text(file_path)
        elif file_path.suffix.lower() == '.html':
            content = self._extract_html_text(file_path)
        
        if not content.strip():
            return None
        
        return {
            'filename': file_path.name,
            'filepath': str(file_path),
            'content': content,
            'file_type': file_path.suffix.lower(),
            'size': len(content)
        }
    
    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from PDF files."""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
        return text
    
    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX files."""
        text = ""
        try:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {str(e)}")
        return text
    
    def _extract_txt_text(self, file_path: Path) -> str:
        """Extract text from TXT files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encodings
            for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            logger.error(f"Could not decode text file: {file_path}")
            return ""
    
    def _extract_html_text(self, file_path: Path) -> str:
        """Extract text from HTML files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file.read(), 'html.parser')
                return soup.get_text()
        except Exception as e:
            logger.error(f"Error extracting HTML text: {str(e)}")
            return ""
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and normalize Vietnamese text.
        
        This function demonstrates text preprocessing techniques:
        1. Unicode normalization
        2. Whitespace cleaning
        3. Vietnamese-specific cleaning
        """
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Clean Vietnamese law-specific patterns
        text = re.sub(r'Điều\s+(\d+)', r'Điều \1', text)  # Standardize article format
        text = re.sub(r'Chương\s+([IVX]+)', r'Chương \1', text)  # Standardize chapter format
        
        # Remove page numbers and headers/footers
        text = re.sub(r'Trang\s+\d+', '', text)
        text = re.sub(r'Page\s+\d+', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        
        return text.strip()
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Split documents into smaller chunks for embedding.
        
        This function demonstrates document chunking strategies:
        1. Fixed-size chunking
        2. Semantic chunking (by articles/sections)
        3. Overlap handling
        """
        chunks = []
        
        for doc in documents:
            doc_chunks = self._chunk_single_document(doc)
            chunks.extend(doc_chunks)
        
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks
    
    def _chunk_single_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a single document using multiple strategies."""
        content = self.preprocess_text(document['content'])
        
        # Try semantic chunking first (by articles/sections)
        semantic_chunks = self._semantic_chunking(content, document)
        if semantic_chunks:
            return semantic_chunks
        
        # Fall back to fixed-size chunking
        return self._fixed_size_chunking(content, document)
    
    def _semantic_chunking(self, content: str, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk document by semantic boundaries (articles, sections)."""
        chunks = []
        
        # Split by articles (Điều)
        articles = re.split(r'(Điều\s+\d+)', content)
        
        current_chunk = ""
        current_article = ""
        
        for i, part in enumerate(articles):
            if re.match(r'Điều\s+\d+', part.strip()):
                # Save previous chunk if it exists
                if current_chunk.strip():
                    chunks.append(self._create_chunk(
                        current_chunk, document, current_article
                    ))
                
                current_article = part.strip()
                current_chunk = part + " "
            else:
                current_chunk += part
                
                # If chunk gets too large, split it
                if len(current_chunk) > config.CHUNK_SIZE:
                    chunks.append(self._create_chunk(
                        current_chunk, document, current_article
                    ))
                    current_chunk = ""
        
        # Add remaining content
        if current_chunk.strip():
            chunks.append(self._create_chunk(
                current_chunk, document, current_article
            ))
        
        return chunks
    
    def _fixed_size_chunking(self, content: str, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk document using fixed-size windows with overlap."""
        chunks = []
        start = 0
        
        while start < len(content):
            end = start + config.CHUNK_SIZE
            
            # Try to break at sentence boundary
            if end < len(content):
                # Look for sentence endings
                sentence_end = content.rfind('.', start, end)
                if sentence_end > start + config.CHUNK_SIZE // 2:
                    end = sentence_end + 1
            
            chunk_text = content[start:end].strip()
            if chunk_text:
                chunks.append(self._create_chunk(chunk_text, document))
            
            # Move start position with overlap
            start = end - config.CHUNK_OVERLAP
            if start >= len(content):
                break
        
        return chunks
    
    def _create_chunk(self, text: str, document: Dict[str, Any], 
                     article: str = None) -> Dict[str, Any]:
        """Create a chunk dictionary with metadata."""
        return {
            'text': text,
            'source_document': document['filename'],
            'source_filepath': document['filepath'],
            'article': article,
            'chunk_size': len(text),
            'metadata': {
                'file_type': document['file_type'],
                'original_size': document['size'],
                'chunk_index': len(text)  # This would be better tracked globally
            }
        }
    
    def save_processed_documents(self, chunks: List[Dict[str, Any]], 
                               output_dir: str = None) -> str:
        """Save processed chunks to files for later use."""
        if output_dir is None:
            output_dir = config.PROCESSED_DOCS_DIR
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save chunks as JSON for easy loading
        output_file = output_path / "processed_chunks.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(chunks)} chunks to {output_file}")
        return str(output_file)

def main():
    """Example usage of the DocumentProcessor."""
    processor = DocumentProcessor()
    
    # Load documents
    documents = processor.load_documents()
    
    if not documents:
        logger.warning("No documents found. Please add documents to the data/raw directory.")
        return
    
    # Process documents into chunks
    chunks = processor.chunk_documents(documents)
    
    # Save processed chunks
    output_file = processor.save_processed_documents(chunks)
    
    print(f"Document processing complete!")
    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    main()
