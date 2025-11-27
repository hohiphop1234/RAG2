"""
RAG Pipeline Module

This module implements the core RAG (Retrieval-Augmented Generation) pipeline.
It demonstrates how retrieval and generation work together to answer questions.
"""

from typing import List, Dict, Any

from config import config
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from src.llm_client import LLMFactory, LLMClient
from logging_config import get_logger

logger = get_logger(__name__)

class RAGPipeline:
    """
    Core RAG pipeline that combines retrieval and generation.
    
    This class demonstrates:
    1. How to retrieve relevant documents for a query
    2. How to format context for the LLM
    3. How to generate answers using retrieved context
    4. How to handle different types of queries
    """
    
    # Tunable constants
    MAX_CONTEXT_LENGTH = 4000  # Leave room for question and system prompt
    CHROMADB_DISTANCE_THRESHOLD = 200.0  # Lower distance = more similar

    def __init__(self, vector_store: VectorStore = None, 
                 embedding_generator: EmbeddingGenerator = None,
                 llm_client: LLMClient = None):
        """
        Initialize the RAG pipeline.
        
        Args:
            vector_store: Pre-initialized vector store
            embedding_generator: Pre-initialized embedding generator
            llm_client: Pre-initialized LLM client
        """
        self.vector_store = vector_store or VectorStore()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        
        # Initialize LLM client (OpenAI or Ollama)
        self.llm_client = llm_client or LLMFactory.create_client()
        self.llm_model = config.LLM_MODEL
        
        logger.info(f"Initialized RAG pipeline with LLM: {self.llm_model} ({config.LLM_PROVIDER})")
    
    def answer_question(self, question: str, 
                       include_sources: bool = True) -> Dict[str, Any]:
        """
        Answer a question using the RAG pipeline.
        
        This is the main method that demonstrates the complete RAG flow:
        1. Convert question to embedding
        2. Retrieve relevant documents
        3. Format context
        4. Generate answer using LLM
        
        Args:
            question: User's question
            include_sources: Whether to include source information
            
        Returns:
            Dictionary with answer and metadata
        """
        try:
            # Step 1: Generate embedding for the question
            logger.info(f"Processing question: {question[:100]}...")
            question_embedding = self.embedding_generator.get_embedding(question)
            
            # Step 2: Retrieve relevant documents
            relevant_docs = self.retrieve_relevant_documents(question_embedding)
            
            if not relevant_docs:
                return {
                    'answer': 'Xin lỗi, tôi không tìm thấy thông tin liên quan đến câu hỏi của bạn trong cơ sở dữ liệu.',
                    'sources': [],
                    'confidence': 0.0,
                    'num_sources': 0,
                    'error': 'No relevant documents found'
                }
            # Step 3: Format context for the LLM
            context = self._format_context(relevant_docs)
            
            # Step 4: Generate answer using LLM
            answer = self._generate_answer(question, context)
            
            # Prepare response
            response = {
                'answer': answer,
                'confidence': self._calculate_confidence(relevant_docs),
                'num_sources': len(relevant_docs)
            }
            
            if include_sources:
                response['sources'] = self._format_sources(relevant_docs)
            
            logger.info(f"Generated answer with {len(relevant_docs)} sources")
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {str(e)}")
            return {
                'answer': 'Xin lỗi, đã xảy ra lỗi khi xử lý câu hỏi của bạn.',
                'sources': [],
                'confidence': 0.0,
                'num_sources': 0,
                'error': str(e)
            }
    
    def retrieve_relevant_documents(self, query_embedding: List[float]) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.
        
        This method demonstrates the retrieval step of RAG:
        1. Vector similarity search
        2. Filtering by relevance threshold
        3. Ranking by similarity score
        
        Args:
            query_embedding: Embedding of the user's query
            
        Returns:
            List of relevant documents with metadata
        """
        # Perform similarity search
        search_results = self.vector_store.similarity_search(
            query_embedding,
            top_k=config.TOP_K_RESULTS
        )

        # Map scores to unified similarity and filter
        relevant_docs: List[Dict[str, Any]] = []
        for result in search_results:
            if self.vector_store.db_type == "chromadb":
                similarity_score = self._distance_to_similarity(
                    distance=result['score'],
                    threshold=self.CHROMADB_DISTANCE_THRESHOLD
                )
                if similarity_score is None:
                    continue
            else:
                # Non-Chroma path is not expected in Chroma-only build
                continue

            result['similarity_score'] = similarity_score
            relevant_docs.append(result)

        # Sort by similarity score (highest first)
        relevant_docs.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
        return relevant_docs
    
    def _format_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context for the LLM.
        
        This method demonstrates how to prepare context:
        1. Combine multiple document chunks
        2. Add source information
        3. Limit context length
        
        Args:
            relevant_docs: List of relevant documents
            
        Returns:
            Formatted context string
        """
        context_parts: List[str] = []

        for i, doc in enumerate(relevant_docs, 1):
            metadata = doc.get('metadata', {})
            source_document = metadata.get('source_document') or metadata.get('document') or 'unknown'
            source_info = f"[Nguồn {i}: {source_document}"
            article = metadata.get('article')
            if article:
                source_info += f", {article}"
            source_info += "]"

            context_parts.append(f"{source_info}\n{doc['text']}\n")

        context = "\n".join(context_parts)

        # Limit context length to avoid token limits
        if len(context) > self.MAX_CONTEXT_LENGTH:
            context = context[:self.MAX_CONTEXT_LENGTH] + "..."
            logger.warning("Context truncated due to length limit")

        return context
    
    def _generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using the LLM with retrieved context.
        
        This method demonstrates the generation step of RAG:
        1. Create a prompt with context and question
        2. Call the LLM API
        3. Extract and return the answer
        
        Args:
            question: User's question
            context: Retrieved context from documents
            
        Returns:
            Generated answer
        """
        # Create the prompt
        prompt = f"""Dựa trên thông tin được cung cấp dưới đây, hãy trả lời câu hỏi một cách chính xác và chi tiết.

Thông tin tham khảo:
{context}

Câu hỏi: {question}

Hướng dẫn:
- Trả lời bằng tiếng Việt
- Chỉ sử dụng thông tin từ tài liệu được cung cấp
- Nếu thông tin không đủ để trả lời, hãy nói rõ
- Trích dẫn các điều luật cụ thể khi có thể
- Trả lời ngắn gọn nhưng đầy đủ

Trả lời:"""

        try:
            # Use the unified LLM client interface
            messages = [
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            
            answer = self.llm_client.generate_response(
                messages=messages,
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE
            )
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "Xin lỗi, không thể tạo câu trả lời do lỗi kỹ thuật."
    
    def _calculate_confidence(self, relevant_docs: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on retrieved documents.
        
        Args:
            relevant_docs: List of relevant documents
            
        Returns:
            Confidence score between 0 and 1
        """
        if not relevant_docs:
            return 0.0
        
        # Calculate average similarity score
        avg_similarity = sum(doc['similarity_score'] for doc in relevant_docs) / len(relevant_docs)
        
        # Adjust confidence based on number of sources
        num_sources_factor = min(len(relevant_docs) / 3, 1.0)  # More sources = higher confidence
        
        confidence = (avg_similarity * 0.7) + (num_sources_factor * 0.3)
        return min(confidence, 1.0)
    
    def _format_sources(self, relevant_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format source information for display.
        
        Args:
            relevant_docs: List of relevant documents
            
        Returns:
            List of formatted source information
        """
        sources = []
        for doc in relevant_docs:
            source = {
                'document': doc['metadata']['source_document'],
                'similarity_score': doc['similarity_score'],
                'text_preview': doc['text'][:200] + "..." if len(doc['text']) > 200 else doc['text']
            }
            
            if 'article' in doc['metadata']:
                source['article'] = doc['metadata']['article']
            # Include chunk_index when available for easier traceability
            if 'chunk_index' in doc['metadata']:
                source['chunk_index'] = doc['metadata']['chunk_index']
            
            sources.append(source)
        
        return sources

    def _distance_to_similarity(self, distance: float, threshold: float) -> Any:
        """
        Convert a distance score (lower is better) to a 0-1 similarity.
        Returns None if distance exceeds threshold.
        """
        if distance > threshold:
            return None
        return max(0.0, 1.0 - (distance / threshold))
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG pipeline."""
        vector_stats = self.vector_store.get_stats()
        llm_info = self.llm_client.get_model_info()
        
        return {
            'llm_info': llm_info,
            'embedding_model': self.embedding_generator.model_type,
            'vector_database': vector_stats,
            'similarity_threshold': config.SIMILARITY_THRESHOLD,
            'top_k_results': config.TOP_K_RESULTS
        }

def main():
    """Example usage of the RAG pipeline."""
    try:
        # Initialize RAG pipeline
        rag = RAGPipeline()
        
        # Get pipeline statistics
        stats = rag.get_pipeline_stats()
        print("RAG Pipeline Statistics:")
        print(f"LLM Provider: {stats['llm_info']['provider']}")
        print(f"LLM Model: {stats['llm_info']['model']}")
        print(f"LLM Type: {stats['llm_info']['type']}")
        print(f"Embedding Model: {stats['embedding_model']}")
        print(f"Vector Database: {stats['vector_database']['database_type']}")
        print(f"Document Count: {stats['vector_database']['document_count']}")
        
        # Test with sample questions
        sample_questions = [
            "Luật giao thông đường bộ quy định gì về tốc độ xe máy?",
            "Điều kiện để được cấp giấy phép lái xe là gì?",
            "Quy định về xử phạt vi phạm giao thông như thế nào?",
            "Luật đất đai quy định gì về quyền sử dụng đất?"
        ]
        
        print("\n" + "="*50)
        print("Testing RAG Pipeline with Sample Questions")
        print("="*50)
        
        for i, question in enumerate(sample_questions, 1):
            print(f"\n{i}. Question: {question}")
            print("-" * 50)
            
            response = rag.answer_question(question)
            
            print(f"Answer: {response['answer']}")
            print(f"Confidence: {response['confidence']:.2f}")
            print(f"Sources: {response['num_sources']}")
            
            if response.get('sources'):
                print("Source documents:")
                for j, source in enumerate(response['sources'][:2], 1):  # Show top 2 sources
                    print(f"  {j}. {source['document']} (score: {source['similarity_score']:.3f})")
            
            print()
    
    except Exception as e:
        logger.error(f"Error in RAG pipeline test: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
