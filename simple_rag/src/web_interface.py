

import streamlit as st
from typing import Dict, List
import time
from pathlib import Path
import sys
import os
import tempfile
import shutil

# Add parent directory to path so we can import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our RAG components
from config import config
from src.rag_pipeline import RAGPipeline
from src.document_processor import DocumentProcessor
from src.embeddings import EmbeddingGenerator
from src.vector_store import VectorStore
from logging_config import get_logger

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title=config.WEB_TITLE,
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        color: #333333;
        font-size: 1rem;
        line-height: 1.5;
    }
    .user-message {
        background-color: #e8f4fd;
        border-left: 4px solid #2196f3;
        color: #1565c0;
        font-weight: 500;
    }
    .bot-message {
        background-color: #f8f5ff;
        border-left: 4px solid #9c27b0;
        color: #4a148c;
        font-weight: 400;
    }
    .source-info {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
        font-style: italic;
    }
    .confidence-bar {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin-top: 0.5rem;
    }
    .confidence-fill {
        height: 100%;
        background-color: #4caf50;
        transition: width 0.3s ease;
    }
    .confidence-text {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.25rem;
    }
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .chat-message {
            color: #ffffff;
        }
        .user-message {
            background-color: #1e3a8a;
            color: #dbeafe;
        }
        .bot-message {
            background-color: #581c87;
            color: #e9d5ff;
        }
        .source-info {
            color: #9ca3af;
        }
        .confidence-text {
            color: #9ca3af;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_pipeline():
    """Initialize the RAG pipeline with caching."""
    try:
        # Check if vector store exists
        vector_store = VectorStore()
        stats = vector_store.get_stats()
        
        if stats['document_count'] == 0:
            st.warning("No documents found in vector store. Please upload documents first.")
            return None
        
        # Initialize RAG pipeline
        rag = RAGPipeline(vector_store=vector_store)
        return rag
    except Exception as e:
        st.error(f"Error initializing RAG pipeline: {str(e)}")
        return None


def display_chat_message(message: str, is_user: bool = True, 
                        sources: List[Dict] = None, confidence: float = None):
    """Display a chat message with proper styling."""
    if is_user:
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>Bạn:</strong> {message}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>Trợ lý pháp lý:</strong> {message}
        </div>
        """, unsafe_allow_html=True)
        
        # Display confidence score
        if confidence is not None:
            confidence_percent = int(confidence * 100)
            st.markdown(f"""
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence_percent}%"></div>
            </div>
            <p class="confidence-text">Độ tin cậy: {confidence_percent}%</p>
            """, unsafe_allow_html=True)
        
        # Display sources
        if sources:
            with st.expander("📚 Nguồn tham khảo"):
                for i, source in enumerate(sources, 1):
                    chunk_info = f" (Đoạn #{source['chunk_index']})" if 'chunk_index' in source else ""
                    st.markdown(f"""
                    **{i}. {source['document']}{chunk_info}**
                    - Điểm tương đồng: {source['similarity_score']:.3f}
                    - Nội dung: {source['text_preview']}
                    """)

def show_document_list():
    """Display list of documents currently in the vector store."""
    try:
        vector_store = VectorStore()
        
        # Check if ChromaDB is being used
        if hasattr(vector_store, 'collection') and vector_store.collection:
            # Get all documents from ChromaDB
            results = vector_store.collection.get()
            
            if results and results['ids']:
                total_docs = len(results['ids'])
                st.success(f"📊 Tổng số đoạn văn: {total_docs}")
                
                # Group by document name from metadata
                doc_groups = {}
                for i, (doc_id, metadata, content) in enumerate(zip(
                    results['ids'], 
                    results['metadatas'] if results['metadatas'] else [{}] * len(results['ids']),
                    results['documents'] if results['documents'] else [''] * len(results['ids'])
                )):
                    # Prefer standardized key used by vector store
                    doc_name = (metadata.get('source_document') or
                                metadata.get('document') or
                                'Unknown Document') if metadata else 'Unknown Document'
                    if doc_name not in doc_groups:
                        doc_groups[doc_name] = []
                    doc_groups[doc_name].append({
                        'id': doc_id,
                        'content': content[:200] + "..." if len(content) > 200 else content,
                        'metadata': metadata
                    })
                
                st.write("### 📚 Danh sách tài liệu:")
                
                for doc_name, chunks in doc_groups.items():
                    with st.expander(f"📄 {doc_name} ({len(chunks)} đoạn)"):
                        for j, chunk in enumerate(chunks[:5]):  # Show first 5 chunks
                            st.write(f"**Đoạn {j+1}:**")
                            st.write(f"- ID: `{chunk['id']}`")
                            st.write(f"- Nội dung: {chunk['content']}")
                            if chunk['metadata']:
                                st.write(f"- Metadata: {chunk['metadata']}")
                            st.write("---")
                        
                        if len(chunks) > 5:
                            st.write(f"... và {len(chunks) - 5} đoạn khác")
                            
            else:
                st.warning("🔍 Chưa có tài liệu nào trong cơ sở dữ liệu.")
                st.info("💡 Hãy tải lên tài liệu bằng chức năng 'Quản lý tài liệu' ở trên.")
        else:
            st.error("❌ Không thể kết nối đến cơ sở dữ liệu vector.")
            
    except Exception as e:
        st.error(f"❌ Lỗi khi lấy danh sách tài liệu: {str(e)}")
        st.write("Chi tiết lỗi:", str(e))

def show_raw_stored_data():
    """Display the raw data structure of what's actually stored in the vector database."""
    st.write("# 🔬 Dữ liệu được lưu trữ")
    
    try:
        vector_store = VectorStore()
        
        if hasattr(vector_store, 'collection') and vector_store.collection:
            # Get all stored data with full details
            results = vector_store.collection.get(
                include=['metadatas', 'documents', 'embeddings']
            )
            
            if results and results['ids']:
                total_items = len(results['ids'])
                st.success(f"📊 **Tổng số mục được lưu:** {total_items}")
                
                # Show data structure overview
                st.write("### 📋 Cấu trúc dữ liệu:")
                st.write(f"- **IDs**: {len(results['ids'])} mục")
                st.write(f"- **Documents**: {len(results['documents']) if results.get('documents') is not None else 0} văn bản")
                st.write(f"- **Metadatas**: {len(results['metadatas']) if results.get('metadatas') is not None else 0} metadata")
                
                # Safe embedding check
                embeddings_count = 0
                embedding_dim = 0
                if results.get('embeddings') is not None and len(results['embeddings']) > 0:
                    embeddings_count = len(results['embeddings'])
                    if len(results['embeddings']) > 0 and results['embeddings'][0] is not None:
                        embedding_dim = len(results['embeddings'][0])
                
                st.write(f"- **Embeddings**: {embeddings_count} vector ({embedding_dim} chiều)")
                
                # Show sample entries
                st.write("### 📝 Mẫu dữ liệu được lưu:")
                
                for i in range(min(3, total_items)):  # Show first 3 items
                    with st.expander(f"📄 Mục {i+1}: {results['ids'][i]}"):
                        
                        # Document content
                        st.write("**📄 Nội dung văn bản:**")
                        doc_content = results['documents'][i] if results['documents'] else 'N/A'
                        st.text_area(
                            f"Content {i+1}", 
                            value=doc_content[:500] + ("..." if len(doc_content) > 500 else ""),
                            height=100,
                            disabled=True
                        )
                        
                        # Metadata
                        st.write("**🏷️ Metadata:**")
                        metadata = results['metadatas'][i] if results.get('metadatas') is not None and len(results['metadatas']) > i else {}
                        if metadata:
                            for key, value in metadata.items():
                                st.write(f"- `{key}`: {value}")
                        else:
                            st.write("*Không có metadata*")
                        
                        # Embedding info
                        st.write("**🔢 Vector embedding:**")
                        if (results.get('embeddings') is not None and 
                            len(results['embeddings']) > i and 
                            results['embeddings'][i] is not None):
                            embedding = results['embeddings'][i]
                            st.write(f"- Số chiều: {len(embedding)}")
                            st.write(f"- Giá trị đầu: {embedding[:5]}...")
                            st.write(f"- Giá trị cuối: {embedding[-5:]}...")
                            # Safe magnitude calculation
                            try:
                                magnitude = sum(float(x)*float(x) for x in embedding)**0.5
                                st.write(f"- Magnitude: {magnitude:.3f}")
                            except (TypeError, ValueError):
                                st.write("- Magnitude: Không tính được")
                        else:
                            st.write("*Không có embedding*")
                
                # Raw data viewer
                with st.expander("🔍 Xem dữ liệu thô (Raw JSON)"):
                    # Show first few items in JSON format
                    # Safe embedding info
                    embeddings_info = "No embeddings"
                    if results.get('embeddings') is not None and len(results['embeddings']) > 0:
                        embed_count = len(results['embeddings'])
                        embed_dim = 0
                        if results['embeddings'][0] is not None:
                            embed_dim = len(results['embeddings'][0])
                        embeddings_info = f"{embed_count} vectors of {embed_dim} dimensions"
                    
                    sample_data = {
                        'ids': results['ids'][:3],
                        'documents': results['documents'][:3] if results.get('documents') is not None else [],
                        'metadatas': results['metadatas'][:3] if results.get('metadatas') is not None else [],
                        'embeddings_info': embeddings_info
                    }
                    st.json(sample_data)
                
                # Search functionality
                st.write("### 🔍 Tìm kiếm trong dữ liệu đã lưu:")
                search_term = st.text_input("Nhập từ khóa để tìm kiếm trong nội dung:")
                if search_term:
                    matches = []
                    documents = results.get('documents', [])
                    metadatas = results.get('metadatas', [])
                    
                    for i, doc in enumerate(documents if documents is not None else []):
                        if doc and search_term.lower() in doc.lower():
                            metadata = metadatas[i] if metadatas and len(metadatas) > i else {}
                            matches.append({
                                'index': i,
                                'id': results['ids'][i],
                                'content': doc,
                                'metadata': metadata
                            })
                    
                    if matches:
                        st.success(f"✅ Tìm thấy {len(matches)} kết quả:")
                        for match in matches[:5]:  # Show first 5 matches
                            st.write(f"**ID:** `{match['id']}`")
                            # Highlight search term
                            highlighted = match['content'].replace(
                                search_term, 
                                f"**{search_term}**"
                            )
                            st.write(f"**Nội dung:** {highlighted[:300]}...")
                            st.write("---")
                    else:
                        st.warning(f"🔍 Không tìm thấy '{search_term}' trong dữ liệu")
                        
            else:
                st.warning("📭 Không có dữ liệu trong vector database")
                
        else:
            st.error("❌ Không thể kết nối đến vector database")
            
    except Exception as e:
        st.error(f"❌ Lỗi khi tải dữ liệu: {str(e)}")
        import traceback
        with st.expander("🔍 Chi tiết lỗi"):
            st.code(traceback.format_exc())

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown(f"""
    <div class="main-header">
        🇻🇳 {config.WEB_TITLE}
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"**{config.WEB_DESCRIPTION}**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Cài đặt")
        
        # Initialize components
        rag = initialize_rag_pipeline()
        
        # System status
        st.subheader("📊 Trạng thái hệ thống")
        if rag:
            stats = rag.get_pipeline_stats()
            st.success("✅ RAG Pipeline: Sẵn sàng")
            st.info(f"📄 Số tài liệu: {stats['vector_database']['document_count']}")
            st.info(f"🤖 LLM: {stats['llm_info']['model']} ({stats['llm_info']['provider']})")
            st.info(f"🔤 Embedding: {stats['embedding_model']}")
        else:
            st.error("❌ RAG Pipeline: Chưa sẵn sàng")
        
        
        # Document upload
        st.subheader("📁 Quản lý tài liệu")
        uploaded_files = st.file_uploader(
            "Tải lên tài liệu pháp luật",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True,
            help="Hỗ trợ định dạng: PDF, DOCX, TXT"
        )
        
        if uploaded_files:
            if st.button("🔄 Xử lý tài liệu"):
                with st.spinner("Đang xử lý tài liệu..."):
                    success = process_documents(uploaded_files)
                    if success:
                        st.session_state.last_doc_update = time.time()
                        st.rerun()  # Only refresh if processing was successful
        
        # View current documents
        st.subheader("📄 Tài liệu hiện có")
        
        # Refresh buttons
        col1, col2, col3 = st.columns([2, 1, 1])
        with col2:
            if st.button("🔄 Refresh", help="Cập nhật danh sách tài liệu"):
                st.rerun()
        with col3:
            if st.button("🗑️ Clear Cache", help="Xóa cache và tải lại"):
                # Clear any cached data
                if hasattr(st.session_state, 'last_doc_update'):
                    del st.session_state.last_doc_update
                st.rerun()
        
        # Show quick summary
        try:
            vector_store = VectorStore()
            if hasattr(vector_store, 'collection') and vector_store.collection:
                # Force refresh ChromaDB collection - get all data
                try:
                    results = vector_store.collection.get(include=['metadatas', 'documents'])
                except Exception as e:
                    st.warning(f"⚠️ Lỗi khi tải dữ liệu: {str(e)}")
                    # Try alternative method
                    results = vector_store.collection.get()
                
                if results and results['ids']:
                    # Group by document name from metadata
                    doc_groups = {}
                    unique_docs = set()
                    
                    for i, metadata in enumerate(results['metadatas'] if results['metadatas'] else []):
                        if metadata:
                            # Prefer standardized key used by vector store
                            doc_name = (metadata.get('source_document') or 
                                      metadata.get('document') or 
                                      metadata.get('filename') or 
                                      metadata.get('source_filepath', '').split('/')[-1] or
                                      f"Document_{i}")
                        else:
                            doc_name = f"Unknown_Document_{i}"
                        
                        unique_docs.add(doc_name)
                        doc_groups[doc_name] = doc_groups.get(doc_name, 0) + 1
                    
                    total_chunks = len(results['ids'])
                    total_docs = len(unique_docs)
                    
                    st.success(f"📊 **Tổng cộng:** {total_docs} tài liệu, {total_chunks} đoạn văn")
                    
                    # Show debug info
                    with st.expander("🔍 Thông tin debug"):
                        st.write(f"**Metadata keys found:** {list(set([k for m in (results['metadatas'] or []) for k in (m.keys() if m else [])]))}")
                        st.write(f"**Sample metadata:** {results['metadatas'][0] if results['metadatas'] else 'None'}")
                    
                    st.write("**Chi tiết:**")
                    for doc_name, count in sorted(doc_groups.items()):
                        st.write(f"• 📄 {doc_name}: {count} đoạn")
                else:
                    st.info("📭 Chưa có tài liệu nào trong cơ sở dữ liệu")
                    st.write("💡 Hãy tải lên tài liệu bằng chức năng 'Quản lý tài liệu' ở trên")
        except Exception as e:
            st.error(f"❌ Không thể tải thông tin: {str(e)}")
            st.write("💡 Hãy thử nhấn nút 'Refresh' hoặc khởi động lại ứng dụng")
            # Show more debug info
            with st.expander("🔍 Debug Error"):
                st.write(f"Error details: {str(e)}")
                st.write(f"Vector store type: {type(vector_store)}")
                st.write(f"Has collection: {hasattr(vector_store, 'collection')}")
                if hasattr(vector_store, 'collection'):
                    st.write(f"Collection type: {type(vector_store.collection)}")
        
        if st.button("👁️ Xem chi tiết tài liệu"):
            show_document_list()
        
        if st.button("🔬 Xem dữ liệu được lưu trữ"):
            show_raw_stored_data()
        
        # Clear chat
        if st.button("🗑️ Xóa lịch sử chat"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface
    st.subheader("💬 Trò chuyện")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'last_doc_update' not in st.session_state:
        st.session_state.last_doc_update = 0
    
    # Display chat history
    for message in st.session_state.chat_history:
        display_chat_message(
            message['content'], 
            message['is_user'],
            message.get('sources'),
            message.get('confidence')
        )
    
    # Chat input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "Nhập câu hỏi của bạn:",
            placeholder="Ví dụ: Luật giao thông quy định gì về tốc độ xe máy?"
        )
    
    with col2:
        send_button = st.button("📤 Gửi", type="primary")
    
    # Process user input
    if send_button and user_input and rag:
        # Add user message to history
        st.session_state.chat_history.append({
            'content': user_input,
            'is_user': True
        })
        
        # Get response from RAG pipeline
        with st.spinner("Đang tìm kiếm thông tin..."):
            response = rag.answer_question(user_input)
        
        # Add bot response to history
        st.session_state.chat_history.append({
            'content': response['answer'],
            'is_user': False,
            'sources': response.get('sources', []),
            'confidence': response.get('confidence', 0.0)
        })
        
        
        st.rerun()
    
    # Example questions
    st.subheader("💡 Câu hỏi mẫu")
    example_questions = [
        "Luật giao thông đường bộ quy định gì về tốc độ xe máy?",
        "Điều kiện để được cấp giấy phép lái xe là gì?",
        "Quy định về xử phạt vi phạm giao thông như thế nào?",
        "Luật đất đai quy định gì về quyền sử dụng đất?"
    ]
    
    cols = st.columns(2)
    for i, question in enumerate(example_questions):
        with cols[i % 2]:
            if st.button(f"❓ {question[:50]}...", key=f"example_{i}"):
                st.session_state.example_question = question
                st.rerun()
    
    # Handle example question
    if 'example_question' in st.session_state:
        example_q = st.session_state.example_question
        del st.session_state.example_question
        
        # Add to chat history
        st.session_state.chat_history.append({
            'content': example_q,
            'is_user': True
        })
        
        # Get response
        if rag:
            with st.spinner("Đang tìm kiếm thông tin..."):
                response = rag.answer_question(example_q)
            
            st.session_state.chat_history.append({
                'content': response['answer'],
                'is_user': False,
                'sources': response.get('sources', []),
                'confidence': response.get('confidence', 0.0)
            })
        
        st.rerun()

def process_documents(uploaded_files):
    """Process uploaded documents and add them to the vector store."""
    try:
        # Create temporary directory for uploaded files using system temp
        temp_dir = Path(tempfile.mkdtemp(prefix="rag_uploads_"))
        # temp_dir.mkdir(exist_ok=True)  # Not needed as mkdtemp already creates it
        
        # Save uploaded files
        saved_files = []
        for uploaded_file in uploaded_files:
            file_path = temp_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_files.append(file_path)
        
        # Process documents
        processor = DocumentProcessor()
        documents = []
        
        for file_path in saved_files:
            doc = processor._load_single_document(file_path)
            if doc:
                documents.append(doc)
        
        if not documents:
            st.error("Không thể xử lý tài liệu. Vui lòng kiểm tra định dạng file.")
            return
        
        # Chunk documents
        chunks = processor.chunk_documents(documents)
        
        # Generate embeddings
        embedding_gen = EmbeddingGenerator()
        embedded_chunks = embedding_gen.embed_documents(chunks)
        
        # Add to vector store
        vector_store = VectorStore()
        success = vector_store.add_documents(embedded_chunks)
        
        if success:
            st.success(f"✅ Đã xử lý và thêm {len(embedded_chunks)} đoạn văn bản vào cơ sở dữ liệu!")
            st.info("Bạn có thể bắt đầu đặt câu hỏi ngay bây giờ.")
            return True  # Return success
        else:
            st.error("❌ Có lỗi xảy ra khi thêm tài liệu vào cơ sở dữ liệu.")
            return False  # Return failure
        
    except Exception as e:
        st.error(f"Lỗi xử lý tài liệu: {str(e)}")
        return False  # Return failure
    
    finally:
        # Clean up temporary files and directory
        if 'temp_dir' in locals() and temp_dir.exists():
            try:
                shutil.rmtree(temp_dir)
            except Exception as cleanup_error:
                # Don't fail the whole operation if cleanup fails
                logger.warning(f"Failed to cleanup temp directory: {cleanup_error}")

if __name__ == "__main__":
    main()
