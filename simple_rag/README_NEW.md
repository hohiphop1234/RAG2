# 🇻🇳 Simple RAG System - Vietnamese Law Chatbot

A complete Retrieval-Augmented Generation (RAG) system for Vietnamese legal documents, supporting both local (Ollama) and cloud (OpenAI) operation.

## 📖 Overview

This RAG system allows you to:
- 📄 Upload Vietnamese legal documents (PDF, DOCX, TXT)
- 🔍 Ask questions in natural Vietnamese language
- 💬 Get accurate answers with source citations
- 🔒 Run completely offline (with Ollama) or use OpenAI

## ✨ Key Features

- **Flexible Deployment**: Choose between local (Ollama) or cloud (OpenAI) models
- **Smart Document Processing**: Automatic chunking with semantic boundaries
- **Multi-format Support**: PDF, DOCX, TXT, HTML documents
- **Vector Search**: ChromaDB or FAISS for fast retrieval
- **Web Interface**: User-friendly Streamlit UI
- **Source Tracking**: All answers include document references
- **Vietnamese Optimized**: Specialized for Vietnamese legal text

## 🏗️ Architecture

```
User Question
    ↓
[Document Processing] → Text chunking with overlap
    ↓
[Embedding] → Convert text to vectors (Ollama/OpenAI)
    ↓
[Vector Store] → Store in ChromaDB/FAISS
    ↓
[Retrieval] → Find relevant document chunks
    ↓
[Generation] → Generate answer with LLM (Ollama/OpenAI)
    ↓
Answer + Sources
```

## 📁 Project Structure

```
simple_rag/
├── config.py                 # Centralized configuration
├── logging_config.py         # Logging setup
├── main.py                   # Main entry point
├── requirements.txt          # Python dependencies
├── run_web.bat              # Windows launcher
├── run_web.ps1              # PowerShell launcher
├── data/
│   ├── raw/                 # Put your documents here
│   └── processed/           # Auto-generated processed docs
├── models/
│   └── chromadb/            # Vector database storage
├── src/
│   ├── document_processor.py  # Load & chunk documents
│   ├── embeddings.py          # Generate embeddings
│   ├── vector_store.py        # Vector database management
│   ├── llm_client.py          # LLM provider abstraction
│   ├── rag_pipeline.py        # Core RAG logic
│   └── web_interface.py       # Streamlit UI
└── temp_uploads/            # Temporary file storage
```

## 🚀 Quick Start

### Option 1: Local Setup (Recommended)

**Advantages**: Free, private, no API costs

1. **Install Ollama** from [ollama.ai](https://ollama.ai/)

2. **Pull required models**:
   ```bash
   ollama pull deepseek-r1           # Language model
   ollama pull mxbai-embed-large     # Embedding model
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the web interface**:
   ```bash
   streamlit run src/web_interface.py
   ```

5. **Open browser**: http://localhost:8501

### Option 2: OpenAI Setup

**Advantages**: Faster, more powerful models

1. **Get OpenAI API key** from [platform.openai.com](https://platform.openai.com)

2. **Create `.env` file**:
   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```

3. **Update `config.py`**:
   ```python
   LLM_PROVIDER = "openai"
   EMBEDDING_PROVIDER = "openai"
   LLM_MODEL = "gpt-4"
   EMBEDDING_MODEL = "text-embedding-3-small"
   ```

4. **Install and run**:
   ```bash
   pip install -r requirements.txt
   streamlit run src/web_interface.py
   ```

## 📚 Usage

### Web Interface

1. **Upload Documents**:
   - Click "📁 Quản lý tài liệu" in sidebar
   - Select PDF/DOCX/TXT files
   - Click "🔄 Xử lý tài liệu"

2. **Ask Questions**:
   - Type your question in Vietnamese
   - Click "📤 Gửi"
   - View answer with confidence score and sources

3. **Try Examples**:
   - Click any example question button
   - See how the system works

### Command Line

```bash
# Setup the RAG system (process documents)
python main.py setup

# Run web interface
python main.py web
```

### Windows Users

Double-click `run_web.bat` or run `run_web.ps1` in PowerShell.

## ⚙️ Configuration

Edit `config.py` to customize behavior:

### Document Processing
```python
CHUNK_SIZE = 1000              # Text chunk size (characters)
CHUNK_OVERLAP = 200            # Overlap between chunks
MAX_CHUNKS_PER_DOCUMENT = 50   # Limit per document
```

### Embedding Settings
```python
EMBEDDING_PROVIDER = "ollama"          # "ollama" or "openai"
EMBEDDING_MODEL = "mxbai-embed-large"  # Model name
EMBEDDING_DIMENSION = 1024             # Vector dimension
```

### Vector Database
```python
VECTOR_DB_TYPE = "chromadb"       # "chromadb" or "faiss"
SIMILARITY_THRESHOLD = 0.3        # Min similarity (0-1)
TOP_K_RESULTS = 5                 # Results to retrieve
```

### LLM Settings
```python
LLM_PROVIDER = "ollama"           # "ollama" or "openai"
LLM_MODEL = "deepseek-r1"         # Model name
MAX_TOKENS = 1000                 # Response length
TEMPERATURE = 0.7                 # Creativity (0-1)
```

## 🎯 Example Questions

- "Luật giao thông đường bộ quy định gì về tốc độ xe máy?"
- "Điều kiện để được cấp giấy phép lái xe là gì?"
- "Quy định về xử phạt vi phạm giao thông như thế nào?"
- "Luật đất đai quy định gì về quyền sử dụng đất?"

## 🔧 System Requirements

### Minimum (Ollama)
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space
- **CPU**: 64-bit Intel/AMD
- **Python**: 3.8 - 3.13

### Minimum (OpenAI)
- **RAM**: 4GB
- **Storage**: 2GB free space
- **Python**: 3.8 - 3.13
- **Internet**: Required for API calls

## 📊 Performance

### Local (Ollama)
- Initial model load: ~10-15 seconds
- Document embedding: ~1-2 seconds per document
- Query response: ~3-5 seconds

### OpenAI
- Document embedding: ~0.5-1 second per document
- Query response: ~1-2 seconds

## 🐛 Troubleshooting

### Ollama Issues

**"Connection refused"**
```bash
# Check if Ollama is running
ollama list

# Restart Ollama service
```

**"Model not found"**
```bash
# Re-pull the models
ollama pull deepseek-r1
ollama pull mxbai-embed-large
```

### OpenAI Issues

**"Invalid API key"**
- Check your `.env` file
- Verify key at platform.openai.com
- Restart the application

### Memory Issues

**"Out of memory"**
- Reduce `CHUNK_SIZE` in config.py
- Use fewer documents
- Switch to smaller model (Ollama: `llama3.2`)

### Unicode Issues

**Text encoding errors**
- Use Python 3.8-3.13 (avoid 3.14 alpha)
- Ensure files are UTF-8 encoded

## 🧪 Testing

Run tests to verify installation:

```bash
# Test complete system
python test_complete_local_rag.py

# Test embeddings only
python test_local_embeddings.py
```

Expected output: `PASS ✅`

## 📦 Dependencies

Key libraries:
- `streamlit` - Web interface
- `chromadb` - Vector database
- `ollama` - Local LLM client
- `openai` - OpenAI API client
- `sentence-transformers` - Alternative embeddings
- `PyPDF2` - PDF processing
- `python-docx` - DOCX processing
- `beautifulsoup4` - HTML processing

See `requirements.txt` for complete list.

## 🔐 Privacy & Security

### Local Mode (Ollama)
- ✅ All data stays on your machine
- ✅ No internet required after setup
- ✅ No API costs
- ✅ Complete privacy

### OpenAI Mode
- ⚠️ Data sent to OpenAI servers
- ⚠️ Requires internet connection
- ⚠️ API costs apply
- ℹ️ Subject to OpenAI's privacy policy

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Add more document formats
- Implement re-ranking
- Add query rewriting
- Support more languages
- Add conversation memory
- Implement caching

## 📝 License

[Specify your license here]

## 🆘 Support

For issues and questions:
1. Check the Troubleshooting section
2. Review SETUP.md for detailed installation
3. Check logs in console for error details
4. Open an issue on GitHub

## 🙏 Acknowledgments

- Ollama for local LLM runtime
- ChromaDB for vector storage
- Streamlit for web interface
- OpenAI for API access

---

**Made with ❤️ for Vietnamese legal document understanding**
