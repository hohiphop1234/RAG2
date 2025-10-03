# Hướng dẫn Cài đặt RAG System

## 📋 Tổng quan

Hệ thống RAG (Retrieval-Augmented Generation) hoàn toàn local cho tài liệu pháp luật Việt Nam. Hệ thống này cho phép bạn đặt câu hỏi về luật giao thông và đất đai mà không cần kết nối internet sau khi cài đặt.

## 🎯 Tính năng chính

- ✅ **Xử lý hoàn toàn local**: Không cần API key, không có chi phí
- ✅ **Bảo mật dữ liệu**: Tất cả dữ liệu được xử lý trên máy tính của bạn
- ✅ **Hỗ trợ tiếng Việt**: Tối ưu cho tài liệu pháp luật Việt Nam
- ✅ **Giao diện web**: Dễ sử dụng với Streamlit
- ✅ **Đa định dạng**: Hỗ trợ .txt, .pdf, .docx

## 🔧 Yêu cầu hệ thống

### Phần cứng tối thiểu
- **RAM**: 8GB (khuyến nghị 16GB)
- **Ổ cứng**: 10GB trống
- **CPU**: Intel/AMD 64-bit

### Phần mềm
- **Python**: 3.8 - 3.13 (tránh 3.14 alpha)
- **Ollama**: Để chạy mô hình AI local
- **Git**: Để clone repository

## 📥 Cài đặt từng bước

### Bước 1: Cài đặt Ollama

1. **Tải Ollama**:
   - Truy cập [ollama.ai](https://ollama.ai/)
   - Tải phiên bản phù hợp với hệ điều hành của bạn
   - Cài đặt theo hướng dẫn

2. **Tải các mô hình cần thiết**:
   ```bash
   # Mô hình ngôn ngữ (8.2B tham số, tối ưu cho tiếng Việt)
   ollama pull deepseek-r1
   
   # Mô hình embedding (334M tham số, 1024 chiều)
   ollama pull mxbai-embed-large
   ```

3. **Kiểm tra Ollama hoạt động**:
   ```bash
   ollama list
   ```
   Bạn sẽ thấy 2 mô hình đã tải.

### Bước 2: Cài đặt Python Environment

1. **Clone repository**:
   ```bash
   git clone <repository-url>
   cd simple_rag
   ```

2. **Tạo virtual environment**:
   ```bash
   # Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Cài đặt dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Bước 3: Cấu hình hệ thống

1. **Kiểm tra cấu hình** (đã được thiết lập sẵn):
   ```python
   # config.py - Cấu hình mặc định
   LLM_PROVIDER = "ollama"                    # Sử dụng Ollama local
   EMBEDDING_PROVIDER = "ollama"              # Embedding local
   LLM_MODEL = "deepseek-r1"                  # Mô hình tiếng Việt
   EMBEDDING_MODEL = "mxbai-embed-large"      # Mô hình embedding chất lượng cao
   ```

2. **Tạo file .env** (tùy chọn):
   ```bash
   # Copy từ file mẫu
   cp env_example.txt .env
   # Chỉnh sửa nếu cần thiết
   ```

### Bước 4: Kiểm tra cài đặt

1. **Test hệ thống hoàn chỉnh**:
   ```bash
   python test_complete_local_rag.py
   ```
   Kết quả mong đợi: `Final result: PASS ✅`

2. **Test embedding**:
   ```bash
   python test_local_embeddings.py
   ```
   Kết quả mong đợi: `Local embedding test PASSED ✅`

## 🚀 Sử dụng hệ thống

### 1. Thêm tài liệu

Đặt các file tài liệu pháp luật vào thư mục `data/raw/`:
```bash
# Ví dụ
cp luat_giao_thong.pdf data/raw/
cp luat_dat_dai.docx data/raw/
```

**Định dạng hỗ trợ**:
- `.txt` - Văn bản thuần
- `.pdf` - Tài liệu PDF
- `.docx` - Tài liệu Word

### 2. Chạy giao diện web

```bash
# Từ thư mục gốc của project
streamlit run src/web_interface.py
```

Truy cập: http://localhost:8501

### 3. Sử dụng từ command line

```bash
# Setup hệ thống
python main.py setup

# Test hệ thống
python main.py test

# Chạy web interface
python main.py web
```

## ⚙️ Cấu hình nâng cao

### Điều chỉnh hiệu suất

```python
# config.py
CHUNK_SIZE = 1000              # Kích thước đoạn văn bản
SIMILARITY_THRESHOLD = 0.3     # Ngưỡng tương đồng (thấp hơn = tìm nhiều hơn)
TOP_K_RESULTS = 5              # Số lượng kết quả trả về
TEMPERATURE = 0.7              # Độ sáng tạo của AI (0.0-1.0)
```

### Sử dụng OpenAI (tùy chọn)

Nếu muốn sử dụng OpenAI thay vì local:

1. **Thêm API key**:
   ```bash
   # Trong file .env
   OPENAI_API_KEY=your_key_here
   ```

2. **Cập nhật config**:
   ```python
   # config.py
   LLM_PROVIDER = "openai"
   EMBEDDING_PROVIDER = "openai"
   ```

## 🔍 Troubleshooting

### Lỗi thường gặp

1. **"Connection refused"**:
   - Kiểm tra Ollama đang chạy: `ollama list`
   - Khởi động lại Ollama nếu cần

2. **"Model not found"**:
   - Tải lại mô hình: `ollama pull deepseek-r1`
   - Kiểm tra tên mô hình trong config.py

3. **"Out of memory"**:
   - Đóng các ứng dụng khác
   - Giảm CHUNK_SIZE trong config.py
   - Sử dụng mô hình nhỏ hơn

4. **Unicode errors**:
   - Sử dụng Python 3.13 hoặc thấp hơn
   - Tránh Python 3.14 alpha

### Kiểm tra hệ thống

```bash
# Kiểm tra Ollama
curl http://localhost:11434/api/tags

# Kiểm tra Python packages
pip list | grep -E "(openai|chromadb|streamlit)"

# Kiểm tra dung lượng
ollama ps
```

## 📊 Hiệu suất hệ thống

### Thời gian xử lý
- **Embedding**: ~1-2 giây/tài liệu
- **Truy vấn**: ~3-5 giây (bao gồm tìm kiếm + tạo câu trả lời)
- **Khởi động**: ~10-15 giây (tải mô hình lần đầu)

### Sử dụng tài nguyên
- **RAM**: 4-6GB khi chạy
- **Ổ cứng**: ~6GB cho mô hình Ollama + vector database
- **CPU**: Sử dụng đa lõi khi có thể

## 🎯 Ví dụ sử dụng

### Câu hỏi mẫu
- "Luật giao thông đường bộ quy định gì về tốc độ xe máy?"
- "Điều kiện để được cấp giấy phép lái xe là gì?"
- "Luật đất đai quy định gì về quyền sử dụng đất?"
- "Xử phạt vi phạm giao thông như thế nào?"

### Cấu trúc thư mục sau cài đặt
```
simple_rag/
├── data/
│   ├── raw/           # Tài liệu gốc
│   └── processed/     # Tài liệu đã xử lý
├── models/
│   └── chromadb/      # Vector database
├── src/               # Source code
├── .env               # Environment variables
├── config.py          # Cấu hình hệ thống
└── requirements.txt   # Dependencies
```

## ✅ Kiểm tra cuối cùng

Sau khi cài đặt, chạy lệnh sau để đảm bảo mọi thứ hoạt động:

```bash
python test_complete_local_rag.py
```

Nếu thấy `Final result: PASS ✅`, hệ thống đã sẵn sàng sử dụng!

## 🆘 Hỗ trợ

Nếu gặp vấn đề:
1. Kiểm tra phần Troubleshooting ở trên
2. Đảm bảo đã cài đặt đúng các yêu cầu
3. Kiểm tra log lỗi chi tiết
4. Thử chạy lại từ đầu nếu cần

---

**Chúc bạn sử dụng hệ thống hiệu quả! 🚀**