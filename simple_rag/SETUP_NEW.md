# 📖 Complete Setup Guide - Vietnamese Law RAG System

Step-by-step installation guide for both local (Ollama) and cloud (OpenAI) deployment.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Setup (Ollama)](#local-setup-ollama)
3. [Cloud Setup (OpenAI)](#cloud-setup-openai)
4. [Verification](#verification)
5. [First Run](#first-run)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

**For Local Setup (Ollama):**
- Operating System: Windows 10/11, macOS 10.15+, or Linux
- RAM: 8GB minimum (16GB recommended)
- Storage: 10GB free space
- CPU: 64-bit processor (Intel/AMD)
- Python: 3.8, 3.9, 3.10, 3.11, 3.12, or 3.13

**For Cloud Setup (OpenAI):**
- Operating System: Any (Windows/macOS/Linux)
- RAM: 4GB minimum
- Storage: 2GB free space
- Python: 3.8 - 3.13
- Internet: Stable connection required
- OpenAI Account: With API access and credits

### Check Your Python Version

```bash
# Check Python version
python --version

# Should output: Python 3.x.x (where x is 8-13)
```

⚠️ **Important**: Avoid Python 3.14 alpha versions - they may have compatibility issues.

---

## Local Setup (Ollama)

### Step 1: Install Ollama

#### Windows

1. Visit [ollama.ai/download](https://ollama.ai/download)
2. Download `OllamaSetup.exe`
3. Run the installer
4. Follow installation wizard
5. Ollama will start automatically

#### macOS

1. Visit [ollama.ai/download](https://ollama.ai/download)
2. Download `Ollama-darwin.zip`
3. Unzip and drag to Applications
4. Launch Ollama from Applications

#### Linux

```bash
curl https://ollama.ai/install.sh | sh
```

### Step 2: Verify Ollama Installation

```bash
# Check Ollama is running
ollama --version

# Should output: ollama version x.x.x
```

### Step 3: Download AI Models

Open terminal/command prompt and run:

```bash
# Download language model (8.2B parameters, ~4.9GB)
ollama pull deepseek-r1

# Download embedding model (334M parameters, ~669MB)
ollama pull mxbai-embed-large
```

⏱️ **Time Required**: 10-30 minutes depending on internet speed

Verify models are downloaded:

```bash
ollama list
```

Expected output:
```
NAME                    SIZE
deepseek-r1:latest     4.9 GB
mxbai-embed-large:latest   669 MB
```

### Step 4: Clone Repository

```bash
# Navigate to your projects folder
cd C:\Users\YourName\Documents\GitHub  # Windows
# or
cd ~/projects  # macOS/Linux

# Clone the repository
git clone https://github.com/hohiphop1234/RAG2.git

# Navigate to project
cd RAG2/simple_rag
```

### Step 5: Create Virtual Environment

**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Windows (Command Prompt):**
```cmd
# Create virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate
```

You should see `(.venv)` at the start of your command prompt.

### Step 6: Install Python Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

⏱️ **Time Required**: 3-5 minutes

Expected packages:
- streamlit
- chromadb
- ollama
- PyPDF2
- python-docx
- beautifulsoup4
- sentence-transformers
- numpy
- and more...

### Step 7: Verify Configuration

The default configuration should already be set for Ollama. Let's verify:

```bash
# Open config.py and check these settings:
```

Ensure `config.py` has:
```python
EMBEDDING_PROVIDER: str = "ollama"
EMBEDDING_MODEL: str = "mxbai-embed-large"
LLM_PROVIDER: str = "ollama"
LLM_MODEL: str = "deepseek-r1"
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
```

✅ These should be the default values - no changes needed!

### Step 8: Test Installation

```bash
# Test if Ollama is accessible
curl http://localhost:11434/api/tags

# Should return JSON with model list
```

---

## Cloud Setup (OpenAI)

### Step 1: Get OpenAI API Key

1. Visit [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Go to API Keys section
4. Click "Create new secret key"
5. Copy the key (starts with `sk-...`)

⚠️ **Important**: Store this key securely! You won't be able to see it again.

### Step 2: Clone Repository (if not done)

```bash
# Navigate to your projects folder
cd C:\Users\YourName\Documents\GitHub  # Windows
# cd ~/projects  # macOS/Linux

# Clone the repository
git clone https://github.com/hohiphop1234/RAG2.git

# Navigate to project
cd RAG2/simple_rag
```

### Step 3: Create Virtual Environment (if not done)

Follow Step 5 from [Local Setup](#step-5-create-virtual-environment)

### Step 4: Install Python Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### Step 5: Configure OpenAI

**Create `.env` file:**

```bash
# Windows (PowerShell)
New-Item -Path .env -ItemType File

# macOS/Linux
touch .env
```

**Edit `.env` file** and add:
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### Step 6: Update Configuration

Edit `config.py` and change these lines:

```python
# Find these lines and change them:
EMBEDDING_PROVIDER: str = "openai"  # Changed from "ollama"
EMBEDDING_MODEL: str = "text-embedding-3-small"  # Changed from "mxbai-embed-large"

LLM_PROVIDER: str = "openai"  # Changed from "ollama"
LLM_MODEL: str = "gpt-4"  # Changed from "deepseek-r1"
```

**Cost-effective alternative:**
```python
LLM_MODEL: str = "gpt-3.5-turbo"  # Cheaper than gpt-4
```

---

## Verification

### Test Ollama Setup

```bash
# Activate virtual environment first
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

# Run test
python -c "import ollama; client = ollama.Client(); print(client.list())"
```

Expected: List of models without errors

### Test OpenAI Setup

```bash
# Activate virtual environment first

# Run test
python -c "import openai; import os; from dotenv import load_dotenv; load_dotenv(); openai.api_key = os.getenv('OPENAI_API_KEY'); print('API Key loaded:', openai.api_key[:10] + '...')"
```

Expected: API key prefix displayed

### Test Complete System

```bash
# For systems with test files:
python test_complete_local_rag.py
```

Expected output: `Final result: PASS ✅`

---

## First Run

### Add Sample Documents

1. **Create sample document:**

**Windows:**
```powershell
New-Item -Path "data\raw\sample_law.txt" -ItemType File -Force
```

**macOS/Linux:**
```bash
mkdir -p data/raw
touch data/raw/sample_law.txt
```

2. **Add some content** to `data/raw/sample_law.txt`:

```
Luật Giao thông Đường bộ Việt Nam

Điều 1: Phạm vi điều chỉnh
Luật này quy định về quy tắc giao thông đường bộ, trật tự, an toàn giao thông đường bộ.

Điều 2: Tốc độ xe máy
Trong khu vực đông dân cư: Tốc độ tối đa 50 km/h
Ngoài khu vực đông dân cư: Tốc độ tối đa 60 km/h
Trên đường cao tốc: Không được phép đi xe máy

Điều 3: Giấy phép lái xe
Điều kiện cấp giấy phép lái xe:
- Đủ 18 tuổi trở lên
- Có đủ sức khỏe
- Thi đạt chứng chỉ đào tạo lái xe
```

### Launch Web Interface

```bash
# Make sure virtual environment is activated
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

# Run Streamlit
streamlit run src/web_interface.py
```

**Or use shortcuts:**

**Windows:**
```bash
# Double-click
run_web.bat

# Or in PowerShell
.\run_web.ps1
```

### Access the Interface

1. Browser should open automatically to: http://localhost:8501
2. If not, manually open the URL
3. You should see: "🇻🇳 Chatbot pháp luật Việt Nam"

### First-Time Usage

1. **Check sidebar** - Should show "✅ RAG Pipeline: Sẵn sàng"
2. **Upload your document**:
   - Click "📁 Quản lý tài liệu"
   - Select files or use the sample already in `data/raw/`
   - Click "🔄 Xử lý tài liệu"
   - Wait for processing (30 seconds - 2 minutes)

3. **Ask a question**:
   - Type: "Tốc độ xe máy trong khu dân cư là bao nhiêu?"
   - Click "📤 Gửi"
   - View answer with sources

---

## Troubleshooting

### Common Issues

#### 1. "ModuleNotFoundError"

**Problem**: Python can't find installed packages

**Solution**:
```bash
# Make sure virtual environment is activated
# You should see (.venv) in your prompt

# If not activated:
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

#### 2. "Connection refused" (Ollama)

**Problem**: Can't connect to Ollama server

**Solution**:
```bash
# Check if Ollama is running
ollama list

# If not running:
# Windows: Restart Ollama from Start Menu
# macOS: Open Ollama from Applications
# Linux: sudo systemctl restart ollama
```

#### 3. "Invalid API key" (OpenAI)

**Problem**: OpenAI authentication failed

**Solution**:
```bash
# Check .env file exists and contains key
# Windows:
type .env

# macOS/Linux:
cat .env

# Should show: OPENAI_API_KEY=sk-...

# If not, create/edit .env file and add your key
```

#### 4. "Model not found" (Ollama)

**Problem**: AI model not downloaded

**Solution**:
```bash
# List current models
ollama list

# Pull missing models
ollama pull deepseek-r1
ollama pull mxbai-embed-large

# Verify
ollama list
```

#### 5. "Out of memory"

**Problem**: System running out of RAM

**Solution**:
```python
# Edit config.py and reduce these values:
CHUNK_SIZE = 500  # Changed from 1000
BATCH_SIZE = 50   # Changed from 100
TOP_K_RESULTS = 3 # Changed from 5
```

Or use a smaller Ollama model:
```bash
ollama pull llama3.2  # Smaller alternative
```

Then update config.py:
```python
LLM_MODEL = "llama3.2"
```

#### 6. "Streamlit not found"

**Problem**: Streamlit not installed or not in PATH

**Solution**:
```bash
# Ensure virtual environment is activated
# Then install/reinstall
pip install --upgrade streamlit

# Try running with python -m
python -m streamlit run src/web_interface.py
```

#### 7. Unicode/Encoding Errors

**Problem**: Can't process Vietnamese text

**Solution**:
- Ensure Python 3.8-3.13 (not 3.14)
- Check file encoding:
```bash
# Files should be UTF-8 encoded
# Re-save problematic files as UTF-8
```

#### 8. Port Already in Use

**Problem**: Port 8501 already occupied

**Solution**:
```bash
# Run on different port
streamlit run src/web_interface.py --server.port 8502

# Or find and kill process using port 8501:
# Windows:
netstat -ano | findstr :8501
taskkill /PID <process_id> /F

# macOS/Linux:
lsof -ti:8501 | xargs kill -9
```

### Getting Help

If issues persist:

1. **Check logs**:
   - Terminal output shows detailed errors
   - Look for stack traces

2. **Verify installation**:
   ```bash
   # Python version
   python --version
   
   # Installed packages
   pip list
   
   # Ollama status
   ollama list
   ```

3. **Start fresh**:
   ```bash
   # Delete virtual environment
   # Windows: rmdir /s .venv
   # macOS/Linux: rm -rf .venv
   
   # Recreate and reinstall
   python -m venv .venv
   # Activate and reinstall dependencies
   ```

4. **Check system resources**:
   - Available RAM
   - Disk space
   - CPU usage

---

## Next Steps

After successful setup:

1. **Add your documents** to `data/raw/`
2. **Customize settings** in `config.py`
3. **Start asking questions** in the web interface
4. **Monitor performance** and adjust as needed

## Performance Tips

### For Better Speed (Ollama):
- Use SSD storage
- Close unnecessary applications
- Increase RAM if possible
- Use smaller models for testing

### For Lower Costs (OpenAI):
- Use `gpt-3.5-turbo` instead of `gpt-4`
- Use `text-embedding-3-small` instead of `text-embedding-3-large`
- Reduce `MAX_TOKENS` in config
- Process fewer documents at once

---

## Summary Checklist

- [ ] Python 3.8-3.13 installed
- [ ] Ollama installed (local) OR OpenAI API key (cloud)
- [ ] Models downloaded (local) OR API key configured (cloud)
- [ ] Repository cloned
- [ ] Virtual environment created and activated
- [ ] Dependencies installed
- [ ] Configuration verified
- [ ] Test passed
- [ ] Web interface launches
- [ ] Sample documents added
- [ ] First question answered successfully

✅ **All checked? You're ready to go!**

---

**Need more help?** Check the main README.md or open an issue on GitHub.
