@echo off
echo Starting RAG Web Interface...
echo.

REM Change to project root directory
cd /d "C:\Users\votru\OneDrive - vthnb\Desktop\RAG2"

REM Activate virtual environment
echo Activating virtual environment...
call .\.venv\Scripts\activate.bat

REM Change to simple_rag directory
cd simple_rag

REM Check if Ollama is running
echo Checking Ollama status...
ollama list >nul 2>&1
if %errorlevel% neq 0 (
    echo Warning: Ollama might not be running. Please start Ollama first.
    echo You can start Ollama by running: ollama serve
    echo.
)

REM Run Streamlit web interface
echo Starting Streamlit web interface...
echo The web interface will be available at: http://localhost:8501
echo.
python -m streamlit run src/web_interface.py

pause
