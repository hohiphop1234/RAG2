# PowerShell script to run RAG Web Interface
Write-Host "Starting RAG Web Interface..." -ForegroundColor Green
Write-Host ""

# Change to project root directory
Set-Location "C:\Users\votru\OneDrive - vthnb\Desktop\RAG2"

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\\.venv\Scripts\Activate.ps1"

# Change to simple_rag directory
Set-Location "simple_rag"

# Check if Ollama is running
Write-Host "Checking Ollama status..." -ForegroundColor Yellow
try {
    ollama list | Out-Null
    Write-Host "Ollama is running ✓" -ForegroundColor Green
} catch {
    Write-Host "Warning: Ollama might not be running. Please start Ollama first." -ForegroundColor Red
    Write-Host "You can start Ollama by running: ollama serve" -ForegroundColor Yellow
    Write-Host ""
}

# Run Streamlit web interface
Write-Host "Starting Streamlit web interface..." -ForegroundColor Green
Write-Host "The web interface will be available at: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""

python -m streamlit run src/web_interface.py
