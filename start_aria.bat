@echo off
echo Starting ARIA...

REM Start Ollama in background
start "Ollama" ollama serve

REM Wait 3 seconds for Ollama to load
timeout /t 3 /nobreak >nul

REM Start ARIA backend
start "ARIA Backend" cmd /k ".venv\Scripts\python.exe server.py"

REM Wait 3 seconds for backend to load
timeout /t 3 /nobreak >nul

REM Start React frontend
start "ARIA Frontend" cmd /k "cd app && npm run dev"

REM Wait 5 seconds then open browser
timeout /t 5 /nobreak >nul
start http://localhost:1420

echo ARIA is starting... open http://localhost:1420