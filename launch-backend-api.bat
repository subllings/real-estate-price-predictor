@echo off
echo 🚀 Starting Real Estate AI Platform Backend...

cd /d "%~dp0\app\backend"

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.7+ to run the backend.
    pause
    exit /b 1
)

REM Install required packages
echo 📦 Checking dependencies...
python -c "import fastapi, uvicorn" >nul 2>&1
if %errorlevel% neq 0 (
    echo 📦 Installing FastAPI and Uvicorn...
    pip install fastapi uvicorn python-multipart
)

echo 🌟 Starting backend API server on http://localhost:8002
echo 📋 Available endpoints:
echo    • ESG Chat: POST /api/chat
echo    • Training Jobs: GET /api/training-jobs
echo    • Health Check: GET /
echo.
echo 💡 Use Ctrl+C to stop the server
echo.

REM Start the FastAPI server
python main.py

pause
