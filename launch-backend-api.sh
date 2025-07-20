#!/bin/bash

# Start Real Estate AI Platform Backend API
echo "🚀 Starting Real Estate AI Platform Backend..."

cd "$(dirname "$0")/app/backend"

# Check if Python environment is set up
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found. Please install Python 3.7+ to run the backend."
    exit 1
fi

# Install required packages if not installed
echo "📦 Checking dependencies..."
$PYTHON_CMD -c "import fastapi, uvicorn" 2>/dev/null || {
    echo "📦 Installing FastAPI and Uvicorn..."
    pip install fastapi uvicorn python-multipart
}

echo "🌟 Starting backend API server on http://localhost:8002"
echo "📋 Available endpoints:"
echo "   • ESG Chat: POST /api/chat"
echo "   • Training Jobs: GET /api/training-jobs"
echo "   • Health Check: GET /"
echo ""
echo "💡 Use Ctrl+C to stop the server"
echo ""

# Start the FastAPI server
$PYTHON_CMD main.py
