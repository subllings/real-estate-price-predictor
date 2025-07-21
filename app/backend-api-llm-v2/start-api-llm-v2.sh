#!/bin/bash

# cd e:/_SoftEng/_BeCode/real-estate-price-predictor/app/backend-api-llm-v2
# chmod +x start-api-llm-v2.sh
# ./start-api-llm-v2.sh

echo "🚀 Starting Azure OpenAI LLM API v2..." 

if [ ! -d "./conda-env" ]; then
    echo "❌ Conda environment not found. Run ./setup-conda-env-api-llm.sh first"
    exit 1
fi

echo "🔍 Testing environment..."
./conda-env/python --version || exit 1

echo "🔍 Testing dependencies..."
./conda-env/python -c "import fastapi, uvicorn; print('✅ FastAPI OK')" || {
    echo "❌ FastAPI not installed correctly"
    echo "💡 Please run ./setup-conda-env-api-llm.sh again"
    exit 1
}

echo "🔍 Testing Azure OpenAI dependencies..."
./conda-env/python -c "import openai, langchain; print('✅ Azure OpenAI OK')" || {
    echo "❌ Azure OpenAI dependencies missing"
    echo "💡 Please run ./setup-conda-env-api-llm.sh again"
    exit 1
}

echo "🔍 Testing document processing..."
./conda-env/python -c "import fitz, docx, faiss; print('✅ Document processing OK')" || {
    echo "❌ Document processing dependencies missing"
    echo "💡 Please run ./setup-conda-env-api-llm.sh again"
    exit 1
}

echo "🔍 Testing .env configuration..."
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found - API will start but Azure OpenAI won't work"
else
    echo "✅ .env file found"
fi

echo "🌟 Starting server at http://127.0.0.1:8010"
echo "📖 Documentation: http://127.0.0.1:8010/docs"
echo "❤️  Health check: http://127.0.0.1:8010/health"
echo ""
echo "ℹ️  Note: Pydantic warnings about 'model_name' are normal and can be ignored"
echo "Press Ctrl+C to stop..."

./conda-env/python -m uvicorn main:app --host 127.0.0.1 --port 8010 --reload
