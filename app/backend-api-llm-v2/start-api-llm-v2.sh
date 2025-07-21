#!/bin/bash

# cd e:/_SoftEng/_BeCode/real-estate-price-predictor/app/backend-api-llm-v2
# chmod +x start-api-llm-v2.sh
# ./start-api-llm-v2.sh

echo "🚀 Starting Azure OpenAI LLM API v2 (using base environment)..." 

echo "🔍 Testing dependencies..."
python -c "import fastapi, uvicorn; print('✅ FastAPI OK')" || {
    echo "❌ FastAPI not installed. Run ./setup-conda-env-api-llm.sh first"
    exit 1
}

echo "🔍 Testing Azure OpenAI dependencies..."
python -c "import openai, langchain; print('✅ Azure OpenAI OK')" || {
    echo "❌ Azure OpenAI dependencies missing. Run ./setup-conda-env-api-llm.sh first"
    exit 1
}

echo "🔍 Testing document processing..."
python -c "import fitz, docx, faiss; print('✅ Document processing OK')" || {
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

python -m uvicorn main:app --host 127.0.0.1 --port 8010 --reload
echo "Press Ctrl+C to stop..."

python -m uvicorn main:app --host 127.0.0.1 --port 8010 --reload
echo "🔍 Testing document processing..."
python -c "import fitz, docx, faiss; print('✅ Document processing OK')" || {
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

python -m uvicorn main:app --host 127.0.0.1 --port 8010 --reload
echo "Press Ctrl+C to stop..."

python -m uvicorn main:app --host 127.0.0.1 --port 8010 --reload
