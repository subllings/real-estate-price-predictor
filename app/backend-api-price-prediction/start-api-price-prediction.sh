#!/bin/bash

# cd e:/_SoftEng/_BeCode/real-estate-price-predictor/app/backend-api-price-prediction
# chmod +x start-api-price-prediction.sh
# ./start-api-price-prediction.sh

echo "🤖 Starting Price Prediction API (using base environment)..."

echo "🔍 Testing dependencies..."
python -c "import fastapi, uvicorn; print('✅ FastAPI OK')" || {
    echo "❌ FastAPI not installed. Run ./setup-conda-env-api-price-prediction.sh first"
    exit 1
}

echo "🔍 Testing ML dependencies..."
python -c "import pandas, sklearn, catboost; print('✅ ML OK')" || {
    echo "❌ ML dependencies missing. Run ./setup-conda-env-api-price-prediction.sh first"
    exit 1
}

echo "🔍 Testing Azure Cosmos DB..."
python -c "import azure.cosmos; print('✅ Azure Cosmos DB OK')" || {
    echo "❌ Azure Cosmos DB missing. Install with: pip install azure-cosmos"
    exit 1
}

echo "🌟 Starting server at http://127.0.0.1:8000"
echo "📖 Documentation: http://127.0.0.1:8000/docs"
echo "❤️  Health check: http://127.0.0.1:8000/health"
echo ""
echo "Press Ctrl+C to stop..."

python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
    echo "💡 Please run ./setup-conda-env-api-price-prediction.sh again"
    exit 1
}

echo "🔍 Testing .env configuration..."
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found - API will start with default settings"
else
    echo "✅ .env file found"
fi

echo "🌟 Starting server at http://127.0.0.1:8020"
echo "📖 Documentation: http://127.0.0.1:8020/docs"
echo "❤️  Health check: http://127.0.0.1:8020/health"
echo "🤖 ML Models: CatBoost, XGBoost, LightGBM"
echo ""
echo "ℹ️  Note: Dependency warnings are normal and can be ignored"
echo "Press Ctrl+C to stop..."

./conda-env/python -m uvicorn main:app --host 127.0.0.1 --port 8020 --reload
echo "🌟 Starting server at http://127.0.0.1:8020"
echo "📖 Documentation: http://127.0.0.1:8020/docs"
echo "❤️  Health check: http://127.0.0.1:8020/health"
echo "🤖 ML Models: CatBoost, XGBoost, LightGBM"
echo ""
echo "ℹ️  Note: Dependency warnings are normal and can be ignored"
echo "Press Ctrl+C to stop..."

./conda-env/python -m uvicorn main:app --host 127.0.0.1 --port 8020 --reload
