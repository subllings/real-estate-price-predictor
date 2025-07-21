#!/bin/bash

# cd e:/_SoftEng/_BeCode/real-estate-price-predictor/app/backend-api-price-prediction
# chmod +x start-api-price-prediction.sh
# ./start-api-price-prediction.sh

ENV_NAME="real-estate-price-api"

echo "🤖 Starting Price Prediction API..."

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Check if environment exists
if ! conda env list | grep -q "${ENV_NAME}"; then
    echo "❌ Conda environment '${ENV_NAME}' not found. Run ./setup-conda-env-api-price-prediction.sh first"
    exit 1
fi

# Activate environment
conda activate "${ENV_NAME}"

echo "🔍 Testing ML dependencies..."
python -c "import pandas, numpy, sklearn; print('✅ Basic ML OK')" || {
    echo "❌ Basic ML dependencies missing"
    echo "💡 Please run ./setup-conda-env-api-price-prediction.sh again"
    exit 1
}

echo "🔍 Testing boosting libraries..."
python -c "import catboost, xgboost, lightgbm, joblib; print('✅ Boosting libraries OK')" || {
    echo "❌ Boosting libraries missing"
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

python -m uvicorn main:app --host 127.0.0.1 --port 8020 --reload
echo "🌟 Starting server at http://127.0.0.1:8020"
echo "📖 Documentation: http://127.0.0.1:8020/docs"
echo "❤️  Health check: http://127.0.0.1:8020/health"
echo "🤖 ML Models: CatBoost, XGBoost, LightGBM"
echo ""
echo "ℹ️  Note: Dependency warnings are normal and can be ignored"
echo "Press Ctrl+C to stop..."

python -m uvicorn main:app --host 127.0.0.1 --port 8020 --reload
./conda-env/python -m uvicorn main:app --host 127.0.0.1 --port 8020 --reload
