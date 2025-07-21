#!/bin/bash

# cd e:/_SoftEng/_BeCode/real-estate-price-predictor/app/frontend-react
# chmod +x check-apis.sh
# ./check-apis.sh

echo "🔍 Checking API connections..."

# Check API LLM (port 8010)
echo "🤖 Testing API LLM on port 8010..."
if curl -s http://localhost:8010/health > /dev/null; then
    echo "✅ API LLM is running on http://localhost:8010"
    echo "📖 Docs: http://localhost:8010/docs"
else
    echo "❌ API LLM not responding on port 8010"
    echo "💡 Restart with: cd ../backend-api-llm-v2 && ./start-api-llm-v2.sh"
fi

echo ""

# Check API Price Prediction (port 8020)
echo "📊 Testing API Price Prediction on port 8020..."
if curl -s http://localhost:8020/health > /dev/null; then
    echo "✅ API Price Prediction is running on http://localhost:8020"
    echo "📖 Docs: http://localhost:8020/docs"
else
    echo "❌ API Price Prediction not responding on port 8020"
    echo "💡 Restart with: cd ../backend-api-price-prediction && ./start-api-price-prediction.sh"
fi

echo ""

# Check React app (port 3000)
echo "⚛️ Testing React app on port 3000..."
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ React app is running on http://localhost:3000"
else
    echo "❌ React app not responding on port 3000"
    echo "💡 Restart with: npm start"
fi

echo ""
echo "🔍 Active ports:"
netstat -an | grep -E ":3000|:8010|:8020" | grep LISTEN || echo "No active ports found"
