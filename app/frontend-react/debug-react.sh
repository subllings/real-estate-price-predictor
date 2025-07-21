#!/bin/bash

# cd e:/_SoftEng/_BeCode/real-estate-price-predictor/app/frontend-react
# chmod +x debug-react.sh
# ./debug-react.sh

echo "🔍 Debugging React app..."

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "❌ node_modules missing. Installing..."
    npm install
fi

# Check package.json
echo "📦 Package info:"
npm list --depth=0 2>/dev/null | head -10

# Start with detailed error reporting
echo "🚀 Starting with error focus..."
npm start 2>&1 | grep -E "(error|Error|ERROR|failed|Failed|FAILED)"
