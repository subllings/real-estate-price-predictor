#!/bin/bash

# cd e:/_SoftEng/_BeCode/real-estate-price-predictor/app/frontend-react
# chmod +x check-react.sh
# ./check-react.sh

echo "🔍 Checking React app status..."

# Check if React is running on port 3000
if curl -s http://localhost:3000 > /dev/null; then
    echo "✅ React app is running on http://localhost:3000"
    echo "🌐 Open in browser: http://localhost:3000"
else
    echo "❌ React app not running on port 3000"
    echo "🚀 Starting React app..."
    npm start &
    echo "⏳ Waiting 10 seconds for startup..."
    sleep 10
    if curl -s http://localhost:3000 > /dev/null; then
        echo "✅ React app now running on http://localhost:3000"
    else
        echo "❌ Failed to start React app"
        echo "📋 Try manually: npm start"
    fi
fi

# Show running processes
echo ""
echo "🔍 Node processes:"
ps aux | grep -E "(node|npm)" | grep -v grep || echo "No Node processes found"
