#!/bin/bash

# Script to test AdminPanel integration

echo "🚀 Starting AdminPanel Integration Test..."

# Navigate to React app directory
cd "e:\_SoftEng\_BeCode\real-estate-price-predictor\app\frontend-react"

echo "📁 Current directory: $(pwd)"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Check if our AdminPanel files exist
if [ -f "src/components/AdminPanel/AdminPanel.jsx" ]; then
    echo "✅ AdminPanel.jsx found"
else
    echo "❌ AdminPanel.jsx not found!"
    exit 1
fi

if [ -f "src/components/AdminPanel/AdminPanel.css" ]; then
    echo "✅ AdminPanel.css found"
else
    echo "❌ AdminPanel.css not found!"
    exit 1
fi

# Start the development server
echo "🌟 Starting React development server..."
echo "🔧 Navigate to the app and toggle Admin Panel from the menu"
echo "🎯 Test the Prompt Visualization tab to see captured LLM prompts"

npm start
