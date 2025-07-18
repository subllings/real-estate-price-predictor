#!/bin/bash

echo "🚀 Testing AdminPanel visibility issue..."

# Navigate to the React app directory
cd "e:\_SoftEng\_BeCode\real-estate-price-predictor\app\frontend-react"

# Kill any existing React dev server
echo "🔪 Killing existing React processes..."
pkill -f "react-scripts start" || true
pkill -f "npm start" || true

# Wait a moment
sleep 2

# Start the React app
echo "🌟 Starting React app..."
echo "📝 After starting:"
echo "  1. Navigate to http://localhost:3000"
echo "  2. Click 'Admin Panel' in the top menu"
echo "  3. Check if SimpleAdminPanelTest appears"
echo "  4. Check browser console for debug logs"

npm start
