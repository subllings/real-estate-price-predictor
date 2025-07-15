#!/bin/bash

# Test script to verify admin panel integration
echo "🔧 Testing Admin Panel Integration..."

echo "📦 Checking dependencies..."
cd app/frontend-react

# Check if lucide-react is installed
if npm list lucide-react > /dev/null 2>&1; then
    echo "✅ lucide-react is installed"
else
    echo "❌ Installing lucide-react..."
    npm install lucide-react
fi

# Check if react-router-dom is installed  
if npm list react-router-dom > /dev/null 2>&1; then
    echo "✅ react-router-dom is installed"
else
    echo "❌ Installing react-router-dom..."
    npm install react-router-dom
fi

echo "🚀 Starting development server..."
echo "   - Admin panel accessible with Ctrl+A"
echo "   - Available on all pages for demo"
echo "   - Check console for any remaining errors"

# Start on different port if 3000 is busy
PORT=3001 npm start
