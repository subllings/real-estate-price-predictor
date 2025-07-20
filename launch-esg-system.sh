#!/bin/bash

# Launch Complete ESG Agent System with Authentication
# Real Estate Price Predictor Platform

echo "🚀 Launching Real Estate AI Platform with Authentication..."
echo "📋 Features: ESG Agent + User Profiles + Admin Panel"

cd "$(dirname "$0")/app/frontend-react"

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Check if @react-oauth/google is installed
if ! npm list @react-oauth/google > /dev/null 2>&1; then
    echo "🔐 Installing Google OAuth package..."
    npm install @react-oauth/google
fi

echo "🌟 Starting React development server..."
echo "🔗 Application will be available at: http://localhost:3000"
echo ""
echo "📋 Test Accounts Available:"
echo "   Demo User: demo@company.com / password123"
echo "   Admin User: admin@company.com / admin123"
echo ""
echo "✨ New Features:"
echo "   • User Authentication (Google OAuth2 + Local)"
echo "   • Personalized ESG Questions by Role"
echo "   • User Profile Management"
echo "   • Admin Panel with User Management"
echo "   • Sidebar Chat Interface"
echo ""

npm start
