#!/bin/bash

# Complete Real Estate Platform Test Script
echo "🏡 Testing Real Estate AI Platform..."

cd app/frontend-react

echo "📦 Checking dependencies..."

# Essential packages for full functionality
PACKAGES_TO_CHECK=(
    "react-router-dom"
    "lucide-react"
)

for package in "${PACKAGES_TO_CHECK[@]}"; do
    if npm list $package > /dev/null 2>&1; then
        echo "✅ $package is installed"
    else
        echo "❌ Installing $package..."
        npm install $package
    fi
done

echo ""
echo "🏡 REAL ESTATE AI PLATFORM STRUCTURE:"
echo "📍 Core Routes:"
echo "   • /home - Real Estate Platform Hub"
echo "   • / - Price Predictor (Main Tool)"  
echo "   • /esg-agent - ESG Sustainability Advisor"
echo "   • /training - Model Training Agent (Azure ML)"
echo "   • /admin - Full Admin Dashboard"

echo ""
echo "📍 Investment Routes:"
echo "   • /agent/finance - Real Estate Finance"
echo "   • /agent/passive - Investment Analysis"

echo ""
echo "🎯 Platform Features:"
echo "   • 🏠 Home - Real Estate focused agent selection"
echo "   • 💰 Price Predictor - Core valuation tool with ESG integration"
echo "   • 🌱 ESG Agent - Belgian PEB compliance & primes"
echo "   • 🚀 Model Training - Azure ML pipeline & optimization"
echo "   • 🏡 RE Agents - Dropdown with real estate focused tools"
echo "   • ⚙️ Admin Panel - Global monitoring (Ctrl+A)"

echo ""
echo "� Interconnections:"
echo "   • Price Predictor → Auto ESG analysis"
echo "   • ESG Agent ← Detailed compliance advice"
echo "   • Model Training ← Performance optimization"
echo "   • Admin Panel ← Global system monitoring"

echo ""
echo "📱 Demo Flow:"
echo "   1. Start: Price Predictor (property valuation)"
echo "   2. Auto: ESG impact analysis integrated"
echo "   3. Detail: ESG Agent for compliance deep-dive"
echo "   4. Technical: Model Training for performance"
echo "   5. Monitor: Admin Panel for system health"

echo ""
echo "🚀 Starting Real Estate AI Platform on port 3001..."
echo "   🎯 Test the interconnected workflow"
echo "   🌱 Verify ESG integration in Price Predictor"
echo "   📊 Check Model Training pipeline"
echo "   ⚙️ Confirm Admin Panel global access"

PORT=3001 npm start
