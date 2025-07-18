#!/bin/bash

# Test script to validate risk assessment improvements
echo "🔍 Testing Risk Assessment Improvements..."

# Check if the risk assessment section has been improved
echo "Checking Risk Assessment section in SidePanel.jsx..."
grep -A 10 "Risk Assessment" app/frontend-react/src/components/SidePanel/SidePanel.jsx

echo ""
echo "✅ Risk Assessment section should now include specific categories:"
echo "   - Market volatility and economic factors"
echo "   - Property-specific risks"
echo "   - ESG compliance risks"
echo "   - Financial risks"
echo "   - Operational risks"
echo "   - Climate and environmental risks"

echo ""
echo "✅ System message should now include specific instructions for Risk Assessment"

echo ""
echo "🚀 Ready to test with a real property analysis to verify the LLM response quality."
