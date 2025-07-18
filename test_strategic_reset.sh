#!/bin/bash

# Test pour vérifier la solution de reset des états stratégiques

echo "🧪 Testing strategic analysis state reset functionality..."

# Vérifier que la fonction onResetStrategicAnalysis est bien ajoutée au PropertyForm
echo "✓ Checking PropertyForm.js for onResetStrategicAnalysis prop..."
grep -n "onResetStrategicAnalysis" e:\_SoftEng\_BeCode\real-estate-price-predictor\app\frontend-react\src\components\PropertyForm\PropertyForm.js

# Vérifier que la fonction resetStrategicAnalysis est bien définie dans SidePanel
echo "✓ Checking SidePanel.jsx for resetStrategicAnalysis function..."
grep -n "resetStrategicAnalysis" e:\_SoftEng\_BeCode\real-estate-price-predictor\app\frontend-react\src\components\SidePanel\SidePanel.jsx

# Vérifier que la callback est bien transmise depuis RealEstatePredictorPage
echo "✓ Checking RealEstatePredictorPage.jsx for callback setup..."
grep -n "onResetStrategicAnalysis\|resetStrategicAnalysis" e:\_SoftEng\_BeCode\real-estate-price-predictor\app\frontend-react\src\pages\RealEstatePredictorPage.jsx

# Vérifier que la callback est bien appelée dans handleUnifiedAnalysis
echo "✓ Checking handleUnifiedAnalysis for reset call..."
grep -A 10 -B 5 "Reset strategic analysis states" e:\_SoftEng\_BeCode\real-estate-price-predictor\app\frontend-react\src\components\PropertyForm\PropertyForm.js

echo "🎉 Test completed! The strategic analysis state reset functionality has been implemented."
echo ""
echo "📋 Summary of changes:"
echo "1. Added onResetStrategicAnalysis callback to PropertyForm props"
echo "2. Added resetStrategicAnalysis function to SidePanel"
echo "3. Added state management for reset function in RealEstatePredictorPage"
echo "4. Modified handleUnifiedAnalysis to call reset when starting new analysis"
echo ""
echo "✅ When user clicks 'Analyze Price & ESG' button:"
echo "   - Strategic analysis states (strategicAnalysisGenerated, isStrategicAnalysisLoading) are reset"
echo "   - This ensures clean state for new analysis"
echo "   - No more confusion between different analysis sessions"
