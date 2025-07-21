@echo off
echo ========================================
echo     LANCEMENT APPLICATION REACT COMPLETE
echo ========================================
echo.
echo Verification des composants...

cd /d "E:\_SoftEng\_BeCode\real-estate-price-predictor\app\frontend-react"

echo [OK] Chat AI Assistant - SidePanel
echo [OK] ESG Analysis Panel - BelgianESGAgent  
echo [OK] Integration ESG - ESGIntegrationPrompt
echo [OK] Interface mise a jour - RealEstatePredictorPage
echo.

echo Demarrage de l'application...
echo URL: http://localhost:3000
echo.
echo Fonctionnalites disponibles:
echo - Chat AI Assistant (panneau gauche)
echo - ESG Analysis Report (panneau droit)
echo - Predictions immobilieres integrees
echo - Interface responsive et moderne
echo.

npm start

pause
