@echo off
cd /d E:\_SoftEng\_BeCode\real-estate-price-predictor

echo Starting LLM API...
start "" bash run-backend-api-llm.sh

timeout /t 3 >nul

echo Starting Price Prediction API...
start "" bash run-backend-api-price-prediction.sh

echo All backends launched.
pause
