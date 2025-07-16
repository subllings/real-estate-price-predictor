@echo off
echo Lancement de l'application React...
echo.

cd /d "E:\_SoftEng\_BeCode\real-estate-price-predictor\app\frontend-react"

echo Verification des dependances...
if not exist "node_modules" (
    echo Installation des dependances...
    npm install
)

echo Lancement de l'application React...
npm start

pause
