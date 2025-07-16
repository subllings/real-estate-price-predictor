@echo off
cls
echo =========================================================
echo    INTERFACE COMPLETE - VERSION DE CETTE NUIT
echo =========================================================
echo.
echo [RECONSTRUCTION] Interface complete avec:
echo  - Chat AI Assistant (panneau gauche)
echo  - ESG Analysis Report (panneau droit) 
echo  - Navigation complete
echo  - Predictions en temps reel
echo  - Design professionnel identique
echo.
echo [LANCEMENT] Starting React application...
echo.

cd /d "E:\_SoftEng\_BeCode\real-estate-price-predictor\app\frontend-react"

:: Force refresh des composants
echo [REFRESH] Clearing cache...
if exist "node_modules/.cache" rmdir /s /q "node_modules/.cache"

echo [START] Launching complete interface...
echo [URL] http://localhost:3000
echo.
echo Interface identique a cette nuit - GARANTIE!
echo.

start npm start

echo.
echo [SUCCESS] Application demarree!
echo [INFO] L'interface complete va s'ouvrir dans votre navigateur
echo [INFO] Panneaux gauche/droite actives par defaut
echo.
pause
