@echo off
echo 🚀 Lancement de l'interface React - Model Training
echo ================================================

echo 🔍 Vérification de l'API...
curl -s http://127.0.0.1:8000/health >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ API accessible sur le port 8000
) else (
    echo ❌ API non accessible sur le port 8000
    echo    Assurez-vous que l'API est démarrée avec:
    echo    python app/backend-api-price-prediction/main.py
    echo    ou
    echo    ./run-backend-api-price-prediction.sh
    pause
    exit /b 1
)

echo 🧪 Vérification des données d'expériences...
curl -s http://127.0.0.1:8000/experiments >nul 2>&1
if %errorlevel% equ 0 (
    echo ✅ Endpoint /experiments accessible
) else (
    echo ❌ Endpoint /experiments non accessible
    pause
    exit /b 1
)

echo 📁 Navigation vers le dossier React...
cd app\frontend-react

echo 📦 Vérification des dépendances...
if not exist node_modules (
    echo Installation des dépendances npm...
    call npm install
)

echo ⚛️  Démarrage de l'application React...
echo    Interface disponible sur: http://localhost:3000
echo    Page Model Training: http://localhost:3000/training
echo.
echo 🎯 L'interface affichera les données réelles depuis CosmosDB
echo    - Tableau des expériences avec R² scores
echo    - Statistiques de résumé
echo    - État en temps réel des modèles
echo.
echo 📱 Naviguez vers 'Model Training' dans le menu pour voir l'interface
echo.

call npm start
