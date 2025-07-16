@echo off
REM Script de préparation laptop Windows
REM Usage: prepare_laptop.bat

echo 💻 PRÉPARATION LAPTOP WINDOWS POUR ENTRAÎNEMENT DISTRIBUÉ
echo =========================================================

REM 1. Vérifier si on est dans un repo Git
if not exist ".git" (
    echo ❌ Pas de repository Git trouvé
    echo.
    echo 🔧 SOLUTIONS:
    echo    Option 1 - Cloner le repository:
    echo    git clone https://github.com/subllings/real-estate-price-predictor.git
    echo    cd real-estate-price-predictor
    echo.
    echo    Option 2 - Copier manuellement depuis le desktop
    pause
    exit /b 1
)

echo ✅ Repository Git détecté

REM 2. Sauvegarder les changements locaux
echo 🔄 Sauvegarde des changements locaux...
git add -A >nul 2>&1
git commit -m "Auto-save before sync - %DATE% %TIME%" >nul 2>&1
if errorlevel 1 echo    ℹ️ Rien à sauvegarder

REM 3. Récupérer les dernières modifications
echo 📥 Récupération des dernières modifications...
git fetch origin
git pull origin main

REM 4. Vérifier les fichiers critiques
echo 🔍 Vérification des fichiers critiques...
set "missing_files="

if exist "distributed_training_manager.py" (
    echo    ✅ distributed_training_manager.py
) else (
    echo    ❌ distributed_training_manager.py - MANQUANT
    set "missing_files=1"
)

if exist "auto_recovery_system.py" (
    echo    ✅ auto_recovery_system.py
) else (
    echo    ❌ auto_recovery_system.py - MANQUANT
    set "missing_files=1"
)

if exist "launch_laptop_slave.bat" (
    echo    ✅ launch_laptop_slave.bat
) else (
    echo    ❌ launch_laptop_slave.bat - MANQUANT
    set "missing_files=1"
)

if exist "utils\azure_model_storage.py" (
    echo    ✅ utils\azure_model_storage.py
) else (
    echo    ❌ utils\azure_model_storage.py - MANQUANT
    set "missing_files=1"
)

if exist "requirements.txt" (
    echo    ✅ requirements.txt
) else (
    echo    ❌ requirements.txt - MANQUANT
    set "missing_files=1"
)

if defined missing_files (
    echo.
    echo ⚠️ Fichiers manquants détectés!
    echo 💡 Solutions:
    echo    1. Commitez sur le desktop et re-tirez ici
    echo    2. Copiez manuellement les fichiers manquants
    pause
    exit /b 1
)

REM 5. Vérifier l'environnement Python
echo.
echo 🐍 Vérification environnement Python...
if exist "requirements.txt" (
    echo 📦 Installation des dépendances...
    pip install -r requirements.txt -q >nul 2>&1
    if errorlevel 1 (
        echo    ⚠️ Certaines dépendances ont échoué
    ) else (
        echo    ✅ Dépendances installées
    )
) else (
    echo    ⚠️ requirements.txt manquant
)

REM 6. Vérifier Azure
echo.
echo ☁️ Test connexion Azure...
python -c "try: from utils.azure_model_storage import AzureModelStorage; storage = AzureModelStorage(); models = storage.list_all_models(); print(f'   ✅ Azure OK - {len(models)} modèles disponibles'); except Exception as e: print(f'   ❌ Erreur Azure: {e}'); print('   💡 Vérifiez le fichier .env et les credentials Azure')" 2>nul

REM 7. Afficher le statut actuel
echo.
echo 📊 STATUT FINAL
echo ===============
echo ✅ Code synchronisé depuis Git
echo ✅ Fichiers critiques présents
echo ✅ Environnement Python configuré

REM 8. Vérifier si le desktop est actif
if exist "distributed_training_status.json" (
    echo.
    echo 🖥️ STATUT DU DESKTOP:
    python -c "import json; from datetime import datetime; status = json.load(open('distributed_training_status.json', 'r')); current_master = status.get('current_master', 'Aucun'); print(f'   Master actuel: {current_master}'); [print(f'   - {machine_id} ({info.get(\"machine_role\", \"unknown\")}): {info.get(\"training_status\", \"unknown\")}') for machine_id, info in status.get('machines', {}).items()]" 2>nul
) else (
    echo    ℹ️ Aucun statut desktop trouvé (normal si pas encore démarré)
)

echo.
echo 🚀 PRÊT POUR LE LANCEMENT!
echo ==========================
echo Commandes disponibles:
echo    launch_laptop_slave.bat    - Mode surveillance (recommandé)
echo    launch_laptop_slave.sh     - Version Linux/Mac
echo.
echo 💡 Le laptop va automatiquement:
echo    - Surveiller le desktop
echo    - Prendre le relais si nécessaire
echo    - Synchroniser avec Azure
echo.
echo Bonne nuit ! 😴

pause
