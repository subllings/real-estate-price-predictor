@echo off
REM Script de lancement pour LAPTOP Windows (Slave/Backup)
REM Usage: launch_laptop_slave.bat

echo 💻 LANCEMENT LAPTOP WINDOWS - MODE SLAVE/BACKUP
echo ==============================================

REM Vérifier les prérequis
if not exist "distributed_training_manager.py" (
    echo ❌ distributed_training_manager.py manquant
    echo    💡 Copiez les fichiers depuis le desktop ou clonez le repo
    pause
    exit /b 1
)

REM Définir variables d'environnement
set MACHINE_ROLE=slave
set MACHINE_TYPE=laptop
set TRAINING_PRIORITY=medium

REM Afficher infos système
echo 📊 Informations système:
echo    - Machine: %COMPUTERNAME%
echo    - Rôle: SLAVE/BACKUP (Laptop Windows)
echo    - Date: %DATE% %TIME%
echo    - Dossier: %CD%
echo.

REM Vérifier si git est synchronisé
if exist ".git" (
    echo 🔄 Synchronisation Git...
    git fetch origin >nul 2>&1
    git pull origin main >nul 2>&1
    if errorlevel 1 echo ⚠️ Pas de mise à jour Git
)

REM Vérifier l'environnement Python
echo 🐍 Vérification environnement Python...
if exist "requirements.txt" (
    pip install -r requirements.txt -q >nul 2>&1
    if errorlevel 1 echo ⚠️ Certains packages manquants
)

REM Vérifier la connexion Azure
echo ☁️ Vérification Azure...
python -c "try: from utils.azure_model_storage import AzureModelStorage; storage = AzureModelStorage(); models = storage.list_all_models(); print(f'✅ Azure OK - {len(models)} modèles trouvés'); except Exception as e: print(f'⚠️ Azure: {e}'); print('   💡 Assurez-vous que les credentials Azure sont configurés')" 2>nul

REM Vérifier le statut du master
echo.
echo 🔍 Recherche du master (desktop)...
if exist "distributed_training_status.json" (
    python -c "import json; from datetime import datetime; status = json.load(open('distributed_training_status.json', 'r')); current_master = status.get('current_master', 'Aucun'); machines = status.get('machines', {}); print(f'🖥️ Master actuel: {current_master}'); [print(f'   - {machine_id} ({info.get(\"machine_role\", \"unknown\")}): {info.get(\"training_status\", \"unknown\")} - {info.get(\"last_heartbeat\", \"unknown\")}') for machine_id, info in machines.items()]" 2>nul
    if errorlevel 1 echo ⚠️ Erreur lecture statut
) else (
    echo ⚠️ Aucun fichier de statut trouvé
    echo    💡 Soit le desktop n'a pas encore démarré, soit c'est un premier lancement
)

echo.
echo 🤖 MODES DISPONIBLES:
echo    1. 🔄 Mode SURVEILLANCE (par défaut)
echo       - Surveille le desktop
echo       - Prend le relais automatiquement si besoin
echo    2. 🚀 Mode FORCE MASTER
echo       - Démarre immédiatement en master
echo       - Utile si desktop HS
echo.
set /p MODE_CHOICE="Choisir mode (1=surveillance, 2=force): "

if "%MODE_CHOICE%"=="2" (
    set LAUNCH_MODE=master
    echo 👑 Mode FORCE MASTER sélectionné
) else (
    set LAUNCH_MODE=slave
    echo 🔄 Mode SURVEILLANCE sélectionné
)

echo.
echo 🚀 Démarrage du système distribué...
echo    - Ctrl+C pour arrêt
echo    - Logs en temps réel

REM Créer nom de fichier de log
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set DATE_LOG=%%c%%b%%a
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set TIME_LOG=%%a%%b
set LOG_FILE=laptop_slave_%DATE_LOG%_%TIME_LOG%.log

echo.
echo 📊 MONITORING EN TEMPS RÉEL
echo =============================
echo 📝 Logs: %LOG_FILE%
echo.

REM Lancer le système distribué
python distributed_training_launcher.py %LAUNCH_MODE% 2>&1 | tee %LOG_FILE%

REM Vérifier le code de sortie
set EXIT_CODE=%ERRORLEVEL%

echo.
echo 📋 RAPPORT FINAL LAPTOP
echo =======================
echo    - Code de sortie: %EXIT_CODE%
echo    - Logs: %LOG_FILE%
echo    - Status final dans: distributed_training_status.json

if exist "night_report.txt" (
    echo.
    echo 📊 Rapport de nuit:
    type night_report.txt
)

echo.
echo ✅ Mission accomplie sur laptop!

pause
