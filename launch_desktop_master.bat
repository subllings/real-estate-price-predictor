@echo off
REM Script de lancement pour DESKTOP Windows (Master principal)
REM Usage: launch_desktop_master.bat

echo 🖥️ LANCEMENT DESKTOP WINDOWS - MODE MASTER
echo ==========================================

REM Vérifier les prérequis
if not exist "distributed_training_manager.py" (
    echo ❌ distributed_training_manager.py manquant
    pause
    exit /b 1
)

if not exist "auto_recovery_system.py" (
    echo ❌ auto_recovery_system.py manquant
    pause
    exit /b 1
)

REM Définir variables d'environnement
set MACHINE_ROLE=master
set MACHINE_TYPE=desktop
set TRAINING_PRIORITY=high

REM Afficher infos système
echo 📊 Informations système:
echo    - Machine: %COMPUTERNAME%
echo    - Rôle: MASTER (Desktop Windows)
echo    - Date: %DATE% %TIME%
echo    - Dossier: %CD%
echo.

REM Nettoyer anciens fichiers de statut si nécessaire
if exist "distributed_training_status.json" (
    echo 🧹 Nettoyage ancien statut...
    copy "distributed_training_status.json" "distributed_training_status_backup_%DATE:~-4%%DATE:~3,2%%DATE:~0,2%_%TIME:~0,2%%TIME:~3,2%%TIME:~6,2%.json" >nul 2>&1
)

REM Vérifier la connexion Azure
echo ☁️ Vérification Azure...
python -c "try: from utils.azure_model_storage import AzureModelStorage; storage = AzureModelStorage(); models = storage.list_all_models(); print(f'✅ Azure OK - {len(models)} modèles trouvés'); except Exception as e: print(f'⚠️ Azure: {e}')" 2>nul

echo.
echo 🚀 Démarrage du système distribué...
echo    - Ctrl+C pour arrêt propre et transfert vers laptop
echo    - Le laptop peut démarrer en parallèle en mode slave
echo.

REM Créer nom de fichier de log
for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set DATE_LOG=%%c%%b%%a
for /f "tokens=1-2 delims=: " %%a in ('time /t') do set TIME_LOG=%%a%%b
set LOG_FILE=desktop_master_%DATE_LOG%_%TIME_LOG%.log

echo 📝 Logs sauvegardés dans: %LOG_FILE%
echo.

REM Lancer le système distribué avec capture des logs
python distributed_training_launcher.py master 2>&1 | tee %LOG_FILE%

REM Vérifier le code de sortie
set EXIT_CODE=%ERRORLEVEL%

echo.
echo 📋 RAPPORT FINAL DESKTOP
echo ========================
echo    - Code de sortie: %EXIT_CODE%
echo    - Logs: %LOG_FILE%
echo    - Status final dans: distributed_training_status.json

if exist "night_report.txt" (
    echo.
    echo 📊 Rapport de nuit:
    type night_report.txt
)

echo.
echo 💡 Le laptop peut maintenant prendre le relais automatiquement
echo    Lancez launch_laptop_slave.bat sur le laptop

pause
