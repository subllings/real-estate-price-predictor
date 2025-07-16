@echo off
REM Script de lancement pour Windows avec système de récupération automatique

echo 🌙 ENTRAÎNEMENT NOCTURNE AVEC AUTO-RECOVERY
echo ============================================
echo 📅 Début: %date% %time%
echo.

REM Aller dans le répertoire du projet
cd /d "%~dp0"

REM Lancer le système de récupération automatique
echo 🚀 Lancement du système de surveillance automatique...
python auto_recovery_system.py

REM Récupérer le code de sortie
set EXIT_CODE=%ERRORLEVEL%

echo.
echo ============================================
if %EXIT_CODE%==0 (
    echo ✅ ENTRAÎNEMENT NOCTURNE RÉUSSI!
    echo 📊 Vérifiez night_report.txt pour les détails
) else (
    echo ❌ ENTRAÎNEMENT NOCTURNE ÉCHOUÉ!
    echo 📝 Vérifiez auto_recovery_system.log pour les détails
)

echo 📅 Fin: %date% %time%
echo.

REM Afficher le rapport rapide s'il existe
if exist "night_report.txt" (
    echo 📋 RAPPORT RAPIDE:
    type "night_report.txt"
)

pause
exit /b %EXIT_CODE%
