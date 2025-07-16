#!/bin/bash
# Script de lancement simple avec le système de récupération automatique

echo "🌙 ENTRAÎNEMENT NOCTURNE AVEC AUTO-RECOVERY"
echo "============================================"
echo "📅 Début: $(date)"
echo ""

# Aller dans le répertoire du projet
cd "$(dirname "$0")"

# Lancer le système de récupération automatique
echo "🚀 Lancement du système de surveillance automatique..."
python auto_recovery_system.py

# Récupérer le code de sortie
EXIT_CODE=$?

echo ""
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ENTRAÎNEMENT NOCTURNE RÉUSSI!"
    echo "📊 Vérifiez night_report.txt pour les détails"
else
    echo "❌ ENTRAÎNEMENT NOCTURNE ÉCHOUÉ!"
    echo "📝 Vérifiez auto_recovery_system.log pour les détails"
fi

echo "📅 Fin: $(date)"
echo ""

# Afficher le rapport rapide s'il existe
if [ -f "night_report.txt" ]; then
    echo "📋 RAPPORT RAPIDE:"
    cat night_report.txt
fi

exit $EXIT_CODE
