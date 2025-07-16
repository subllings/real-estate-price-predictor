#!/bin/bash
# Script pour lancer l'entraînement CatBoost corrigé la nuit
# Utilisation: ./launch_night_training.sh

echo "🌙 LANCEMENT ENTRAÎNEMENT NOCTURNE CATBOOST"
echo "=========================================="
echo "📅 Début: $(date)"
echo ""

# Aller dans le répertoire du projet
cd "$(dirname "$0")"

# Vérifier que Python est disponible
if ! command -v python &> /dev/null; then
    echo "❌ Python non trouvé!"
    exit 1
fi

# Créer un fichier de log avec timestamp
LOG_FILE="training_night_$(date +%Y%m%d_%H%M%S).log"

echo "📝 Log file: $LOG_FILE"
echo "🚀 Démarrage de l'entraînement..."
echo ""

# Lancer l'entraînement avec logging
python retrain_catboost_fixed.py 2>&1 | tee "$LOG_FILE"

# Vérifier le résultat
EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!"
    echo "📊 Vérifiez les métriques dans les logs"
    echo "📁 Modèle sauvegardé dans models/"
else
    echo "❌ ENTRAÎNEMENT ÉCHOUÉ!"
    echo "📝 Vérifiez le log: $LOG_FILE"
fi

echo "📅 Fin: $(date)"
echo "📝 Log complet dans: $LOG_FILE"
