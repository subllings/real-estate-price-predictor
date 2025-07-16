#!/bin/bash
# Script de lancement pour DESKTOP (Master principal)
# Usage: ./launch_desktop_master.sh

echo "🖥️ LANCEMENT DESKTOP - MODE MASTER"
echo "=================================="

# Vérifier les prérequis
if [ ! -f "distributed_training_manager.py" ]; then
    echo "❌ distributed_training_manager.py manquant"
    exit 1
fi

if [ ! -f "auto_recovery_system.py" ]; then
    echo "❌ auto_recovery_system.py manquant"
    exit 1
fi

# Définir variables d'environnement
export MACHINE_ROLE="master"
export MACHINE_TYPE="desktop"
export TRAINING_PRIORITY="high"

# Afficher infos système
echo "📊 Informations système:"
echo "   - Machine: $(hostname)"
echo "   - Rôle: MASTER (Desktop)"
echo "   - Date: $(date)"
echo "   - Dossier: $(pwd)"
echo

# Nettoyer anciens fichiers de statut si nécessaire
if [ -f "distributed_training_status.json" ]; then
    echo "🧹 Nettoyage ancien statut..."
    # Garder une sauvegarde
    cp distributed_training_status.json "distributed_training_status_backup_$(date +%Y%m%d_%H%M%S).json"
fi

# Vérifier la connexion Azure
echo "☁️ Vérification Azure..."
python -c "
try:
    from utils.azure_model_storage import AzureModelStorage
    storage = AzureModelStorage()
    models = storage.list_all_models()
    print(f'✅ Azure OK - {len(models)} modèles trouvés')
except Exception as e:
    print(f'⚠️ Azure: {e}')
" 2>/dev/null

echo
echo "🚀 Démarrage du système distribué..."
echo "   - Ctrl+C pour arrêt propre et transfert vers laptop"
echo "   - Le laptop peut démarrer en parallèle en mode slave"
echo

# Créer fichier de log
LOG_FILE="desktop_master_$(date +%Y%m%d_%H%M%S).log"

# Fonction pour arrêt propre
cleanup() {
    echo
    echo "🛑 Arrêt demandé - Transfert vers laptop..."
    echo "   - Le laptop va reprendre automatiquement"
    echo "   - Logs sauvegardés dans: $LOG_FILE"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Lancer le système distribué
python distributed_training_launcher.py master 2>&1 | tee "$LOG_FILE"

# Si on arrive ici, l'entraînement est terminé ou a échoué
EXIT_CODE=$?

echo
echo "📋 RAPPORT FINAL DESKTOP"
echo "========================"
echo "   - Code de sortie: $EXIT_CODE"
echo "   - Logs: $LOG_FILE"
echo "   - Status final dans: distributed_training_status.json"

if [ -f "night_report.txt" ]; then
    echo
    echo "📊 Rapport de nuit:"
    cat night_report.txt
fi

echo
echo "💡 Le laptop peut maintenant prendre le relais automatiquement"
echo "   Lancez ./launch_laptop_slave.sh sur le laptop"
