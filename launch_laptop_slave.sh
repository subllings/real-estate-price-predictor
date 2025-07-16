#!/bin/bash
# Script de lancement pour LAPTOP (Slave/Backup)
# Usage: ./launch_laptop_slave.sh

echo "💻 LANCEMENT LAPTOP - MODE SLAVE/BACKUP"
echo "======================================="

# Vérifier les prérequis
if [ ! -f "distributed_training_manager.py" ]; then
    echo "❌ distributed_training_manager.py manquant"
    echo "   💡 Copiez les fichiers depuis le desktop ou clonez le repo"
    exit 1
fi

# Définir variables d'environnement
export MACHINE_ROLE="slave"
export MACHINE_TYPE="laptop"
export TRAINING_PRIORITY="medium"

# Afficher infos système
echo "📊 Informations système:"
echo "   - Machine: $(hostname)"
echo "   - Rôle: SLAVE/BACKUP (Laptop)"
echo "   - Date: $(date)"
echo "   - Dossier: $(pwd)"
echo

# Vérifier si git est synchronisé
if [ -d ".git" ]; then
    echo "🔄 Synchronisation Git..."
    git fetch origin 2>/dev/null
    git pull origin main 2>/dev/null || echo "⚠️ Pas de mise à jour Git"
fi

# Vérifier l'environnement Python
echo "🐍 Vérification environnement Python..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt -q 2>/dev/null || echo "⚠️ Certains packages manquants"
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
    print('   💡 Assurez-vous que les credentials Azure sont configurés')
" 2>/dev/null

# Vérifier le statut du master
echo
echo "🔍 Recherche du master (desktop)..."
if [ -f "distributed_training_status.json" ]; then
    python -c "
import json
from datetime import datetime
try:
    with open('distributed_training_status.json', 'r') as f:
        status = json.load(f)
    
    current_master = status.get('current_master', 'Aucun')
    machines = status.get('machines', {})
    
    print(f'🖥️ Master actuel: {current_master}')
    
    for machine_id, info in machines.items():
        role = info.get('machine_role', 'unknown')
        training_status = info.get('training_status', 'unknown')
        last_heartbeat = info.get('last_heartbeat', 'unknown')
        print(f'   - {machine_id} ({role}): {training_status} - {last_heartbeat}')
        
except Exception as e:
    print('⚠️ Pas de statut disponible - Premier démarrage ou master absent')
"
else
    echo "⚠️ Aucun fichier de statut trouvé"
    echo "   💡 Soit le desktop n'a pas encore démarré, soit c'est un premier lancement"
fi

echo
echo "🤖 MODES DISPONIBLES:"
echo "   1. 🔄 Mode SURVEILLANCE (par défaut)"
echo "      - Surveille le desktop"
echo "      - Prend le relais automatiquement si besoin"
echo "   2. 🚀 Mode FORCE MASTER"
echo "      - Démarre immédiatement en master"
echo "      - Utile si desktop HS"
echo
read -p "Choisir mode (1=surveillance, 2=force): " MODE_CHOICE

if [ "$MODE_CHOICE" = "2" ]; then
    LAUNCH_MODE="master"
    echo "👑 Mode FORCE MASTER sélectionné"
else
    LAUNCH_MODE="slave"
    echo "🔄 Mode SURVEILLANCE sélectionné"
fi

echo
echo "🚀 Démarrage du système distribué..."
echo "   - Ctrl+C pour arrêt"
echo "   - Logs en temps réel"

# Créer fichier de log
LOG_FILE="laptop_slave_$(date +%Y%m%d_%H%M%S).log"

# Fonction pour arrêt propre
cleanup() {
    echo
    echo "🛑 Arrêt laptop demandé"
    echo "   - Logs sauvegardés dans: $LOG_FILE"
    exit 0
}

trap cleanup SIGINT SIGTERM

# Afficher monitoring en temps réel
echo
echo "📊 MONITORING EN TEMPS RÉEL"
echo "============================="

# Lancer le système distribué
python distributed_training_launcher.py "$LAUNCH_MODE" 2>&1 | tee "$LOG_FILE"

# Si on arrive ici, l'entraînement est terminé ou a échoué
EXIT_CODE=$?

echo
echo "📋 RAPPORT FINAL LAPTOP"
echo "======================="
echo "   - Code de sortie: $EXIT_CODE"
echo "   - Logs: $LOG_FILE"
echo "   - Status final dans: distributed_training_status.json"

if [ -f "night_report.txt" ]; then
    echo
    echo "📊 Rapport de nuit:"
    cat night_report.txt
fi

echo
echo "✅ Mission accomplie sur laptop!"
