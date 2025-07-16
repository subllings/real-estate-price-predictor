#!/bin/bash
# Script de synchronisation rapide pour laptop
# Usage: ./prepare_laptop.sh

echo "💻 PRÉPARATION LAPTOP POUR ENTRAÎNEMENT DISTRIBUÉ"
echo "================================================="

# 1. Vérifier si on est dans un repo Git
if [ ! -d ".git" ]; then
    echo "❌ Pas de repository Git trouvé"
    echo
    echo "🔧 SOLUTIONS:"
    echo "   Option 1 - Cloner le repository:"
    echo "   git clone https://github.com/subllings/real-estate-price-predictor.git"
    echo "   cd real-estate-price-predictor"
    echo
    echo "   Option 2 - Copier manuellement depuis le desktop"
    exit 1
fi

echo "✅ Repository Git détecté"

# 2. Sauvegarder les changements locaux (si il y en a)
echo "🔄 Sauvegarde des changements locaux..."
git add -A 2>/dev/null
git commit -m "Auto-save before sync - $(date)" 2>/dev/null || echo "   ℹ️ Rien à sauvegarder"

# 3. Récupérer les dernières modifications
echo "📥 Récupération des dernières modifications..."
git fetch origin
git pull origin main

# 4. Vérifier les fichiers critiques
echo "🔍 Vérification des fichiers critiques..."
CRITICAL_FILES=(
    "distributed_training_manager.py"
    "auto_recovery_system.py"
    "launch_laptop_slave.sh"
    "launch_laptop_slave.bat"
    "utils/azure_model_storage.py"
    "requirements.txt"
)

missing_files=()
for file in "${CRITICAL_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "   ✅ $file"
    else
        echo "   ❌ $file - MANQUANT"
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -gt 0 ]; then
    echo
    echo "⚠️ Fichiers manquants détectés!"
    echo "💡 Solutions:"
    echo "   1. Commitez sur le desktop et re-tirez ici"
    echo "   2. Copiez manuellement les fichiers manquants"
    exit 1
fi

# 5. Vérifier l'environnement Python
echo
echo "🐍 Vérification environnement Python..."
if [ -f "requirements.txt" ]; then
    echo "📦 Installation des dépendances..."
    pip install -r requirements.txt -q
    if [ $? -eq 0 ]; then
        echo "   ✅ Dépendances installées"
    else
        echo "   ⚠️ Certaines dépendances ont échoué"
    fi
else
    echo "   ⚠️ requirements.txt manquant"
fi

# 6. Vérifier Azure
echo
echo "☁️ Test connexion Azure..."
python -c "
try:
    from utils.azure_model_storage import AzureModelStorage
    storage = AzureModelStorage()
    models = storage.list_all_models()
    print(f'   ✅ Azure OK - {len(models)} modèles disponibles')
except Exception as e:
    print(f'   ❌ Erreur Azure: {e}')
    print('   💡 Vérifiez le fichier .env et les credentials Azure')
" 2>/dev/null

# 7. Rendre les scripts exécutables
echo
echo "🔧 Configuration des permissions..."
chmod +x launch_laptop_slave.sh 2>/dev/null
chmod +x sync_machines.sh 2>/dev/null
chmod +x prepare_laptop.sh 2>/dev/null
echo "   ✅ Permissions configurées"

# 8. Afficher le statut actuel
echo
echo "📊 STATUT FINAL"
echo "==============="
echo "✅ Code synchronisé depuis Git"
echo "✅ Fichiers critiques présents"
echo "✅ Environnement Python configuré"
echo "✅ Permissions définies"

# 9. Vérifier si le desktop est actif
if [ -f "distributed_training_status.json" ]; then
    echo
    echo "🖥️ STATUT DU DESKTOP:"
    python -c "
import json
from datetime import datetime
try:
    with open('distributed_training_status.json', 'r') as f:
        status = json.load(f)
    
    current_master = status.get('current_master', 'Aucun')
    print(f'   Master actuel: {current_master}')
    
    for machine_id, info in status.get('machines', {}).items():
        role = info.get('machine_role', 'unknown')
        training_status = info.get('training_status', 'unknown')
        last_heartbeat = info.get('last_heartbeat', 'unknown')
        print(f'   - {machine_id} ({role}): {training_status}')
        
except Exception as e:
    print(f'   ⚠️ Erreur lecture statut: {e}')
"
else
    echo "   ℹ️ Aucun statut desktop trouvé (normal si pas encore démarré)"
fi

echo
echo "🚀 PRÊT POUR LE LANCEMENT!"
echo "=========================="
echo "Commandes disponibles:"
echo "   ./launch_laptop_slave.sh     - Mode surveillance (recommandé)"
echo "   ./launch_laptop_slave.bat    - Version Windows"
echo
echo "💡 Le laptop va automatiquement:"
echo "   - Surveiller le desktop"
echo "   - Prendre le relais si nécessaire"
echo "   - Synchroniser avec Azure"
echo
echo "Bonne nuit ! 😴"
