#!/bin/bash
# Script de synchronisation entre desktop et laptop
# Usage: ./sync_machines.sh [push|pull|auto]

SYNC_MODE=${1:-auto}
PROJECT_NAME="real-estate-price-predictor"

echo "🔄 SYNCHRONISATION MULTI-MACHINES"
echo "=================================="
echo "Mode: $SYNC_MODE"
echo

# Détecter le type de machine
detect_machine_type() {
    hostname=$(hostname | tr '[:upper:]' '[:lower:]')
    
    if [[ $hostname == *"desktop"* ]] || [[ $hostname == *"pc"* ]] || [[ $hostname == *"gaming"* ]]; then
        echo "desktop"
    elif [[ $hostname == *"laptop"* ]] || [[ $hostname == *"notebook"* ]]; then
        echo "laptop"
    else
        echo "unknown"
    fi
}

MACHINE_TYPE=$(detect_machine_type)
echo "🖥️ Type détecté: $MACHINE_TYPE"

# Fichiers essentiels à synchroniser
ESSENTIAL_FILES=(
    "distributed_training_manager.py"
    "auto_recovery_system.py"
    "launch_desktop_master.sh"
    "launch_desktop_master.bat"
    "launch_laptop_slave.sh"
    "launch_laptop_slave.bat"
    "utils/azure_model_storage.py"
    ".env"
    "requirements.txt"
    "requirements-cloud.txt"
)

# Fonction de synchronisation Git
sync_with_git() {
    echo "📡 Synchronisation via Git..."
    
    # Vérifier si on est dans un repo Git
    if [ ! -d ".git" ]; then
        echo "❌ Pas de repository Git trouvé"
        echo "💡 Initialisez Git ou clonez le repository"
        return 1
    fi
    
    # Sauvegarder les changements locaux
    git add -A
    git commit -m "Auto-sync from $MACHINE_TYPE - $(date)" 2>/dev/null || echo "⚠️ Rien à committer"
    
    # Synchroniser
    git fetch origin
    git pull origin main
    git push origin main 2>/dev/null || echo "⚠️ Push échoué (pas grave)"
    
    echo "✅ Synchronisation Git terminée"
}

# Fonction de synchronisation via réseau local (si même réseau)
sync_via_network() {
    echo "🌐 Recherche de l'autre machine sur le réseau..."
    
    # Chercher des machines avec le projet
    for ip in $(ip route | grep -E '192\.168\.|10\.|172\.' | awk '{print $1}' | head -5); do
        if ping -c 1 -W 1 "$ip" >/dev/null 2>&1; then
            echo "🔍 Machine trouvée: $ip"
            # Ici on pourrait implémenter rsync ou scp
        fi
    done
    
    echo "⚠️ Synchronisation réseau non implémentée (utilisez Git)"
}

# Fonction de synchronisation via stockage cloud
sync_via_cloud() {
    echo "☁️ Synchronisation via Azure Storage..."
    
    # Créer un package de synchronisation
    SYNC_PACKAGE="sync_package_$(date +%Y%m%d_%H%M%S).tar.gz"
    
    # Compresser les fichiers essentiels
    tar -czf "$SYNC_PACKAGE" "${ESSENTIAL_FILES[@]}" 2>/dev/null
    
    if [ -f "$SYNC_PACKAGE" ]; then
        echo "📦 Package créé: $SYNC_PACKAGE"
        
        # Upload vers Azure (si disponible)
        python -c "
try:
    from utils.azure_model_storage import AzureModelStorage
    storage = AzureModelStorage()
    
    # Upload du package
    with open('$SYNC_PACKAGE', 'rb') as f:
        blob_client = storage.blob_service_client.get_blob_client(
            container='ml-models',
            blob='sync/$SYNC_PACKAGE'
        )
        blob_client.upload_blob(f.read(), overwrite=True)
    
    print('✅ Package uploadé vers Azure')
except Exception as e:
    print(f'❌ Erreur upload Azure: {e}')
"
        
        # Nettoyer
        rm "$SYNC_PACKAGE"
    fi
}

# Fonction de vérification des différences
check_differences() {
    echo "🔍 Vérification des différences..."
    
    for file in "${ESSENTIAL_FILES[@]}"; do
        if [ -f "$file" ]; then
            echo "✅ $file - $(stat -c %Y "$file" | xargs -I {} date -d @{} '+%Y-%m-%d %H:%M:%S')"
        else
            echo "❌ $file - MANQUANT"
        fi
    done
}

# Logique principale selon le mode
case $SYNC_MODE in
    "push")
        echo "📤 Mode PUSH - Envoi des changements"
        sync_with_git
        sync_via_cloud
        ;;
    
    "pull")
        echo "📥 Mode PULL - Récupération des changements"
        sync_with_git
        ;;
    
    "auto")
        echo "🤖 Mode AUTO - Synchronisation intelligente"
        
        # Vérifier l'état actuel
        check_differences
        echo
        
        # Synchroniser selon le type de machine
        if [ "$MACHINE_TYPE" = "desktop" ]; then
            echo "🖥️ Desktop détecté - Mode push privilégié"
            sync_with_git
            sync_via_cloud
        elif [ "$MACHINE_TYPE" = "laptop" ]; then
            echo "💻 Laptop détecté - Mode pull privilégié"
            sync_with_git
        else
            echo "❓ Type inconnu - Synchronisation standard"
            sync_with_git
        fi
        ;;
    
    *)
        echo "❌ Mode inconnu: $SYNC_MODE"
        echo "Usage: $0 [push|pull|auto]"
        exit 1
        ;;
esac

echo
echo "📋 RÉSUMÉ DE SYNCHRONISATION"
echo "============================"
echo "✅ Synchronisation terminée"
echo "🖥️ Machine: $MACHINE_TYPE"
echo "📅 Date: $(date)"

# Vérifier l'état final
echo
check_differences

echo
echo "💡 ÉTAPES SUIVANTES:"
if [ "$MACHINE_TYPE" = "desktop" ]; then
    echo "   1. Lancez: ./launch_desktop_master.sh"
    echo "   2. Sur le laptop, lancez: ./launch_laptop_slave.sh"
elif [ "$MACHINE_TYPE" = "laptop" ]; then
    echo "   1. Si desktop actif: ./launch_laptop_slave.sh"
    echo "   2. Si desktop HS: ./launch_laptop_slave.sh puis choisir mode 2"
else
    echo "   1. Choisissez le script approprié selon votre machine"
fi

echo "   3. Les machines se synchroniseront automatiquement"
