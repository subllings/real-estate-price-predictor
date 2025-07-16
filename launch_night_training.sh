#!/bin/bash
# Script de surveillance et auto-correction pour l'entraînement nocturne
# Surveille l'entraînement et corrige automatiquement les erreurs communes

LOG_FILE="night_training_monitor.log"
ERROR_LOG="night_training_errors.log"
TRAINING_PID=""
MAX_RETRIES=3
RETRY_COUNT=0

# Fonction de logging
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: $1" | tee -a "$ERROR_LOG"
}

# Vérification de l'environnement
check_environment() {
    log_message "� Vérification de l'environnement..."
    
    # Vérifier Azure Storage
    if ! python -c "from utils.azure_model_storage import AzureModelStorage; AzureModelStorage()" 2>/dev/null; then
        log_error "Azure Storage non configuré"
        return 1
    fi
    
    # Vérifier les données
    if [ ! -d "data/ml_ready" ]; then
        log_error "Dossier data/ml_ready manquant"
        return 1
    fi
    
    log_message "✅ Environnement OK"
    return 0
}

# Détecter les erreurs communes et les corriger
auto_fix_errors() {
    local error_output="$1"
    
    # Erreur mémoire
    if echo "$error_output" | grep -i "memory\|ram\|out of memory" > /dev/null; then
        log_message "🔧 Correction: Erreur mémoire détectée"
        export REDUCE_MEMORY=true
        return 0
    fi
    
    # Erreur Azure
    if echo "$error_output" | grep -i "azure\|blob\|connection" > /dev/null; then
        log_message "🔧 Correction: Erreur Azure - Mode local"
        export DISABLE_AZURE_UPLOAD=true
        return 0
    fi
    
    # Erreur de données
    if echo "$error_output" | grep -i "data\|file not found" > /dev/null; then
        log_message "🔧 Correction: Régénération données"
        python clean_data_leakage.py 2>/dev/null
        return 0
    fi
    
    return 1
}

# Lancer l'entraînement avec surveillance
start_training() {
    log_message "🚀 Lancement de l'entraînement nocturne..."
    
    # Choisir le script selon disponibilité
    if [ -f "run_loop_tuner_agent.sh" ]; then
        TRAINING_SCRIPT="bash run_loop_tuner_agent.sh"
    elif [ -f "retrain_catboost_fixed.py" ]; then
        TRAINING_SCRIPT="python retrain_catboost_fixed.py"
    else
        log_error "Aucun script d'entraînement trouvé"
        return 1
    fi
    
    log_message "📜 Script utilisé: $TRAINING_SCRIPT"
    
    # Lancer avec timeout de 8h
    timeout 28800 $TRAINING_SCRIPT > training_output.log 2>&1 &
    TRAINING_PID=$!
    
    log_message "🔄 Entraînement lancé (PID: $TRAINING_PID)"
    return 0
}

# Surveiller l'entraînement
monitor_training() {
    log_message "👁️ Surveillance de l'entraînement..."
    
    while kill -0 "$TRAINING_PID" 2>/dev/null; do
        # Vérifier les erreurs dans la sortie
        if [ -f "training_output.log" ]; then
            local recent_errors=$(tail -50 training_output.log | grep -i "error\|exception")
            if [ ! -z "$recent_errors" ]; then
                log_error "Erreurs détectées"
                echo "$recent_errors" >> "$ERROR_LOG"
                
                # Tentative de correction
                if auto_fix_errors "$recent_errors"; then
                    log_message "🔧 Correction appliquée - Redémarrage"
                    kill "$TRAINING_PID" 2>/dev/null
                    sleep 5
                    return 1  # Signal pour retry
                fi
            fi
        fi
        
        sleep 60
    done
    
    wait "$TRAINING_PID"
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        log_message "✅ Entraînement terminé avec succès"
        return 0
    else
        log_error "Entraînement échoué (code: $exit_code)"
        return 1
    fi
}

# Script principal avec retry
main() {
    log_message "🌙 === DÉBUT ENTRAÎNEMENT NOCTURNE SURVEILLÉ ==="
    
    # Vérifications préliminaires
    if ! check_environment; then
        log_error "Environnement non conforme - Arrêt"
        exit 1
    fi
    
    # Boucle de retry
    while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
        log_message "🔄 Tentative $((RETRY_COUNT + 1))/$MAX_RETRIES"
        
        if start_training && monitor_training; then
            log_message "🎉 Entraînement nocturne réussi!"
            
            # Générer rapport de succès
            echo "✅ SUCCÈS - $(date)" > night_report.txt
            python -c "
try:
    from utils.azure_model_storage import AzureModelStorage
    storage = AzureModelStorage()
    models = storage.list_all_models()
    print(f'📊 {len(models)} modèles sur Azure')
    if models: print(f'🏆 Meilleur R²: {models[0].get(\"r2_test\", \"N/A\")}')
except: pass
" >> night_report.txt
            exit 0
        fi
        
        # Échec - retry
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            log_message "⚠️ Échec - Récupération et retry..."
            sleep 30
        fi
    done
    
    # Échec final
    log_error "❌ Entraînement échoué après $MAX_RETRIES tentatives"
    echo "❌ ÉCHEC après $MAX_RETRIES tentatives - $(date)" > night_report.txt
    exit 1
}

# Gérer les signaux pour cleanup
trap 'log_message "🛑 Arrêt demandé"; kill $TRAINING_PID 2>/dev/null; exit 130' INT TERM

# Lancer le monitoring
main "$@"
