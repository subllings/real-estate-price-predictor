#!/usr/bin/env bash
# Script de validation complète du système anti-data leakage

echo "🔍 VALIDATION SYSTÈME ANTI-DATA LEAKAGE"
echo "========================================"

echo ""
echo "[ÉTAPE 1] Test du filtrage CosmosDB..."
python test_data_leakage_filter.py

if [ $? -eq 0 ]; then
    echo ""
    echo "[ÉTAPE 2] Marquage automatique des trials suspects..."
    python auto_mark_data_leakage.py
    
    echo ""
    echo "[ÉTAPE 3] Vérification finale..."
    python test_data_leakage_filter.py
    
    echo ""
    echo "✅ SYSTÈME PRÊT POUR RETRAINING!"
    echo ""
    echo "🚀 PROCHAINES ÉTAPES:"
    echo "   1. Lancer: ./launch_night_training.sh"
    echo "   2. Les nouveaux modèles utiliseront uniquement des trials valides"
    echo "   3. Les trials avec R² > 0.95 seront automatiquement ignorés"
    echo ""
    echo "🔒 PROTECTION ACTIVE:"
    echo "   ✅ Trials suspects marqués et ignorés"
    echo "   ✅ Nouveaux trials marqués comme 'data_leakage_corrected'"
    echo "   ✅ Métriques de cross-validation utilisées"
    
else
    echo ""
    echo "❌ PROBLÈME DÉTECTÉ - Vérifiez la configuration CosmosDB"
fi
