#!/usr/bin/env python3
"""
Test rapide du backend pour vérifier les nouveaux diagnostics
"""

import sys
import os
sys.path.append('.')

from utils.cosmosdb_logger import CosmosDbLogger

def test_backend_diagnostics():
    """Test des diagnostics calculés dans le backend"""
    
    logger = CosmosDbLogger()
    
    # Fonction pour calculer le diagnostic de généralisation (copie du backend)
    def calculate_generalization_status(r2_train, r2_test):
        if not r2_train or not r2_test:
            return "Unknown"
        
        r2_gap = r2_train - r2_test
        
        # Logique alignée avec train_test_metrics_logger.py et CatBoost tuner
        if r2_gap < 0:
            return "Possible underfitting"
        elif r2_gap < 0.05:
            return "Excellent generalization"
        elif r2_gap < 0.08:
            return "Good generalization"
        elif r2_gap < 0.12:
            return "Moderate overfitting"
        else:
            return "Strong overfitting"
    
    try:
        # Tester avec les données legacy
        experiments = logger.get_trials_for_model("catboost", limit=5)
        
        print(f"[INFO] Trouvé {len(experiments)} expériences dans le container legacy")
        
        for i, exp in enumerate(experiments):
            print(f"\n[EXPERIMENT {i+1}]")
            print(f"  ID: {exp.get('id', 'N/A')}")
            
            # Extraire les valeurs R² comme dans le backend
            structured_metrics = exp.get("structured_metrics", {})
            r2_train = structured_metrics.get("r2_train") or exp.get("r2_score", 0)
            r2_test = structured_metrics.get("r2_test") or exp.get("r2_test", 0)
            
            # Calculer R² gap et diagnostic
            r2_gap = (r2_train - r2_test) if (r2_train and r2_test) else 0
            generalization_status = calculate_generalization_status(r2_train, r2_test)
            
            print(f"  R² Train: {r2_train}")
            print(f"  R² Test: {r2_test}")
            print(f"  R² Gap: {r2_gap:.6f}")
            print(f"  Nouveau diagnostic: {generalization_status}")
            print(f"  Ancien diagnostic: {exp.get('generalization_status', 'N/A')}")
        
    except Exception as e:
        print(f"[ERROR] Erreur lors du test: {e}")

if __name__ == "__main__":
    test_backend_diagnostics()
