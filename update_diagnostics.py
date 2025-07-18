#!/usr/bin/env python3
"""
Script pour mettre à jour les diagnostics existants dans ModelMetrics
"""

import sys
sys.path.append('.')

try:
    from utils.cosmosdb_logger import CosmosDbLogger
    
    def analyze_generalization_new(r2_train, r2_test):
        """Nouvelle logique de diagnostic alignée avec train_test_metrics_logger.py"""
        gap = r2_train - r2_test
        if gap < 0:
            return "Possible underfitting"
        elif gap < 0.05:
            return "Excellent generalization"
        elif gap < 0.08:
            return "Good generalization"
        elif gap < 0.12:
            return "Moderate overfitting"
        else:
            return "Strong overfitting"
    
    def update_model_metrics_diagnostics():
        """Mise à jour des diagnostics dans ModelMetrics"""
        
        print("Connexion à Cosmos DB...")
        logger = CosmosDbLogger()
        
        # Récupérer tous les trials du container ModelMetrics
        print("Récupération des données ModelMetrics...")
        trials = logger.get_trials_for_model('CatBoost CV (All Features)', container_name='ModelMetrics')
        
        print(f"Trouvé {len(trials)} expériences à mettre à jour")
        
        # Accéder directement au container ModelMetrics
        model_metrics_container = logger.create_model_metrics_container("ModelMetrics")
        
        updated_count = 0
        for trial in trials:
            try:
                # Calculer le nouveau diagnostic
                r2_train = trial.get('r2_train', 0)
                r2_test = trial.get('r2_test', 0)
                
                if r2_train and r2_test:
                    new_diagnostic = analyze_generalization_new(r2_train, r2_test)
                    old_diagnostic = trial.get('generalization_status', 'Unknown')
                    
                    # Mettre à jour seulement si différent
                    if new_diagnostic != old_diagnostic:
                        print(f"Mise à jour {trial['id']}: '{old_diagnostic}' → '{new_diagnostic}'")
                        
                        # Mettre à jour le record
                        trial['generalization_status'] = new_diagnostic
                        
                        # Sauvegarder
                        model_metrics_container.replace_item(item=trial['id'], body=trial)
                        updated_count += 1
                    else:
                        print(f"Pas de changement pour {trial['id']}: '{old_diagnostic}'")
                        
            except Exception as e:
                print(f"Erreur lors de la mise à jour de {trial.get('id', 'unknown')}: {e}")
        
        print(f"\n✓ {updated_count} expériences mises à jour")
        return updated_count
    
    if __name__ == "__main__":
        update_model_metrics_diagnostics()
        
except ImportError as e:
    print(f"Erreur d'import: {e}")
    print("Assurez-vous que PYTHONPATH est configuré correctement")
except Exception as e:
    print(f"Erreur: {e}")
