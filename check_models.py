#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils.cosmosdb_logger import CosmosDbLogger

def check_catboost_models():
    print("=== VERIFICATION DES MODELES CATBOOST ===\n")
    
    try:
        logger = CosmosDbLogger()
        
        # Chercher les essais optuna pour catboost
        trials = logger.get_trials_for_model('catboost', limit=10)
        print(f"Modeles CatBoost trouves: {len(trials)}")
        
        if trials:
            trial = trials[0]  # Le plus recent
            print("\n=== DERNIER MODELE CATBOOST ENTRAINE ===")
            print(f"Numero d'essai: {trial.get('trial_number')}")
            print(f"Fichier: {trial.get('model_file')}")
            print(f"Date: {trial.get('timestamp')}")
            
            metrics = trial.get('metrics', {})
            test = metrics.get('test', {})
            print("\nMETRIQUES:")
            print(f"  R2 Score: {test.get('r2', 0):.4f}")
            print(f"  MAE: {test.get('mae', 0):,.0f} euros")
            print(f"  RMSE: {test.get('rmse', 0):,.0f} euros")
            
            is_perfect = trial.get('is_perfect', False)
            status = "PARFAIT (R2 >= 0.90)" if is_perfect else "STANDARD"
            print(f"\nStatut: {status}")
            
            # Afficher quelques parametres cles
            params = trial.get('params', {})
            if params:
                print("\nPARAMETRES CLES:")
                key_params = ['learning_rate', 'depth', 'iterations', 'l2_leaf_reg']
                for param in key_params:
                    if param in params:
                        print(f"  {param}: {params[param]}")
            
            return True
        else:
            print("Aucun modele trouve dans CosmosDB")
            return False
            
    except Exception as e:
        print(f"Erreur: {e}")
        return False

if __name__ == "__main__":
    check_catboost_models()
