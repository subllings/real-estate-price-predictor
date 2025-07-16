#!/usr/bin/env python3
"""
Script pour nettoyer les modèles avec data leakage de CosmosDB
Supprime les trials avec des métriques irréalistes (R² > 0.95)
"""

import sys
import os
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

from utils.cosmosdb_logger import CosmosDbLogger

def main():
    print("🧹 NETTOYAGE COSMOSDB - SUPPRESSION DATA LEAKAGE")
    print("=" * 60)
    
    try:
        logger = CosmosDbLogger()
        
        # Récupérer tous les trials CatBoost
        print("[ÉTAPE 1] Récupération des trials CatBoost...")
        trials = logger.get_trials_for_model("catboost")
        print(f"✅ {len(trials)} trials trouvés")
        
        if not trials:
            print("ℹ️  Aucun trial trouvé dans CosmosDB")
            return
        
        # Analyser et identifier les trials avec data leakage
        suspicious_trials = []
        valid_trials = []
        
        print("\n[ÉTAPE 2] Analyse des métriques...")
        for trial in trials:
            if "metrics" in trial and "test" in trial["metrics"]:
                r2_test = trial["metrics"]["test"].get("r2", 0)
                trial_number = trial.get("trial_number", "N/A")
                
                # Seuil de détection du data leakage : R² > 0.95
                if r2_test > 0.95:
                    suspicious_trials.append({
                        "id": trial["id"],
                        "trial_number": trial_number,
                        "r2": r2_test,
                        "timestamp": trial.get("timestamp", "N/A")
                    })
                    print(f"🚨 Trial {trial_number}: R² = {r2_test:.4f} (SUSPECT)")
                else:
                    valid_trials.append({
                        "trial_number": trial_number,
                        "r2": r2_test
                    })
                    print(f"✅ Trial {trial_number}: R² = {r2_test:.4f} (VALIDE)")
        
        print(f"\n📊 RÉSUMÉ:")
        print(f"   - Trials suspects (data leakage): {len(suspicious_trials)}")
        print(f"   - Trials valides: {len(valid_trials)}")
        
        # Demander confirmation si trials suspects trouvés
        if suspicious_trials:
            print(f"\n⚠️  TRIALS SUSPECTS DÉTECTÉS:")
            for trial in suspicious_trials:
                print(f"   - Trial {trial['trial_number']}: R² = {trial['r2']:.4f}")
            
            print(f"\n❓ Voulez-vous supprimer ces {len(suspicious_trials)} trials suspects ?")
            print("   Tapez 'SUPPRIMER' pour confirmer, ou n'importe quoi d'autre pour annuler:")
            
            # Pour ce script, on va simplement afficher les IDs à supprimer
            print("\n🔧 COMMANDES DE SUPPRESSION:")
            print("Exécutez ces commandes dans Azure Portal ou via script:")
            
            for trial in suspicious_trials:
                print(f"DELETE FROM c WHERE c.id = '{trial['id']}'")
            
            print(f"\n📝 LOG: {len(suspicious_trials)} trials marqués pour suppression")
            
        else:
            print("✅ Aucun trial suspect trouvé !")
            
        print(f"\n🕒 Fin du nettoyage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ ERREUR pendant le nettoyage:")
        print(f"   {str(e)}")

if __name__ == "__main__":
    main()
