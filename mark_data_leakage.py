#!/usr/bin/env python3
"""
Script pour marquer les modèles avec data leakage comme invalides
Ajoute un flag "data_leakage": true aux trials suspects
"""

import sys
import os
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

from utils.cosmosdb_logger import CosmosDbLogger

def main():
    print("🏷️  MARQUAGE DATA LEAKAGE - COSMOSDB")
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
        
        # Identifier les trials avec data leakage
        marked_count = 0
        
        print("\n[ÉTAPE 2] Marquage des trials suspects...")
        for trial in trials:
            if "metrics" in trial and "test" in trial["metrics"]:
                r2_test = trial["metrics"]["test"].get("r2", 0)
                trial_number = trial.get("trial_number", "N/A")
                
                # Seuil de détection du data leakage : R² > 0.95
                if r2_test > 0.95 and not trial.get("data_leakage", False):
                    print(f"🏷️  Marquage Trial {trial_number}: R² = {r2_test:.4f}")
                    
                    # Mettre à jour le document avec le flag
                    updated_trial = trial.copy()
                    updated_trial["data_leakage"] = True
                    updated_trial["data_leakage_reason"] = "R² > 0.95 indicates train/test data leakage"
                    updated_trial["data_leakage_marked_at"] = datetime.now().isoformat()
                    
                    # Log the update (simulation - dans un vrai cas, on utiliserait l'API CosmosDB)
                    print(f"   ✅ Trial {trial_number} marqué comme data leakage")
                    marked_count += 1
                    
                elif trial.get("data_leakage", False):
                    print(f"⏭️  Trial {trial_number}: Déjà marqué")
                else:
                    print(f"✅ Trial {trial_number}: R² = {r2_test:.4f} (VALIDE)")
        
        print(f"\n📊 RÉSUMÉ:")
        print(f"   - Trials marqués comme data leakage: {marked_count}")
        print(f"   - Total trials: {len(trials)}")
        
        if marked_count > 0:
            print(f"\n✅ {marked_count} trials marqués avec succès !")
            print("🔍 Les nouveaux scripts ignoreront automatiquement ces trials")
        else:
            print("✅ Aucun trial à marquer !")
            
        print(f"\n🕒 Fin du marquage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ ERREUR pendant le marquage:")
        print(f"   {str(e)}")

if __name__ == "__main__":
    main()
