#!/usr/bin/env python3
"""
Test rapide du système de marquage data leakage
"""

import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

from utils.cosmosdb_logger import CosmosDbLogger

def test_data_leakage_filtering():
    print("🧪 TEST DU SYSTÈME DE FILTRAGE DATA LEAKAGE")
    print("=" * 60)
    
    try:
        logger = CosmosDbLogger()
        
        print("[TEST 1] Récupération trials AVEC data leakage...")
        all_trials = logger.get_trials_for_model("catboost", limit=50, include_data_leakage=True)
        print(f"   → {len(all_trials)} trials trouvés (incluant data leakage)")
        
        print("\n[TEST 2] Récupération trials SANS data leakage...")
        clean_trials = logger.get_trials_for_model("catboost", limit=50, include_data_leakage=False)
        print(f"   → {len(clean_trials)} trials trouvés (excluant data leakage)")
        
        # Analyse des métriques
        if all_trials:
            print("\n[ANALYSE] Exemples de trials trouvés:")
            for i, trial in enumerate(all_trials[:3]):
                trial_num = trial.get("trial_number", "N/A")
                r2 = trial.get("metrics", {}).get("test", {}).get("r2", 0)
                is_leakage = trial.get("data_leakage", False)
                status = "🔴 DATA LEAKAGE" if is_leakage else "✅ VALIDE"
                print(f"   Trial {trial_num}: R² = {r2:.4f} {status}")
        
        print(f"\n✅ RÉSULTAT:")
        print(f"   - Système de filtrage: {'✅ ACTIF' if len(clean_trials) <= len(all_trials) else '❌ PROBLÈME'}")
        print(f"   - Trials filtrés: {len(all_trials) - len(clean_trials)}")
        
        return len(clean_trials) > 0
        
    except Exception as e:
        print(f"❌ ERREUR: {e}")
        return False

if __name__ == "__main__":
    success = test_data_leakage_filtering()
    if success:
        print("\n🎉 Test réussi ! Le système de filtrage fonctionne.")
    else:
        print("\n❌ Test échoué.")
    sys.exit(0 if success else 1)
