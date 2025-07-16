#!/usr/bin/env python3
"""
Script pour marquer automatiquement les trials avec data leakage
et vérifier que les futurs entraînements les ignorent
"""

import sys
import os
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

from utils.cosmosdb_logger import CosmosDbLogger

def main():
    print("🏷️  MARQUAGE AUTOMATIQUE DATA LEAKAGE")
    print("=" * 60)
    
    try:
        logger = CosmosDbLogger()
        
        print("[ÉTAPE 1] Analyse des trials existants...")
        
        # Obtenir un résumé avant marquage
        print("\n📊 AVANT MARQUAGE:")
        summary_before = logger.get_data_leakage_summary()
        
        if summary_before.get("data_leakage_trials", 0) > 0:
            print(f"ℹ️  {summary_before['data_leakage_trials']} trials déjà marqués")
        
        print("\n[ÉTAPE 2] Marquage des nouveaux trials suspects...")
        
        # Marquer les trials avec R² > 0.95
        marked_count = logger.mark_trials_with_data_leakage(r2_threshold=0.95)
        
        print("\n📊 APRÈS MARQUAGE:")
        summary_after = logger.get_data_leakage_summary()
        
        print(f"\n✅ RÉSUMÉ:")
        print(f"   - Nouveaux trials marqués: {marked_count}")
        print(f"   - Total trials marqués: {summary_after.get('data_leakage_trials', 0)}")
        print(f"   - Trials valides disponibles: {summary_after.get('valid_trials', 0)}")
        
        if summary_after.get('valid_trials', 0) > 0:
            print(f"\n🎯 LES FUTURS ENTRAÎNEMENTS:")
            print(f"   ✅ Utiliseront uniquement {summary_after['valid_trials']} trials valides")
            print(f"   ⏭️  Ignoreront automatiquement {summary_after['data_leakage_trials']} trials corrompus")
        else:
            print(f"\n⚠️  ATTENTION: Aucun trial valide trouvé!")
            print(f"   → Il faudra relancer un entraînement complet")
        
        print(f"\n🔍 VALIDATION:")
        print(f"   - Les scripts CatBoostTuner ignoreront automatiquement les trials marqués")
        print(f"   - Seuls les trials avec R² ≤ 0.95 seront utilisés comme référence")
        
        print(f"\n🕒 Fin du marquage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ ERREUR pendant le marquage:")
        print(f"   {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
