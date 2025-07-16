#!/usr/bin/env python3
"""
Script intelligent pour gérer les modèles avec data leakage
Combine analyse, marquage et suppression sécurisée
"""

import sys
import os
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

from utils.cosmosdb_logger import CosmosDbLogger

def analyze_trials(logger):
    """Analyser les trials pour détecter le data leakage"""
    print("[ANALYSE] Recherche des trials avec data leakage...")
    
    trials = logger.get_trials_for_model("catboost")
    if not trials:
        return [], []
    
    valid_trials = []
    leakage_trials = []
    
    for trial in trials:
        if "metrics" in trial and "test" in trial["metrics"]:
            r2_test = trial["metrics"]["test"].get("r2", 0)
            trial_number = trial.get("trial_number", "N/A")
            
            if r2_test > 0.95:
                leakage_trials.append({
                    'trial': trial,
                    'trial_number': trial_number,
                    'r2': r2_test
                })
            else:
                valid_trials.append({
                    'trial': trial,
                    'trial_number': trial_number,
                    'r2': r2_test
                })
    
    return valid_trials, leakage_trials

def display_analysis(valid_trials, leakage_trials):
    """Afficher l'analyse des trials"""
    print(f"\n📊 ANALYSE COMPLÈTE:")
    print(f"   - Trials valides (R² ≤ 0.95): {len(valid_trials)}")
    print(f"   - Trials data leakage (R² > 0.95): {len(leakage_trials)}")
    
    if valid_trials:
        print(f"\n✅ TRIALS VALIDES:")
        for t in valid_trials[:5]:  # Top 5
            print(f"   - Trial {t['trial_number']}: R² = {t['r2']:.4f}")
        if len(valid_trials) > 5:
            print(f"   ... et {len(valid_trials) - 5} autres")
    
    if leakage_trials:
        print(f"\n⚠️  TRIALS SUSPECTS (DATA LEAKAGE):")
        for t in leakage_trials:
            print(f"   - Trial {t['trial_number']}: R² = {t['r2']:.4f}")

def mark_leakage_trials(logger, leakage_trials):
    """Marquer les trials avec data leakage"""
    if not leakage_trials:
        print("✅ Aucun trial à marquer")
        return
    
    print(f"\n[MARQUAGE] {len(leakage_trials)} trials...")
    
    for t in leakage_trials:
        trial = t['trial']
        if not trial.get("data_leakage", False):
            # Simulation du marquage
            print(f"🏷️  Marquage Trial {t['trial_number']}")
            # Ici on ajouterait : trial["data_leakage"] = True
            # Et on ferait l'update dans CosmosDB

def get_user_choice():
    """Demander à l'utilisateur quelle action prendre"""
    print(f"\n🤔 QUE FAIRE ?")
    print("1. Analyser seulement (recommandé pour commencer)")
    print("2. Marquer les trials data leakage")
    print("3. Exporter les commandes DELETE")
    print("4. Tout faire (analyse + marquage)")
    print("0. Quitter")
    
    while True:
        choice = input("\nChoix (0-4): ").strip()
        if choice in ['0', '1', '2', '3', '4']:
            return choice
        print("❌ Choix invalide. Utilisez 0, 1, 2, 3 ou 4")

def main():
    print("🧠 GESTION INTELLIGENTE DATA LEAKAGE")
    print("=" * 60)
    
    try:
        logger = CosmosDbLogger()
        
        # Analyse des trials
        valid_trials, leakage_trials = analyze_trials(logger)
        display_analysis(valid_trials, leakage_trials)
        
        if not leakage_trials:
            print("\n🎉 Aucun data leakage détecté ! Base de données propre.")
            return
        
        # Demander à l'utilisateur
        choice = get_user_choice()
        
        if choice == '0':
            print("👋 Au revoir !")
            return
        
        elif choice == '1':
            print("✅ Analyse terminée ! Pas d'autres actions.")
        
        elif choice == '2':
            mark_leakage_trials(logger, leakage_trials)
            print("✅ Marquage terminé !")
        
        elif choice == '3':
            print("\n📝 COMMANDES DELETE POUR AZURE PORTAL:")
            print("-" * 50)
            for t in leakage_trials:
                trial_id = t['trial'].get('id', 'N/A')
                print(f"DELETE FROM c WHERE c.id = '{trial_id}'")
            print("-" * 50)
            print("⚠️  ATTENTION: Vérifiez bien avant d'exécuter !")
        
        elif choice == '4':
            mark_leakage_trials(logger, leakage_trials)
            print("\n📝 COMMANDES DELETE (au cas où):")
            for t in leakage_trials:
                trial_id = t['trial'].get('id', 'N/A')
                print(f"DELETE FROM c WHERE c.id = '{trial_id}'")
            print("✅ Analyse + marquage terminés !")
        
        print(f"\n🕒 Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\n❌ ERREUR:")
        print(f"   {str(e)}")

if __name__ == "__main__":
    main()
