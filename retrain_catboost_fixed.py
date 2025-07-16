#!/usr/bin/env python3
"""
Script pour réentraîner CatBoost avec la correction du data leakage
Utilise les vraies métriques de cross-validation
"""

import sys
import os
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(project_root)

from utils.data_loader import DataLoader
from agents.tuner_agent.catboost_tuner import CatBoostTuner
from utils.constants import ML_READY_DATA_FILE

def main():
    print("🚀 RÉENTRAÎNEMENT CATBOOST - CORRECTION DATA LEAKAGE")
    print("=" * 60)
    print(f"📅 Démarrage: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Charger les données
        print("\n[ÉTAPE 1] Chargement des données...")
        data_loader = DataLoader()
        X, y = data_loader.load_data(ML_READY_DATA_FILE)
        
        print(f"✅ Données chargées: {X.shape[0]} lignes, {X.shape[1]} features")
        print(f"📊 Prix min: {y.min():,.0f}€, Prix max: {y.max():,.0f}€")
        print(f"📊 Prix moyen: {y.mean():,.0f}€, Prix médian: {y.median():,.0f}€")
        
        # Configuration de l'entraînement
        print("\n[ÉTAPE 2] Configuration de l'entraînement...")
        config = {
            "n_trials": 50,  # Plus de trials pour une nuit complète
            "n_splits": 5,   # Cross-validation 5-fold
            "early_stopping_rounds": 50,
            "random_state": 42
        }
        
        print(f"🔧 Configuration:")
        for key, value in config.items():
            print(f"   - {key}: {value}")
        
        # Créer et lancer l'optimisation
        print("\n[ÉTAPE 3] Démarrage de l'optimisation Optuna...")
        print("⚠️  Ceci peut prendre plusieurs heures...")
        
        tuner = CatBoostTuner(
            X=X,
            y=y,
            n_trials=config["n_trials"],
            n_splits=config["n_splits"],
            early_stopping_rounds=config["early_stopping_rounds"],
            random_state=config["random_state"]
        )
        
        # Lancer l'étude
        best_trial = tuner.run_study()
        
        # Récupérer les métriques finales
        final_metrics = tuner.get_final_metrics()
        
        print("\n" + "=" * 60)
        print("🎯 RÉSULTATS FINAUX (VRAIES MÉTRIQUES CV)")
        print("=" * 60)
        print(f"🏆 Meilleur trial: {best_trial.number}")
        print(f"📈 R² Score (CV): {final_metrics['r2_test']:.4f}")
        print(f"💰 MAE (CV): {final_metrics['mae_test']:,.0f}€")
        print(f"📊 RMSE (CV): {final_metrics['rmse_test']:,.0f}€")
        
        # Évaluation de la performance
        r2_score = final_metrics['r2_test']
        if r2_score >= 0.85:
            print("🌟 EXCELLENT modèle!")
        elif r2_score >= 0.80:
            print("✅ TRÈS BON modèle!")
        elif r2_score >= 0.75:
            print("👍 BON modèle!")
        elif r2_score >= 0.70:
            print("⚠️  Modèle CORRECT mais peut être amélioré")
        else:
            print("❌ Modèle FAIBLE - Nécessite plus d'optimisation")
        
        print(f"\n🕒 Fin de l'entraînement: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("✅ Modèle sauvegardé et métriques loggées!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR pendant l'entraînement:")
        print(f"   {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
