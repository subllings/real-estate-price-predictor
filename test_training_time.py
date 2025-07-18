#!/usr/bin/env python3
"""
Test script pour vérifier le training_time dans CatBoost Tuner
"""

import sys
import os
sys.path.append('.')

# Import du tuner modifié
from agents.tuner_agent.catboost_tuner import CatBoostTuner
import pandas as pd
import numpy as np

def test_training_time():
    """Test simple pour vérifier la mesure du temps d'entraînement"""
    
    # Générer des données de test
    print("[TEST] Génération des données de test...")
    n_samples = 1000
    n_features = 20
    
    # Créer des données factices
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = np.random.randn(n_samples) * 100000 + 500000  # Prix entre 400k et 600k
    
    print(f"[TEST] Données générées: {X.shape} features, {len(y)} samples")
    
    # Initialiser le tuner avec peu de trials pour test rapide
    tuner = CatBoostTuner(
        X=X, 
        y=y,
        n_trials=3,  # Seulement 3 trials pour test rapide
        n_splits=3,  # Moins de splits pour accélérer
        early_stopping_rounds=10,
        random_state=42
    )
    
    print("[TEST] Lancement du tuning...")
    tuner.run_study()
    
    # Vérifier le temps d'entraînement
    print(f"[TEST] Temps d'entraînement mesuré: {tuner.training_time:.2f} seconds")
    
    if tuner.training_time > 0:
        print(f"[✓] Training time correctement mesuré: {tuner.training_time:.2f}s")
        
        # Formatage en minutes/heures
        if tuner.training_time < 60:
            print(f"[✓] Formatage: {tuner.training_time:.1f}s")
        elif tuner.training_time < 3600:
            minutes = int(tuner.training_time / 60)
            seconds = int(tuner.training_time % 60)
            print(f"[✓] Formatage: {minutes}m {seconds}s")
        else:
            hours = int(tuner.training_time / 3600)
            minutes = int((tuner.training_time % 3600) / 60)
            print(f"[✓] Formatage: {hours}h {minutes}m")
    else:
        print("[✗] Training time non mesuré (valeur 0)")
    
    return tuner.training_time

if __name__ == "__main__":
    test_training_time()
