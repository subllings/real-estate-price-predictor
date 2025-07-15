#!/usr/bin/env python3
"""
Script pour créer des modèles de test dans le dossier ml_models/
pour tester le système de registry
"""

import os
import joblib
import json
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import numpy as np

# Créer le dossier s'il n'existe pas
models_dir = "ml_models"
os.makedirs(models_dir, exist_ok=True)

# Créer des données fictives pour entraîner les modèles
np.random.seed(42)
X = np.random.randn(1000, 10)
y = np.random.randn(1000)

# Modèle 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X, y)

lr_metadata = {
    "name": "linear_regression_v1",
    "type": "linear_regression",
    "version": "1.0.0",
    "created_date": "2024-01-15T10:30:00",
    "metrics": {
        "mse": 0.95,
        "r2": 0.78,
        "mae": 0.65
    },
    "features": ["surface", "rooms", "bedrooms", "bathrooms", "garden", "terrace", "pool", "garage", "fireplace", "location_score"],
    "description": "Baseline linear regression model for property price prediction"
}

# Sauvegarder le modèle
joblib.dump(lr_model, os.path.join(models_dir, "linear_regression_v1.pkl"))
with open(os.path.join(models_dir, "linear_regression_v1_metadata.json"), "w") as f:
    json.dump(lr_metadata, f, indent=2)

# Modèle 2: Random Forest (meilleur)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

rf_metadata = {
    "name": "random_forest_v2",
    "type": "random_forest",
    "version": "2.1.0",
    "created_date": "2024-01-20T14:45:00",
    "metrics": {
        "mse": 0.72,
        "r2": 0.85,
        "mae": 0.54
    },
    "features": ["surface", "rooms", "bedrooms", "bathrooms", "garden", "terrace", "pool", "garage", "fireplace", "location_score"],
    "description": "Random Forest model with improved performance and feature importance analysis",
    "is_production": True
}

# Sauvegarder le modèle
joblib.dump(rf_model, os.path.join(models_dir, "random_forest_v2.pkl"))
with open(os.path.join(models_dir, "random_forest_v2_metadata.json"), "w") as f:
    json.dump(rf_metadata, f, indent=2)

# Modèle 3: Random Forest ancien (pour montrer l'évolution)
rf_old_model = RandomForestRegressor(n_estimators=50, random_state=42)
rf_old_model.fit(X, y)

rf_old_metadata = {
    "name": "random_forest_v1",
    "type": "random_forest",
    "version": "1.0.0",
    "created_date": "2024-01-10T09:15:00",
    "metrics": {
        "mse": 0.88,
        "r2": 0.76,
        "mae": 0.62
    },
    "features": ["surface", "rooms", "bedrooms", "bathrooms", "garden", "terrace", "pool", "garage", "fireplace", "location_score"],
    "description": "Initial Random Forest model - baseline version"
}

joblib.dump(rf_old_model, os.path.join(models_dir, "random_forest_v1.pkl"))
with open(os.path.join(models_dir, "random_forest_v1_metadata.json"), "w") as f:
    json.dump(rf_old_metadata, f, indent=2)

print("✅ Modèles de test créés avec succès !")
print(f"📁 Dossier: {models_dir}")
print("📋 Modèles créés:")
print("  - linear_regression_v1.pkl (MSE: 0.95, R²: 0.78)")
print("  - random_forest_v1.pkl (MSE: 0.88, R²: 0.76)")
print("  - random_forest_v2.pkl (MSE: 0.72, R²: 0.85) 🏆 PRODUCTION")
print("\n🚀 Vous pouvez maintenant tester le système de registry !")
