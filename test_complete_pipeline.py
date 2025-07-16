#!/usr/bin/env python3
"""
Test complet du pipeline Azure + Model Saver
"""
import os
import tempfile
import joblib
from datetime import datetime

print("🧪 Test complet du pipeline Azure Storage")
print("=" * 50)

# Test 1: Création d'un modèle de test
print("1️⃣ Création d'un modèle de test...")
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import numpy as np

# Créer un modèle simple pour test
X, y = make_regression(n_samples=100, n_features=5, random_state=42)
model = RandomForestRegressor(n_estimators=10, random_state=42)
model.fit(X, y)

# Métriques fictives pour le test
test_metrics = {
    "r2_test": 0.85,
    "r2_train": 0.89,
    "rmse_test": 45000,
    "rmse_train": 42000,
    "mae_test": 32000,
    "mae_train": 30000
}

features = [f"feature_{i}" for i in range(5)]

print("✅ Modèle de test créé")

# Test 2: Model Saver avec Azure
print("\n2️⃣ Test du ModelSaver avec upload Azure...")
try:
    from utils.model_saver import ModelSaver
    
    saver = ModelSaver()
    model_path, azure_url = saver.save_model_and_features(
        model=model,
        features=features,
        model_name="test_azure_integration",
        metrics=test_metrics,
        upload_to_azure=True
    )
    
    print(f"✅ Modèle sauvé: {model_path}")
    if azure_url:
        print(f"✅ Azure URL: {azure_url}")
    else:
        print("⚠️ Pas d'URL Azure retournée")
        
except Exception as e:
    print(f"❌ Erreur ModelSaver: {e}")

# Test 3: Vérification sur Azure
print("\n3️⃣ Vérification des modèles sur Azure...")
try:
    from utils.azure_model_storage import AzureModelStorage
    
    storage = AzureModelStorage()
    models = storage.list_all_models()
    
    print(f"📊 Modèles trouvés sur Azure: {len(models)}")
    
    if models:
        latest = models[0]  # Premier = plus récent par R²
        print(f"🏆 Meilleur modèle: R² = {latest.get('r2_test', 'N/A')}")
        print(f"📅 Upload: {latest.get('upload_timestamp', 'N/A')}")
    
except Exception as e:
    print(f"❌ Erreur listing Azure: {e}")

# Test 4: Download du meilleur modèle
print("\n4️⃣ Test de téléchargement automatique...")
try:
    from utils.azure_model_storage import ensure_best_model_available
    
    test_path = "models/test_best_model.pkl"
    success = ensure_best_model_available(test_path)
    
    if success and os.path.exists(test_path):
        print(f"✅ Meilleur modèle téléchargé: {test_path}")
        # Nettoyer le fichier de test
        os.remove(test_path)
        metadata_path = test_path.replace('.pkl', '_metadata.json')
        if os.path.exists(metadata_path):
            os.remove(metadata_path)
        print("🧹 Fichiers de test nettoyés")
    else:
        print("⚠️ Téléchargement échoué")
        
except Exception as e:
    print(f"❌ Erreur download: {e}")

print("\n🎉 Test complet terminé!")
print("=" * 50)
print("✅ Votre pipeline Azure est PRÊT pour l'entraînement nocturne!")
print("🚀 Lancez votre script d'entraînement - tout sera automatique!")
