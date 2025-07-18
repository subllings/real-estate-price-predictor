#!/usr/bin/env python3
"""
Script pour tester l'API /experiments et voir les diagnostics R2 Gap
"""

import requests
import json

def test_experiments_api():
    """Test de l'endpoint /experiments"""
    
    try:
        # Appeler l'API
        response = requests.get("http://localhost:8001/experiments", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Response successful - Found {len(data)} experiments")
            
            # Afficher les détails des premiers experiments
            for i, exp in enumerate(data[:3]):
                print(f"\n📊 Experiment {i+1}:")
                print(f"  Model: {exp.get('model_name', 'N/A')}")
                print(f"  R² Train: {exp.get('r2_train', 'N/A'):.6f}")
                print(f"  R² Test: {exp.get('r2_test', 'N/A'):.6f}")
                print(f"  R² Gap: {exp.get('r2_gap', 'N/A'):.6f}")
                print(f"  Generalization Status: {exp.get('generalization_status', 'N/A')}")
                print(f"  Training Time: {exp.get('training_time', 'N/A')}")
                print(f"  Timestamp: {exp.get('timestamp', 'N/A')}")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Backend server not running on localhost:8001")
        print("💡 Start the backend with: cd app/backend-api-price-prediction && uvicorn main:app --reload --port 8001")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_experiments_api()
