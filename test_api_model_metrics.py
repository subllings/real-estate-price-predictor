#!/usr/bin/env python3
"""
Test simple pour l'API FastAPI avec ModelMetrics
"""

import sys
import os
import requests
import json
from datetime import datetime

# Ajouter le répertoire courant au path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def test_api_experiments_endpoint():
    """Test de l'endpoint /experiments de l'API FastAPI"""
    print("🔍 Test API FastAPI - /experiments")
    print("=" * 50)
    
    try:
        # Tester l'endpoint local
        api_url = "http://localhost:8000/experiments"
        
        print(f"📞 Appel API: {api_url}")
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            experiments = data.get("experiments", [])
            
            print(f"✅ API répond avec succès")
            print(f"📊 {len(experiments)} expériences récupérées")
            
            if experiments:
                print("\n📋 Première expérience:")
                exp = experiments[0]
                print(f"   - ID: {exp.get('id', 'N/A')}")
                print(f"   - Model: {exp.get('model_name', 'N/A')}")
                print(f"   - R² Test: {exp.get('r2_test', 'N/A')}")
                print(f"   - Status: {exp.get('generalization_status', 'N/A')}")
                print(f"   - Timestamp: {exp.get('timestamp', 'N/A')}")
                
                # Vérifier les champs attendus par React
                expected_fields = [
                    'id', 'trial_number', 'experiment_name', 'model_type', 
                    'model_name', 'timestamp', 'r2_score', 'r2_test', 'r2_train',
                    'mae_test', 'rmse_test', 'generalization_status', 'is_production_ready'
                ]
                
                print(f"\n🔍 Vérification des champs pour React:")
                for field in expected_fields:
                    if field in exp:
                        print(f"   ✅ {field}: {exp[field]}")
                    else:
                        print(f"   ❌ {field}: MANQUANT")
                        
            return True
            
        else:
            print(f"❌ API erreur: {response.status_code}")
            print(f"   Message: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Impossible de se connecter à l'API")
        print("   Vérifiez que FastAPI est lancé sur http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du test API: {e}")
        return False

def test_api_summary_endpoint():
    """Test de l'endpoint /experiments/summary de l'API FastAPI"""
    print("\n🔍 Test API FastAPI - /experiments/summary")
    print("=" * 50)
    
    try:
        # Tester l'endpoint local
        api_url = "http://localhost:8000/experiments/summary"
        
        print(f"📞 Appel API: {api_url}")
        response = requests.get(api_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            print(f"✅ API répond avec succès")
            print(f"📊 Résumé récupéré:")
            print(f"   - Total expériences: {data.get('total_experiments', 0)}")
            print(f"   - Meilleur R²: {data.get('best_r2_score', 0):.4f}")
            print(f"   - R² moyen: {data.get('average_r2_score', 0):.4f}")
            
            if data.get('latest_experiment'):
                latest = data['latest_experiment']
                print(f"   - Dernière expérience: {latest.get('model_type', 'N/A')} (R²: {latest.get('r2_score', 0):.4f})")
                        
            return True
            
        else:
            print(f"❌ API erreur: {response.status_code}")
            print(f"   Message: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Impossible de se connecter à l'API")
        print("   Vérifiez que FastAPI est lancé sur http://localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Erreur lors du test API: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Test API FastAPI avec ModelMetrics")
    print("=" * 60)
    
    success1 = test_api_experiments_endpoint()
    success2 = test_api_summary_endpoint()
    
    print("\n" + "=" * 60)
    if success1 and success2:
        print("🎉 Tests API réussis!")
        print("\n💡 Prochaines étapes:")
        print("   1. Interface React peut maintenant récupérer les vraies données")
        print("   2. Lancer un entraînement CatBoost pour voir le logging automatique")
        print("   3. Tester l'interface React complète")
    else:
        print("❌ Certains tests API ont échoué")
        print("   Vérifiez que l'API FastAPI est lancée et accessible")
