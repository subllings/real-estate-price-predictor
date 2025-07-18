#!/usr/bin/env python3

import requests
import json
import time
import sys

API_BASE_URL = "http://127.0.0.1:8000"

def test_api_connection():
    """Test la connexion à l'API"""
    print("🔍 Test de connexion à l'API...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API accessible")
            return True
        else:
            print(f"❌ API retourne le statut {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur de connexion: {e}")
        return False

def test_experiments_endpoint():
    """Test l'endpoint des expériences"""
    print("🧪 Test de l'endpoint /experiments...")
    try:
        response = requests.get(f"{API_BASE_URL}/experiments", timeout=10)
        if response.status_code == 200:
            data = response.json()
            experiments = data.get("experiments", [])
            print(f"✅ {len(experiments)} expériences trouvées")
            
            if experiments:
                latest = experiments[0]
                print(f"   Dernière expérience: {latest.get('id', 'N/A')}")
                print(f"   R² Score: {latest.get('r2_test', latest.get('r2_score', 0))}")
                print(f"   Timestamp: {latest.get('timestamp', 'N/A')}")
            
            return True
        else:
            print(f"❌ Erreur {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_summary_endpoint():
    """Test l'endpoint du résumé"""
    print("📊 Test de l'endpoint /experiments/summary...")
    try:
        response = requests.get(f"{API_BASE_URL}/experiments/summary", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Résumé récupéré:")
            print(f"   Total expériences: {data.get('total_experiments', 0)}")
            print(f"   Meilleur R² score: {data.get('best_r2_score', 0)}")
            print(f"   Score moyen: {data.get('average_r2_score', 0):.3f}")
            
            latest = data.get('latest_experiment', {})
            if latest:
                print(f"   Dernière expérience: {latest.get('id', 'N/A')}")
            
            return True
        else:
            print(f"❌ Erreur {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_react_data_format():
    """Test le format des données pour React"""
    print("⚛️  Test du format des données pour React...")
    try:
        # Récupérer les données comme le ferait React
        exp_response = requests.get(f"{API_BASE_URL}/experiments", timeout=10)
        summary_response = requests.get(f"{API_BASE_URL}/experiments/summary", timeout=10)
        
        if exp_response.status_code == 200 and summary_response.status_code == 200:
            exp_data = exp_response.json()
            summary_data = summary_response.json()
            
            # Vérifier la structure des données
            experiments = exp_data.get("experiments", [])
            
            print(f"✅ Format des données validé")
            print(f"   Structure expériences: {len(experiments)} éléments")
            
            if experiments:
                exp = experiments[0]
                required_fields = ['id', 'timestamp', 'r2_test', 'mae_test', 'status']
                missing_fields = [field for field in required_fields if field not in exp or exp[field] is None]
                
                if missing_fields:
                    print(f"⚠️  Champs manquants: {missing_fields}")
                else:
                    print(f"✅ Tous les champs requis sont présents")
            
            return True
        else:
            print(f"❌ Erreur dans les réponses API")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    print("=== Test de l'interface React - Model Training ===\n")
    
    # Test de connexion
    if not test_api_connection():
        print("\n❌ Échec du test de connexion API")
        sys.exit(1)
    
    print()
    
    # Test des endpoints
    if not test_experiments_endpoint():
        print("\n❌ Échec du test des expériences")
        sys.exit(1)
    
    print()
    
    if not test_summary_endpoint():
        print("\n❌ Échec du test du résumé")
        sys.exit(1)
    
    print()
    
    if not test_react_data_format():
        print("\n❌ Échec du test du format des données")
        sys.exit(1)
    
    print("\n🎉 Tous les tests sont passés avec succès!")
    print("\n📱 L'interface React peut maintenant être lancée avec:")
    print("   ./launch-react-training.sh")
    print("\n🔗 Ou naviguez vers http://localhost:3000/training après avoir démarré React")

if __name__ == "__main__":
    main()
