#!/usr/bin/env python3

import requests
import json

API_BASE_URL = "http://127.0.0.1:8000"

def test_data_structure():
    """Test la structure des données pour React"""
    print("🔍 Test de la structure des données...")
    
    try:
        # Tester l'endpoint summary
        response = requests.get(f"{API_BASE_URL}/experiments/summary", timeout=10)
        if response.status_code == 200:
            summary = response.json()
            print("✅ Structure du résumé:")
            print(json.dumps(summary, indent=2))
            
            # Vérifier la structure de latest_experiment
            if "latest_experiment" in summary:
                latest = summary["latest_experiment"]
                print(f"\n📊 latest_experiment est un {type(latest).__name__}")
                if isinstance(latest, dict):
                    print(f"✅ Propriétés: {list(latest.keys())}")
                    print(f"   - id: {latest.get('id', 'N/A')}")
                    print(f"   - timestamp: {latest.get('timestamp', 'N/A')}")
                    print(f"   - r2_score: {latest.get('r2_score', 'N/A')}")
                else:
                    print(f"❌ latest_experiment n'est pas un dict: {latest}")
            
            return True
        else:
            print(f"❌ Erreur API: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_experiments_structure():
    """Test la structure des expériences"""
    print("\n🧪 Test de la structure des expériences...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/experiments", timeout=10)
        if response.status_code == 200:
            data = response.json()
            experiments = data.get("experiments", [])
            
            if experiments:
                print(f"✅ {len(experiments)} expériences trouvées")
                print("🔍 Structure de la première expérience:")
                print(json.dumps(experiments[0], indent=2))
                return True
            else:
                print("⚠️  Aucune expérience trouvée")
                return False
        else:
            print(f"❌ Erreur API: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Test de la structure des données pour React")
    print("=" * 50)
    
    success1 = test_data_structure()
    success2 = test_experiments_structure()
    
    if success1 and success2:
        print("\n🎉 Tous les tests de structure sont OK!")
        print("✅ L'erreur React devrait être résolue")
    else:
        print("\n❌ Problème de structure des données détecté")
