#!/usr/bin/env python3
"""
Test rapide de connexion CosmosDB et récupération des données
"""

import sys
import os
sys.path.append('.')

def test_cosmos_connection():
    """Test de la connexion et récupération des données"""
    print("🔍 Testing CosmosDB connection...")
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        print("✅ Environment loaded")
        
        # Vérifier les variables d'environnement
        cosmos_endpoint = os.getenv('COSMOS_ENDPOINT')
        cosmos_key = os.getenv('COSMOS_KEY')
        cosmos_db_name = os.getenv('COSMOS_DATABASE_NAME')
        cosmos_container = os.getenv('COSMOS_CONTAINER_NAME')
        
        print(f"🔧 COSMOS_ENDPOINT: {cosmos_endpoint[:50] if cosmos_endpoint else 'NOT SET'}...")
        print(f"🔧 COSMOS_KEY: {'[SET]' if cosmos_key else 'NOT SET'}")
        print(f"🔧 COSMOS_DATABASE_NAME: {cosmos_db_name}")
        print(f"🔧 COSMOS_CONTAINER_NAME: {cosmos_container}")
        
        if not cosmos_endpoint or not cosmos_key:
            print("❌ CosmosDB credentials not configured")
            return False
        
        # Tester la connexion
        from utils.cosmosdb_logger import CosmosDbLogger
        cosmos_logger = CosmosDbLogger()
        print("✅ CosmosDB logger created")
        
        # Test 1: ModelMetrics container
        print("\n📊 Testing ModelMetrics container...")
        try:
            experiments = cosmos_logger.get_model_metrics('catboost', limit=5, container_name='ModelMetrics')
            print(f"✅ Found {len(experiments)} experiments in ModelMetrics")
            
            for i, exp in enumerate(experiments[:3]):
                print(f"   [{i+1}] ID: {exp.get('id', 'Unknown')}")
                print(f"       R² Test: {exp.get('r2_test', 0):.4f}")
                print(f"       R² Gap: {exp.get('r2_gap', 0):.4f}")
                print(f"       Status: {exp.get('generalization_status', 'Unknown')}")
                print()
                
        except Exception as e:
            print(f"❌ ModelMetrics failed: {e}")
        
        # Test 2: Legacy container
        print("\n🔄 Testing legacy container...")
        try:
            experiments = cosmos_logger.get_trials_for_model('catboost', limit=5)
            print(f"✅ Found {len(experiments)} experiments in legacy container")
            
            for i, exp in enumerate(experiments[:3]):
                print(f"   [{i+1}] ID: {exp.get('id', 'Unknown')}")
                print(f"       R² Test: {exp.get('r2_test', 0):.4f}")
                print(f"       Type: {exp.get('type', 'Unknown')}")
                print()
                
        except Exception as e:
            print(f"❌ Legacy container failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ General error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_cosmos_connection()
    if success:
        print("\n✅ CosmosDB connection test completed")
    else:
        print("\n❌ CosmosDB connection test failed")
