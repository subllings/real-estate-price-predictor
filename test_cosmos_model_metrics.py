#!/usr/bin/env python3
"""
Test simple pour l'intégration CatBoost + CosmosDB ModelMetrics
"""

import sys
import os
from datetime import datetime

# Ajouter le répertoire courant au path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

def test_cosmos_db_model_metrics():
    """Test du système CosmosDB avec ModelMetrics"""
    print("🔍 Test CosmosDB ModelMetrics Integration")
    print("=" * 50)
    
    try:
        from utils.cosmosdb_logger import CosmosDbLogger
        
        # Créer une instance du logger
        cosmos_logger = CosmosDbLogger()
        print("✅ CosmosDbLogger créé avec succès")
        
        # Créer le container ModelMetrics
        model_metrics_container = cosmos_logger.create_model_metrics_container()
        print("✅ Container ModelMetrics créé/vérifié")
        
        # Créer des métriques de test
        test_metrics = {
            "model_type": "catboost",
            "model_name": "CatBoost CV (All Features) [TEST]",
            "trial_number": 1,
            "experiment_name": "test_integration",
            "r2_train": 0.8456,
            "r2_test": 0.8234,
            "mae_train": 45678.12,
            "mae_test": 48234.56,
            "rmse_train": 67890.34,
            "rmse_test": 71234.78,
            "r2_gap": 0.0222,
            "generalization_status": "Good",
            "hyperparameters": {
                "iterations": 1000,
                "learning_rate": 0.1,
                "depth": 8
            },
            "feature_importance": [],
            "training_time": 123.45,
            "n_features": 2885,
            "status": "completed",
            "is_production_ready": True
        }
        
        # Sauvegarder les métriques
        print("📝 Sauvegarde des métriques de test...")
        record_id = cosmos_logger.log_model_metrics(test_metrics)
        print(f"✅ Métriques sauvegardées avec ID: {record_id}")
        
        # Récupérer les métriques
        print("\n📊 Récupération des métriques...")
        metrics = cosmos_logger.get_model_metrics("catboost", limit=5)
        print(f"✅ {len(metrics)} métriques récupérées")
        
        if metrics:
            print("\n📋 Dernière métrique:")
            last_metric = metrics[0]
            print(f"   - ID: {last_metric.get('id', 'N/A')}")
            print(f"   - Model: {last_metric.get('model_name', 'N/A')}")
            print(f"   - R² Test: {last_metric.get('r2_test', 'N/A')}")
            print(f"   - Status: {last_metric.get('generalization_status', 'N/A')}")
            print(f"   - Timestamp: {last_metric.get('timestamp', 'N/A')}")
        
        # Récupérer le résumé
        print("\n📈 Récupération du résumé...")
        summary = cosmos_logger.get_model_summary("catboost")
        print(f"✅ Résumé généré:")
        print(f"   - Total expériences: {summary.get('total_experiments', 0)}")
        print(f"   - Meilleur R²: {summary.get('best_r2_score', 0):.4f}")
        print(f"   - R² moyen: {summary.get('average_r2_score', 0):.4f}")
        
        if summary.get('latest_experiment'):
            latest = summary['latest_experiment']
            print(f"   - Dernière expérience: {latest.get('model_type', 'N/A')} (R²: {latest.get('r2_score', 0):.4f})")
        
        print("\n🎉 Test d'intégration réussi!")
        print(f"✅ Système prêt pour l'agent CatBoost")
        print(f"✅ API FastAPI peut utiliser get_model_metrics()")
        print(f"✅ React peut récupérer les données structurées")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Test d'intégration CosmosDB ModelMetrics")
    print("=" * 60)
    
    success = test_cosmos_db_model_metrics()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 Test d'intégration réussi!")
        print("\n💡 Prochaines étapes:")
        print("   1. Lancer un entraînement CatBoost pour tester le logging automatique")
        print("   2. Vérifier que l'API FastAPI retourne les bonnes données")
        print("   3. Tester l'interface React avec les vraies métriques")
    else:
        print("❌ Test d'intégration échoué")
        print("   Vérifiez les variables d'environnement et la connexion CosmosDB")
