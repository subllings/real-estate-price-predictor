"""
Script pour créer et gérer le container TrainingJobs dans Cosmos DB
Ce container stocke les informations sur les entraînements en cours et terminés
"""

from azure.cosmos import CosmosClient, PartitionKey
import os
from datetime import datetime, timezone
import json

def create_training_jobs_container():
    """Crée le container TrainingJobs s'il n'existe pas"""
    try:
        # Configuration Cosmos DB
        endpoint = os.getenv('COSMOS_ENDPOINT')
        key = os.getenv('COSMOS_KEY')
        database_name = os.getenv('COSMOS_DATABASE_NAME', 'ml-experiments')
        
        if not endpoint or not key:
            print("❌ Variables d'environnement COSMOS_ENDPOINT et COSMOS_KEY requises")
            return False
            
        # Connexion
        client = CosmosClient(endpoint, key)
        database = client.get_database_client(database_name)
        
        # Créer le container TrainingJobs
        container_name = "TrainingJobs"
        try:
            container = database.create_container(
                id=container_name,
                partition_key=PartitionKey(path="/machine_name"),
                offer_throughput=400
            )
            print(f"✅ Container {container_name} créé avec succès")
        except Exception as e:
            if "already exists" in str(e):
                container = database.get_container_client(container_name)
                print(f"ℹ️ Container {container_name} existe déjà")
            else:
                raise e
        
        # Insérer des données de test
        sample_jobs = [
            {
                "id": "catboost-opt-001",
                "name": "CatBoost Hyperparameter Optimization",
                "status": "running",
                "progress": 78.5,
                "eta_minutes": 7,
                "current_trial": 39,
                "total_trials": 50,
                "best_r2": 0.8512,
                "target_r2": 0.85,
                "current_gap": 0.0234,
                "compute_target": "Desktop-Intel-i7",
                "machine_name": "LAPTOP-DEV-01",
                "started_at": "2024-01-15T09:30:00Z",
                "model_type": "catboost",
                "hyperparameters": {
                    "learning_rate": 0.1,
                    "depth": 8,
                    "iterations": 1000
                },
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": "catboost-distributed-002",
                "name": "Distributed CatBoost Training",
                "status": "running",
                "progress": 45.2,
                "eta_minutes": 12,
                "current_trial": 23,
                "total_trials": 75,
                "best_r2": 0.8387,
                "target_r2": 0.85,
                "current_gap": 0.0456,
                "compute_target": "Azure-ML-Cluster",
                "machine_name": "gpu-cluster-node-2",
                "started_at": "2024-01-15T10:15:00Z",
                "model_type": "catboost",
                "hyperparameters": {
                    "learning_rate": 0.08,
                    "depth": 10,
                    "iterations": 1500
                },
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            },
            {
                "id": "xgboost-weekend-003",
                "name": "Weekend XGBoost Experiment",
                "status": "completed",
                "progress": 100,
                "eta_minutes": 0,
                "current_trial": 100,
                "total_trials": 100,
                "best_r2": 0.8467,
                "target_r2": 0.85,
                "final_gap": 0.0298,
                "compute_target": "Desktop-RTX-3080",
                "machine_name": "DESKTOP-ML-02",
                "started_at": "2024-01-15T08:00:00Z",
                "completed_at": "2024-01-15T10:30:00Z",
                "model_type": "xgboost",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        ]
        
        # Insérer les jobs de test
        for job in sample_jobs:
            try:
                container.create_item(job)
                print(f"✅ Job de test créé: {job['name']}")
            except Exception as e:
                if "already exists" in str(e):
                    print(f"ℹ️ Job {job['id']} existe déjà")
                else:
                    print(f"⚠️ Erreur lors de la création du job {job['id']}: {e}")
        
        print(f"\n🎉 Container TrainingJobs configuré avec succès!")
        print(f"📍 Endpoint: {endpoint}")
        print(f"🗃️ Database: {database_name}")
        print(f"📦 Container: {container_name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors de la création du container: {e}")
        return False

def list_training_jobs():
    """Liste tous les training jobs"""
    try:
        endpoint = os.getenv('COSMOS_ENDPOINT')
        key = os.getenv('COSMOS_KEY')
        database_name = os.getenv('COSMOS_DATABASE_NAME', 'ml-experiments')
        
        client = CosmosClient(endpoint, key)
        database = client.get_database_client(database_name)
        container = database.get_container_client("TrainingJobs")
        
        items = list(container.read_all_items())
        
        print(f"\n📋 Training Jobs ({len(items)} total):")
        print("-" * 60)
        
        for item in items:
            status_emoji = {
                'running': '🔄',
                'completed': '✅',
                'failed': '❌',
                'stopped': '⏹️',
                'queued': '⏳'
            }.get(item.get('status', 'unknown'), '❓')
            
            print(f"{status_emoji} {item.get('name', 'Unnamed')}")
            print(f"   ID: {item.get('id')}")
            print(f"   Status: {item.get('status')}")
            print(f"   Progress: {item.get('progress', 0):.1f}%")
            print(f"   Machine: {item.get('machine_name')}")
            print(f"   Started: {item.get('started_at', 'N/A')}")
            print()
        
        return items
        
    except Exception as e:
        print(f"❌ Erreur lors de la lecture des training jobs: {e}")
        return []

if __name__ == "__main__":
    print("🚀 Configuration du container TrainingJobs...")
    
    # Créer le container
    success = create_training_jobs_container()
    
    if success:
        print("\n" + "="*60)
        list_training_jobs()
    else:
        print("❌ Échec de la configuration")
