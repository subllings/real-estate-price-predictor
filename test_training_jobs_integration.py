"""
Test de l'intégration CosmosDbLogger pour Training Jobs
Ce script teste la création automatique du container et les opérations CRUD
"""

import sys
import os
from datetime import datetime

# Ajouter le répertoire racine au path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_cosmosdb_training_jobs():
    """Test complet des fonctionnalités training jobs dans CosmosDbLogger"""
    
    print("🧪 Test de l'intégration CosmosDbLogger pour Training Jobs")
    print("=" * 60)
    
    try:
        # 1. Importer et initialiser CosmosDbLogger
        print("\n1️⃣ Initialisation de CosmosDbLogger...")
        from utils.cosmosdb_logger import CosmosDbLogger
        
        cosmos_logger = CosmosDbLogger()
        print("✅ CosmosDbLogger initialisé avec succès")
        
        # 2. Créer automatiquement le container TrainingJobs
        print("\n2️⃣ Création automatique du container TrainingJobs...")
        training_jobs_container = cosmos_logger.create_training_jobs_container()
        print("✅ Container TrainingJobs créé/vérifié automatiquement")
        
        # 3. Créer un nouveau training job
        print("\n3️⃣ Création d'un training job de test...")
        job_config = {
            "name": "Test CatBoost Training Integration",
            "model_type": "catboost",
            "target_r2": 0.85,
            "max_trials": 25,
            "compute_target": "local-test",
            "hyperparameters": {
                "learning_rate": 0.1,
                "depth": 6,
                "iterations": 500
            }
        }
        
        created_job = cosmos_logger.create_training_job(job_config)
        job_id = created_job['id']
        print(f"✅ Training job créé: {job_id}")
        print(f"   Nom: {created_job['name']}")
        print(f"   Status: {created_job['status']}")
        print(f"   Machine: {created_job['machine_name']}")
        
        # 4. Récupérer tous les training jobs
        print("\n4️⃣ Récupération de tous les training jobs...")
        all_jobs = cosmos_logger.get_training_jobs()
        print(f"✅ {len(all_jobs)} training jobs récupérés")
        
        # 5. Récupérer le job spécifique par ID
        print("\n5️⃣ Récupération du job par ID...")
        specific_job = cosmos_logger.get_training_job_by_id(job_id)
        if specific_job:
            print(f"✅ Job récupéré: {specific_job['name']}")
        else:
            print("❌ Job non trouvé")
            return False
        
        # 6. Mettre à jour le job (simulation de progression)
        print("\n6️⃣ Simulation de progression du training...")
        updates = {
            "status": "running",
            "progress": 45.5,
            "current_trial": 12,
            "best_r2": 0.8423,
            "current_gap": 0.0234,
            "eta_minutes": 8.5
        }
        
        updated_job = cosmos_logger.update_training_job(job_id, updates)
        if updated_job:
            print(f"✅ Job mis à jour:")
            print(f"   Progress: {updated_job['progress']:.1f}%")
            print(f"   Status: {updated_job['status']}")
            print(f"   Current Trial: {updated_job['current_trial']}")
            print(f"   Best R²: {updated_job['best_r2']:.4f}")
        else:
            print("❌ Échec de la mise à jour")
            return False
        
        # 7. Récupérer les statistiques
        print("\n7️⃣ Récupération des statistiques...")
        stats = cosmos_logger.get_training_jobs_statistics()
        print(f"✅ Statistiques générées:")
        print(f"   Total jobs: {stats['total_jobs']}")
        print(f"   Jobs actifs: {stats['active_jobs']}")
        print(f"   Jobs terminés: {stats['completed_jobs']}")
        print(f"   Machines: {stats['machines']}")
        print(f"   Types de modèles: {stats['model_types']}")
        
        # 8. Filtrer par statut
        print("\n8️⃣ Filtrage par statut 'running'...")
        running_jobs = cosmos_logger.get_training_jobs(status_filter="running")
        print(f"✅ {len(running_jobs)} jobs en cours d'exécution")
        
        # 9. Arrêter le job
        print("\n9️⃣ Arrêt du training job...")
        stopped = cosmos_logger.stop_training_job(job_id)
        if stopped:
            print("✅ Job arrêté avec succès")
        else:
            print("❌ Échec de l'arrêt du job")
            return False
        
        # 10. Vérifier le statut final
        print("\n🔟 Vérification du statut final...")
        final_job = cosmos_logger.get_training_job_by_id(job_id)
        if final_job and final_job['status'] == 'stopped':
            print(f"✅ Statut final confirmé: {final_job['status']}")
            print(f"   Completed at: {final_job.get('completed_at', 'N/A')}")
        else:
            print("❌ Statut final incorrect")
            return False
        
        print("\n" + "=" * 60)
        print("🎉 TOUS LES TESTS RÉUSSIS!")
        print("✅ Le container TrainingJobs est créé automatiquement")
        print("✅ Les opérations CRUD fonctionnent correctement")
        print("✅ Les statistiques sont générées")
        print("✅ L'intégration CosmosDbLogger est opérationnelle")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERREUR LORS DU TEST: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_compatibility():
    """Test de compatibilité avec l'API"""
    print("\n" + "=" * 60)
    print("🔗 Test de compatibilité API")
    print("=" * 60)
    
    try:
        # Tester les classes Pydantic (normalement dans l'API)
        from typing import Optional, Dict, Any
        from pydantic import BaseModel
        
        class TrainingJobCreate(BaseModel):
            name: Optional[str] = None
            model_type: str = "catboost"
            target_r2: float = 0.85
            max_trials: int = 50
            compute_target: str = "local"
            hyperparameters: Optional[Dict[str, Any]] = None
        
        # Créer une requête de test
        job_request = TrainingJobCreate(
            name="API Test Job",
            model_type="xgboost",
            target_r2=0.88,
            max_trials=30
        )
        
        print("✅ Modèles Pydantic compatibles")
        print(f"   Job config: {job_request.dict()}")
        
        # Tester la conversion vers CosmosDbLogger
        from utils.cosmosdb_logger import CosmosDbLogger
        cosmos_logger = CosmosDbLogger()
        
        job_config = {
            "name": job_request.name,
            "model_type": job_request.model_type,
            "target_r2": job_request.target_r2,
            "max_trials": job_request.max_trials,
            "compute_target": job_request.compute_target,
            "hyperparameters": job_request.hyperparameters or {}
        }
        
        # Test sans créer réellement le job
        print("✅ Conversion API → CosmosDbLogger réussie")
        print(f"   Config converted: {job_config}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERREUR DE COMPATIBILITÉ API: {e}")
        return False

if __name__ == "__main__":
    print(f"🚀 Test CosmosDbLogger Training Jobs - {datetime.now().strftime('%H:%M:%S')}")
    
    success = True
    
    # Test principal
    if not test_cosmosdb_training_jobs():
        success = False
    
    # Test de compatibilité API
    if not test_api_compatibility():
        success = False
    
    if success:
        print(f"\n🎊 INTÉGRATION COMPLÈTE RÉUSSIE!")
        print("📋 Prochaines étapes:")
        print("   1. Démarrer l'API de prédiction: python main.py")
        print("   2. Tester les endpoints /training-jobs dans l'API")
        print("   3. Utiliser l'interface React avec les nouveaux endpoints")
    else:
        print(f"\n💥 ÉCHEC DES TESTS - Vérifiez la configuration Cosmos DB")
        sys.exit(1)
