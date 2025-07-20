"""
Test de l'API Training Jobs
"""

import requests
import json
import time
from datetime import datetime

# Configuration
BASE_URL = "http://localhost:8003"
API_URL = f"{BASE_URL}/api"

def test_api_health():
    """Test de santé de l'API"""
    print("🔍 Test de santé de l'API...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API Training en ligne")
            return True
        else:
            print(f"❌ API Training erreur: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Connexion impossible: {e}")
        return False

def test_list_training_jobs():
    """Test de listage des training jobs"""
    print("\n📋 Test: Récupération des training jobs...")
    try:
        response = requests.get(f"{API_URL}/training-jobs", timeout=10)
        if response.status_code == 200:
            data = response.json()
            jobs = data.get('training_jobs', [])
            print(f"✅ {len(jobs)} training jobs récupérés")
            
            # Afficher le premier job s'il existe
            if jobs:
                job = jobs[0]
                print(f"   Premier job: {job.get('name', 'Sans nom')}")
                print(f"   Status: {job.get('status')}")
                print(f"   Progress: {job.get('progress', 0):.1f}%")
            
            return jobs
        else:
            print(f"❌ Erreur {response.status_code}: {response.text}")
            return []
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return []

def test_create_training_job():
    """Test de création d'un nouveau training job"""
    print("\n🚀 Test: Création d'un nouveau training job...")
    try:
        job_data = {
            "name": "Test CatBoost Training",
            "model_type": "catboost",
            "target_r2": 0.85,
            "max_trials": 10,
            "compute_target": "local-test",
            "hyperparameters": {
                "learning_rate": 0.1,
                "depth": 6,
                "iterations": 500
            }
        }
        
        response = requests.post(
            f"{API_URL}/training-jobs/start",
            json=job_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                job = data.get('job', {})
                job_id = job.get('id')
                print(f"✅ Training job créé: {job_id}")
                print(f"   Nom: {job.get('name')}")
                print(f"   Status: {job.get('status')}")
                return job_id
            else:
                print(f"❌ Échec de création: {data}")
                return None
        else:
            print(f"❌ Erreur {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

def test_get_specific_job(job_id):
    """Test de récupération d'un job spécifique"""
    if not job_id:
        return None
        
    print(f"\n🔍 Test: Récupération du job {job_id}...")
    try:
        response = requests.get(f"{API_URL}/training-jobs/{job_id}", timeout=10)
        if response.status_code == 200:
            job = response.json()
            print(f"✅ Job récupéré: {job.get('name')}")
            print(f"   Status: {job.get('status')}")
            print(f"   Progress: {job.get('progress', 0):.1f}%")
            print(f"   Machine: {job.get('machine_name')}")
            return job
        else:
            print(f"❌ Erreur {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

def test_stop_training_job(job_id):
    """Test d'arrêt d'un training job"""
    if not job_id:
        return False
        
    print(f"\n⏹️ Test: Arrêt du job {job_id}...")
    try:
        response = requests.post(f"{API_URL}/training-jobs/{job_id}/stop", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print(f"✅ Job arrêté avec succès")
                return True
            else:
                print(f"❌ Échec d'arrêt: {data}")
                return False
        else:
            print(f"❌ Erreur {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_training_jobs_health():
    """Test de santé spécifique aux training jobs"""
    print("\n🏥 Test: Santé des training jobs...")
    try:
        response = requests.get(f"{API_URL}/training-jobs/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Service training jobs: {data.get('status')}")
            return True
        else:
            print(f"❌ Erreur {response.status_code}: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def run_all_tests():
    """Exécute tous les tests"""
    print("🧪 Tests de l'API Training Jobs")
    print("=" * 50)
    
    # Test de santé général
    if not test_api_health():
        print("\n❌ API non disponible, arrêt des tests")
        return False
    
    # Test de santé des training jobs
    test_training_jobs_health()
    
    # Test de listage
    existing_jobs = test_list_training_jobs()
    
    # Test de création
    new_job_id = test_create_training_job()
    
    # Attendre un peu pour que le job se mette en route
    if new_job_id:
        print("\n⏳ Attente de 5 secondes pour la progression...")
        time.sleep(5)
        
        # Test de récupération spécifique
        test_get_specific_job(new_job_id)
        
        # Test d'arrêt
        test_stop_training_job(new_job_id)
    
    # Test final de listage
    print("\n📋 Test final: Liste après modifications...")
    final_jobs = test_list_training_jobs()
    
    print("\n" + "=" * 50)
    print("🎉 Tests terminés!")
    print(f"📊 Jobs initiaux: {len(existing_jobs)}")
    print(f"📊 Jobs finaux: {len(final_jobs)}")
    
    return True

if __name__ == "__main__":
    print(f"🚀 Test de l'API Training Jobs - {datetime.now().strftime('%H:%M:%S')}")
    print(f"🌐 URL de base: {BASE_URL}")
    
    try:
        run_all_tests()
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrompus par l'utilisateur")
    except Exception as e:
        print(f"\n❌ Erreur lors des tests: {e}")
