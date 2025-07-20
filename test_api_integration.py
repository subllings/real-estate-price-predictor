"""
Script de test simple pour vérifier que l'API Training Jobs fonctionne
"""

import requests
import json
from datetime import datetime

# Configuration - utilisez le port de votre API de prédiction
API_BASE = "http://localhost:8000"  # Port par défaut de l'API de prédiction

def test_training_jobs_api():
    """Test simple des endpoints training jobs"""
    
    print("🧪 Test des endpoints Training Jobs dans l'API de prédiction")
    print("=" * 60)
    print(f"🌐 Base URL: {API_BASE}")
    
    try:
        # 1. Test de santé
        print("\n1️⃣ Test de santé...")
        try:
            response = requests.get(f"{API_BASE}/training-jobs/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ API en ligne - Status: {data.get('status')}")
                if 'statistics' in data:
                    stats = data['statistics']
                    print(f"   📊 Stats: {stats['total_jobs']} jobs total, {stats['active_jobs']} actifs")
            else:
                print(f"❌ Health check échoué: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ Connexion impossible: {e}")
            print("💡 Vérifiez que l'API de prédiction est démarrée sur le port 8000")
            return False
        
        # 2. Lister les training jobs existants
        print("\n2️⃣ Liste des training jobs existants...")
        response = requests.get(f"{API_BASE}/training-jobs", timeout=10)
        if response.status_code == 200:
            data = response.json()
            jobs = data.get('training_jobs', [])
            print(f"✅ {len(jobs)} training jobs récupérés")
            
            # Afficher quelques détails des jobs existants
            for i, job in enumerate(jobs[:3]):  # Montrer max 3 jobs
                print(f"   Job {i+1}: {job.get('name', 'Sans nom')} - {job.get('status', 'unknown')}")
        else:
            print(f"❌ Erreur lors de la récupération: {response.status_code}")
            print(f"   Response: {response.text}")
        
        # 3. Créer un nouveau training job de test
        print("\n3️⃣ Création d'un nouveau training job...")
        job_data = {
            "name": "Test API Training Job",
            "model_type": "catboost",
            "target_r2": 0.85,
            "max_trials": 5,  # Petit nombre pour test rapide
            "compute_target": "api-test",
            "hyperparameters": {
                "learning_rate": 0.1,
                "depth": 4
            }
        }
        
        response = requests.post(
            f"{API_BASE}/training-jobs/start",
            json=job_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                job = data.get('job', {})
                job_id = job.get('id')
                print(f"✅ Job créé avec succès: {job_id}")
                print(f"   Nom: {job.get('name')}")
                print(f"   Status: {job.get('status')}")
                
                # 4. Vérifier que le job a été créé
                print("\n4️⃣ Vérification du job créé...")
                response = requests.get(f"{API_BASE}/training-jobs/{job_id}", timeout=10)
                if response.status_code == 200:
                    job_details = response.json()
                    print(f"✅ Job récupéré: {job_details.get('name')}")
                    print(f"   Machine: {job_details.get('machine_name')}")
                    print(f"   Créé à: {job_details.get('created_at')}")
                else:
                    print(f"❌ Impossible de récupérer le job: {response.status_code}")
                
                return True
            else:
                print(f"❌ Échec de création: {data}")
                return False
        else:
            print(f"❌ Erreur de création: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False

def test_frontend_compatibility():
    """Test de compatibilité avec le frontend React"""
    print("\n" + "=" * 60)
    print("⚛️ Test de compatibilité Frontend React")
    print("=" * 60)
    
    try:
        # Tester la structure de réponse attendue par le hook useTrainingJobs
        response = requests.get(f"{API_BASE}/training-jobs", timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Vérifier que la structure correspond à ce qu'attend le frontend
            required_fields = ['training_jobs', 'count', 'active_count', 'completed_count']
            
            for field in required_fields:
                if field in data:
                    print(f"✅ Champ '{field}' présent: {data[field]}")
                else:
                    print(f"❌ Champ '{field}' manquant")
                    return False
            
            # Vérifier la structure d'un job individuel
            jobs = data.get('training_jobs', [])
            if jobs:
                job = jobs[0]
                job_fields = ['id', 'name', 'status', 'progress', 'machine_name', 'model_type']
                
                print("\n📋 Structure du premier job:")
                for field in job_fields:
                    if field in job:
                        print(f"   ✅ {field}: {job[field]}")
                    else:
                        print(f"   ❌ {field}: manquant")
            
            print("\n✅ Structure de réponse compatible avec le frontend React")
            return True
        else:
            print(f"❌ Erreur API: {response.status_code}")
            return False
    
    except Exception as e:
        print(f"❌ Erreur de compatibilité: {e}")
        return False

if __name__ == "__main__":
    print(f"🚀 Test API Training Jobs - {datetime.now().strftime('%H:%M:%S')}")
    
    success = True
    
    # Test principal de l'API
    if not test_training_jobs_api():
        success = False
    
    # Test de compatibilité frontend
    if not test_frontend_compatibility():
        success = False
    
    if success:
        print(f"\n🎉 TESTS API RÉUSSIS!")
        print("📋 L'intégration avec CosmosDbLogger fonctionne")
        print("⚛️ Compatible avec le frontend React")
        print("🔗 Les endpoints training-jobs sont opérationnels")
        print(f"\n📖 Documentation API: {API_BASE}/docs")
    else:
        print(f"\n💥 ÉCHEC DES TESTS API")
        print("🔧 Vérifiez que l'API de prédiction est démarrée")
        print("🔧 Vérifiez la configuration Cosmos DB")
        print(f"🌐 URL testée: {API_BASE}")

    print(f"\n💡 Pour démarrer l'API de prédiction:")
    print(f"   cd app/backend-api-price-prediction")
    print(f"   python main.py")
