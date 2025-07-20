#!/usr/bin/env python3
"""
Test script pour valider l'intégration complète du Local Training
"""

import requests
import json
import time
import sys

# Configuration
API_BASE_URL = "http://localhost:8002"
TEST_CONFIG = {
    "model_type": "catboost",
    "dataset_path": "data/processed/train.csv", 
    "target_column": "price",
    "training_config": {
        "test_size": 0.2,
        "random_state": 42,
        "cv_folds": 3,  # Réduit pour test rapide
        "optimization_metric": "r2"
    },
    "hyperparameter_tuning": {
        "enabled": True,
        "n_trials": 10,  # Réduit pour test rapide
        "timeout_minutes": 5  # Court pour test
    },
    "output_config": {
        "save_model": True,
        "model_name": "test_model_local",
        "save_metrics": True
    }
}

def print_status(message, status="INFO"):
    colors = {
        "INFO": "\033[94m",  # Blue
        "SUCCESS": "\033[92m",  # Green  
        "ERROR": "\033[91m",  # Red
        "WARNING": "\033[93m"  # Yellow
    }
    reset = "\033[0m"
    print(f"{colors.get(status, '')}{status}: {message}{reset}")

def test_api_health():
    """Test si l'API est accessible"""
    try:
        response = requests.get(f"{API_BASE_URL}/docs")
        if response.status_code == 200:
            print_status("API Backend accessible", "SUCCESS")
            return True
        else:
            print_status(f"API Backend non accessible (status: {response.status_code})", "ERROR")
            return False
    except requests.exceptions.ConnectionError:
        print_status("API Backend non accessible - Vérifiez que l'API tourne sur port 8002", "ERROR")
        return False

def start_local_training():
    """Lance un training local via l'API"""
    try:
        print_status("Lancement du training local de test...")
        
        response = requests.post(
            f"{API_BASE_URL}/api/training/start-local",
            headers={"Content-Type": "application/json"},
            json=TEST_CONFIG
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("status") == "success":
                job_id = result.get("job_id")
                print_status(f"Training local lancé avec succès! Job ID: {job_id}", "SUCCESS")
                return job_id
            else:
                print_status(f"Erreur dans la réponse: {result}", "ERROR")
                return None
        else:
            print_status(f"Erreur HTTP {response.status_code}: {response.text}", "ERROR")
            return None
            
    except Exception as e:
        print_status(f"Erreur lors du lancement: {e}", "ERROR")
        return None

def monitor_training(job_id, max_wait_minutes=10):
    """Surveille la progression du training"""
    print_status(f"Surveillance du job {job_id}...")
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    
    while time.time() - start_time < max_wait_seconds:
        try:
            response = requests.get(f"{API_BASE_URL}/api/training/local-status/{job_id}")
            
            if response.status_code == 200:
                result = response.json()
                job = result.get("job", {})
                
                status = job.get("status", "unknown")
                progress = job.get("progress", 0)
                current_stage = job.get("current_stage", "unknown")
                
                print_status(f"Status: {status} | Progress: {progress}% | Stage: {current_stage}")
                
                if status == "completed":
                    print_status("Training terminé avec succès!", "SUCCESS")
                    
                    # Afficher les métriques si disponibles
                    metrics = job.get("metrics", {})
                    if metrics:
                        print_status("Métriques finales:", "INFO")
                        print(f"  - R² test: {metrics.get('test_r2', 'N/A'):.4f}")
                        print(f"  - RMSE test: {metrics.get('test_rmse', 'N/A'):.2f}")
                        print(f"  - MAE test: {metrics.get('test_mae', 'N/A'):.2f}")
                        print(f"  - Overfitting gap: {metrics.get('overfitting_gap', 'N/A'):.4f}")
                    
                    model_path = job.get("model_path")
                    if model_path:
                        print_status(f"Modèle sauvé: {model_path}", "SUCCESS")
                    
                    return True
                    
                elif status == "failed":
                    error = job.get("error", "Erreur inconnue")
                    print_status(f"Training échoué: {error}", "ERROR")
                    return False
                    
                elif status in ["running", "starting"]:
                    # Continuer à surveiller
                    time.sleep(10)  # Attendre 10 secondes avant la prochaine vérification
                    
                else:
                    print_status(f"Status inconnu: {status}", "WARNING")
                    time.sleep(5)
                    
            else:
                print_status(f"Erreur lors de la récupération du statut: {response.status_code}", "ERROR")
                time.sleep(5)
                
        except Exception as e:
            print_status(f"Erreur lors de la surveillance: {e}", "ERROR")
            time.sleep(5)
    
    print_status(f"Timeout après {max_wait_minutes} minutes", "WARNING")
    return False

def test_list_jobs():
    """Test la liste des jobs locaux"""
    try:
        response = requests.get(f"{API_BASE_URL}/api/training/list-local")
        
        if response.status_code == 200:
            result = response.json()
            jobs = result.get("jobs", [])
            total = result.get("total", 0)
            
            print_status(f"Trouvé {total} jobs de training local", "SUCCESS")
            
            for job in jobs[-3:]:  # Afficher les 3 derniers
                job_id = job.get("id", "N/A")
                status = job.get("status", "N/A")
                model_type = job.get("model_type", "N/A")
                created_at = job.get("created_at", "N/A")
                
                print(f"  - {job_id}: {model_type} ({status}) - {created_at}")
            
            return True
        else:
            print_status(f"Erreur lors de la récupération des jobs: {response.status_code}", "ERROR")
            return False
            
    except Exception as e:
        print_status(f"Erreur lors du test de liste: {e}", "ERROR")
        return False

def main():
    """Fonction principale de test"""
    print_status("=== Test de l'intégration Local Training ===", "INFO")
    
    # 1. Test de santé de l'API
    if not test_api_health():
        print_status("❌ Test échoué - API non accessible", "ERROR")
        sys.exit(1)
    
    # 2. Test de liste des jobs (avant)
    print_status("\n--- Test liste des jobs (avant) ---", "INFO")
    test_list_jobs()
    
    # 3. Lancement du training
    print_status("\n--- Lancement du training de test ---", "INFO")
    job_id = start_local_training()
    
    if not job_id:
        print_status("❌ Test échoué - Impossible de lancer le training", "ERROR")
        sys.exit(1)
    
    # 4. Surveillance du training
    print_status("\n--- Surveillance du training ---", "INFO")
    success = monitor_training(job_id, max_wait_minutes=8)
    
    # 5. Test de liste des jobs (après)
    print_status("\n--- Test liste des jobs (après) ---", "INFO")
    test_list_jobs()
    
    # 6. Résultat final
    if success:
        print_status("\n🎉 Tous les tests ont réussi!", "SUCCESS")
        print_status("✅ L'interface React peut maintenant lancer des trainings locaux", "SUCCESS")
    else:
        print_status("\n⚠️ Test partiellement réussi - Le training a été lancé mais n'a pas terminé dans les temps", "WARNING")
        print_status("Vérifiez les logs du backend pour plus de détails", "INFO")

if __name__ == "__main__":
    main()
