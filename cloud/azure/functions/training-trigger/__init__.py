import azure.functions as func
import logging
import json
import requests
import os
from datetime import datetime
from azure.cosmos import CosmosClient
from azure.storage.blob import BlobServiceClient

def main(mytimer: func.TimerRequest) -> None:
    """
    Agent 2 - Training Auto Agent
    Déclenché automatiquement via Timer Trigger
    Lance l'entraînement du modèle CatBoost via Azure Fabric
    """
    
    logging.info('🚀 Training Agent démarré - %s', datetime.utcnow())
    
    # Configuration
    cosmos_endpoint = os.environ.get("COSMOS_ENDPOINT")
    cosmos_key = os.environ.get("COSMOS_KEY")
    cosmos_database = "RealEstateDB"
    cosmos_container = "model_runs"
    
    fabric_workspace_id = os.environ.get("FABRIC_WORKSPACE_ID")
    fabric_notebook_id = os.environ.get("FABRIC_NOTEBOOK_ID")
    fabric_token = os.environ.get("FABRIC_ACCESS_TOKEN")
    
    try:
        # 1. Déclencher l'entraînement via Azure Fabric
        training_result = trigger_fabric_training(
            workspace_id=fabric_workspace_id,
            notebook_id=fabric_notebook_id,
            access_token=fabric_token
        )
        
        # 2. Logger le démarrage dans CosmosDB
        cosmos_client = CosmosClient(cosmos_endpoint, cosmos_key)
        database = cosmos_client.get_database_client(cosmos_database)
        container = database.get_container_client(cosmos_container)
        
        run_metadata = {
            "id": f"training_run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "agent": "training_auto_agent",
            "status": "started",
            "triggered_at": datetime.utcnow().isoformat(),
            "fabric_job_id": training_result.get("job_id"),
            "model_type": "catboost",
            "trigger_type": "timer_scheduled"
        }
        
        container.create_item(run_metadata)
        
        logging.info('✅ Training job démarré avec succès. Job ID: %s', training_result.get("job_id"))
        
    except Exception as e:
        logging.error('❌ Erreur lors du déclenchement de l\'entraînement: %s', str(e))
        
        # Logger l'erreur dans CosmosDB
        try:
            error_metadata = {
                "id": f"training_error_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                "agent": "training_auto_agent",
                "status": "error",
                "error_message": str(e),
                "triggered_at": datetime.utcnow().isoformat(),
                "trigger_type": "timer_scheduled"
            }
            container.create_item(error_metadata)
        except:
            pass  # Éviter les erreurs en cascade


def trigger_fabric_training(workspace_id: str, notebook_id: str, access_token: str) -> dict:
    """
    Déclenche l'exécution du notebook Fabric d'entraînement
    """
    
    # URL de l'API Fabric pour exécuter un notebook
    fabric_api_url = f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_id}/notebooks/{notebook_id}/jobs"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    # Paramètres pour l'entraînement
    job_payload = {
        "displayName": f"RealEstate_Training_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "parameters": {
            "model_type": "catboost",
            "use_optuna": True,
            "n_trials": 100,
            "cv_folds": 5,
            "save_to_blob": True
        }
    }
    
    response = requests.post(fabric_api_url, headers=headers, json=job_payload)
    
    if response.status_code == 202:  # Accepted
        job_info = response.json()
        logging.info('📊 Notebook Fabric job créé: %s', job_info.get("id"))
        return {"job_id": job_info.get("id"), "status": "accepted"}
    else:
        raise Exception(f"Fabric API error: {response.status_code} - {response.text}")


def check_fabric_job_status(workspace_id: str, job_id: str, access_token: str) -> dict:
    """
    Vérifie le statut d'un job Fabric (pour usage futur)
    """
    
    status_url = f"https://api.fabric.microsoft.com/v1/workspaces/{workspace_id}/notebooks/jobs/{job_id}"
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    response = requests.get(status_url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Status check failed: {response.status_code} - {response.text}")
