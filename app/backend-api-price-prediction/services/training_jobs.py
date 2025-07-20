"""
API endpoints pour gérer les training jobs dans l'API Real Estate Price Prediction
Endpoints pour lister, créer, mettre à jour et arrêter les entraînements
"""

from fastapi import HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from datetime import datetime, timezone
import uuid
import os
from azure.cosmos import CosmosClient
import logging

# Configuration du logging
logger = logging.getLogger(__name__)

# Modèles Pydantic
class TrainingJobCreate(BaseModel):
    name: Optional[str] = None
    model_type: str = "catboost"
    target_r2: float = 0.85
    max_trials: int = 50
    compute_target: str = "local"
    hyperparameters: Optional[Dict[str, Any]] = None

class TrainingJobUpdate(BaseModel):
    status: Optional[str] = None
    progress: Optional[float] = None
    eta_minutes: Optional[float] = None
    current_trial: Optional[int] = None
    best_r2: Optional[float] = None
    current_gap: Optional[float] = None

class TrainingJob(BaseModel):
    id: str
    name: str
    status: str
    progress: float
    eta_minutes: float
    current_trial: int
    total_trials: int
    best_r2: float
    target_r2: float
    current_gap: Optional[float] = None
    final_gap: Optional[float] = None
    compute_target: str
    machine_name: str
    model_type: str
    started_at: str
    completed_at: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str

# Helper functions
def get_cosmos_container():
    """Récupère le container Cosmos DB pour les training jobs"""
    try:
        endpoint = os.getenv('COSMOS_ENDPOINT')
        key = os.getenv('COSMOS_KEY')
        database_name = os.getenv('COSMOS_DATABASE_NAME', 'ml-experiments')
        
        if not endpoint or not key:
            raise HTTPException(
                status_code=500, 
                detail="Configuration Cosmos DB manquante"
            )
        
        client = CosmosClient(endpoint, key)
        database = client.get_database_client(database_name)
        container = database.get_container_client("TrainingJobs")
        
        return container
    except Exception as e:
        logger.error(f"Erreur connexion Cosmos DB: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur de connexion à la base de données: {str(e)}"
        )

def get_machine_name():
    """Récupère le nom de la machine actuelle"""
    import socket
    try:
        return socket.gethostname()
    except:
        return "unknown-machine"

# Fonctions pour les endpoints (à importer dans main.py)
async def get_training_jobs():
    """Récupère tous les training jobs"""
    try:
        container = get_cosmos_container()
        
        # Récupérer tous les jobs et les trier par date de création (plus récents en premier)
        query = "SELECT * FROM c ORDER BY c.created_at DESC"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        logger.info(f"Récupéré {len(items)} training jobs")
        
        return {
            "training_jobs": items,
            "count": len(items),
            "active_count": len([j for j in items if j.get('status') in ['running', 'queued']]),
            "completed_count": len([j for j in items if j.get('status') == 'completed'])
        }
        
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des training jobs: {e}")
        
        # Retourner des données de demo en cas d'erreur Cosmos DB
        demo_jobs = [
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
                "started_at": datetime.now(timezone.utc).isoformat(),
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
                "started_at": datetime.now(timezone.utc).isoformat(),
                "model_type": "catboost",
                "hyperparameters": {
                    "learning_rate": 0.08,
                    "depth": 10,
                    "iterations": 1500
                },
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }
        ]
        
        return {
            "training_jobs": demo_jobs,
            "count": len(demo_jobs),
            "active_count": len([j for j in demo_jobs if j.get('status') in ['running', 'queued']]),
            "completed_count": len([j for j in demo_jobs if j.get('status') == 'completed']),
            "demo_mode": True,
            "error": str(e)
        }

async def get_training_job(job_id: str):
    """Récupère un training job spécifique"""
    try:
        container = get_cosmos_container()
        
        # Rechercher le job par ID
        query = f"SELECT * FROM c WHERE c.id = '{job_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        if not items:
            raise HTTPException(status_code=404, detail="Training job non trouvé")
        
        return items[0]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération du job {job_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de la récupération: {str(e)}"
        )

async def start_training_job(job_request: TrainingJobCreate, background_tasks: BackgroundTasks):
    """Démarre un nouveau training job"""
    try:
        container = get_cosmos_container()
        
        # Générer un ID unique
        job_id = f"{job_request.model_type}-{uuid.uuid4().hex[:8]}"
        machine_name = get_machine_name()
        now = datetime.now(timezone.utc).isoformat()
        
        # Créer le job
        job_data = {
            "id": job_id,
            "name": job_request.name or f"{job_request.model_type.upper()} Training Session",
            "status": "queued",
            "progress": 0.0,
            "eta_minutes": 20.0,
            "current_trial": 0,
            "total_trials": job_request.max_trials,
            "best_r2": 0.0,
            "target_r2": job_request.target_r2,
            "current_gap": 0.0,
            "compute_target": job_request.compute_target,
            "machine_name": machine_name,
            "model_type": job_request.model_type,
            "started_at": now,
            "hyperparameters": job_request.hyperparameters or {},
            "created_at": now,
            "updated_at": now
        }
        
        # Sauvegarder dans Cosmos DB
        created_job = container.create_item(job_data)
        
        logger.info(f"Nouveau training job créé: {job_id}")
        
        # Démarrer le training en arrière-plan (simulation pour la demo)
        background_tasks.add_task(simulate_training_progress, job_id)
        
        return {
            "success": True,
            "job": created_job,
            "message": f"Training job {job_id} démarré avec succès"
        }
        
    except Exception as e:
        logger.error(f"Erreur lors du démarrage du training job: {e}")
        
        # En mode demo, créer un job factice
        job_id = f"{job_request.model_type}-demo-{uuid.uuid4().hex[:6]}"
        demo_job = {
            "id": job_id,
            "name": job_request.name or f"{job_request.model_type.upper()} Demo Training",
            "status": "queued",
            "progress": 0.0,
            "eta_minutes": 15.0,
            "current_trial": 0,
            "total_trials": job_request.max_trials,
            "best_r2": 0.0,
            "target_r2": job_request.target_r2,
            "current_gap": 0.0,
            "compute_target": job_request.compute_target,
            "machine_name": get_machine_name(),
            "model_type": job_request.model_type,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "hyperparameters": job_request.hyperparameters or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        return {
            "success": True,
            "job": demo_job,
            "message": f"Demo training job {job_id} créé (Cosmos DB indisponible)",
            "demo_mode": True,
            "error": str(e)
        }

async def stop_training_job(job_id: str):
    """Arrête un training job"""
    try:
        container = get_cosmos_container()
        
        # Récupérer le job
        query = f"SELECT * FROM c WHERE c.id = '{job_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        if not items:
            raise HTTPException(status_code=404, detail="Training job non trouvé")
        
        job = items[0]
        
        if job.get('status') not in ['running', 'queued']:
            raise HTTPException(
                status_code=400, 
                detail=f"Impossible d'arrêter un job avec le status: {job.get('status')}"
            )
        
        # Mettre à jour le status
        job['status'] = 'stopped'
        job['completed_at'] = datetime.now(timezone.utc).isoformat()
        job['updated_at'] = datetime.now(timezone.utc).isoformat()
        
        # Sauvegarder
        container.replace_item(item=job['id'], body=job)
        
        logger.info(f"Training job {job_id} arrêté")
        
        return {
            "success": True,
            "message": f"Training job {job_id} arrêté avec succès"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de l'arrêt du job {job_id}: {e}")
        
        # En mode demo, simuler l'arrêt
        return {
            "success": True,
            "message": f"Demo: Training job {job_id} marqué comme arrêté",
            "demo_mode": True,
            "error": str(e)
        }

async def update_training_job(job_id: str, update_data: TrainingJobUpdate):
    """Met à jour un training job"""
    try:
        container = get_cosmos_container()
        
        # Récupérer le job
        query = f"SELECT * FROM c WHERE c.id = '{job_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        if not items:
            raise HTTPException(status_code=404, detail="Training job non trouvé")
        
        job = items[0]
        
        # Mettre à jour les champs fournis
        update_dict = update_data.dict(exclude_none=True)
        for key, value in update_dict.items():
            job[key] = value
        
        job['updated_at'] = datetime.now(timezone.utc).isoformat()
        
        # Marquer comme terminé si progression = 100%
        if job.get('progress', 0) >= 100 and job.get('status') == 'running':
            job['status'] = 'completed'
            job['completed_at'] = datetime.now(timezone.utc).isoformat()
            if 'current_gap' in job:
                job['final_gap'] = job['current_gap']
        
        # Sauvegarder
        updated_job = container.replace_item(item=job['id'], body=job)
        
        logger.info(f"Training job {job_id} mis à jour")
        
        return {
            "success": True,
            "job": updated_job,
            "message": f"Training job {job_id} mis à jour avec succès"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la mise à jour du job {job_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de la mise à jour: {str(e)}"
        )

# Fonction de simulation pour la demo
async def simulate_training_progress(job_id: str):
    """Simule la progression d'un training job pour la demo"""
    import asyncio
    import random
    
    try:
        # Attendre un peu puis commencer
        await asyncio.sleep(2)
        
        container = get_cosmos_container()
        
        # Récupérer le job
        query = f"SELECT * FROM c WHERE c.id = '{job_id}'"
        items = list(container.query_items(query=query, enable_cross_partition_query=True))
        
        if not items:
            return
        
        job = items[0]
        
        # Démarrer l'entraînement
        job['status'] = 'running'
        job['updated_at'] = datetime.now(timezone.utc).isoformat()
        container.replace_item(item=job['id'], body=job)
        
        # Simuler la progression
        for trial in range(1, job['total_trials'] + 1):
            if job.get('status') != 'running':
                break
                
            # Progression aléatoire
            progress = (trial / job['total_trials']) * 100
            eta = max(0, (job['total_trials'] - trial) * 0.5)  # 30 sec par trial
            r2_improvement = random.uniform(0.001, 0.01)
            gap_change = random.uniform(-0.002, 0.005)
            
            # Mettre à jour
            job['progress'] = progress
            job['current_trial'] = trial
            job['eta_minutes'] = eta
            job['best_r2'] = min(0.9, job['best_r2'] + r2_improvement)
            job['current_gap'] = max(0.01, min(0.08, job.get('current_gap', 0.05) + gap_change))
            job['updated_at'] = datetime.now(timezone.utc).isoformat()
            
            # Terminer si 100%
            if progress >= 100:
                job['status'] = 'completed'
                job['completed_at'] = datetime.now(timezone.utc).isoformat()
                job['final_gap'] = job['current_gap']
            
            # Sauvegarder
            container.replace_item(item=job['id'], body=job)
            
            # Attendre avant la prochaine itération
            await asyncio.sleep(3)  # 3 secondes par trial pour la demo
            
            # Re-récupérer le job pour vérifier s'il a été arrêté
            items = list(container.query_items(query=query, enable_cross_partition_query=True))
            if items:
                job = items[0]
        
        logger.info(f"Simulation de training terminée pour {job_id}")
        
    except Exception as e:
        logger.error(f"Erreur dans la simulation du training {job_id}: {e}")

# Fonction de santé
async def training_jobs_health():
    """Vérifie la santé des training jobs"""
    try:
        container = get_cosmos_container()
        # Test simple de connexion
        list(container.query_items(query="SELECT VALUE COUNT(1) FROM c", enable_cross_partition_query=True))
        
        return {
            "status": "healthy",
            "message": "Training Jobs service opérationnel",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        return {
            "status": "degraded",
            "message": f"Service partiellement disponible (mode demo): {str(e)}",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
