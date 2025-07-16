"""
Middleware FastAPI pour intégration automatique des modèles Azure
À ajouter dans votre FastAPI main.py
"""

import os
import json
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.base import BaseHTTPMiddleware
from utils.azure_model_storage import ensure_best_model_available

class ModelSyncMiddleware(BaseHTTPMiddleware):
    """
    Middleware pour synchroniser automatiquement le meilleur modèle depuis Azure
    """
    
    def __init__(self, app: FastAPI, check_interval: int = 3600):
        super().__init__(app)
        self.check_interval = check_interval  # Vérification toutes les heures
        self.last_check = 0
        self.current_model_path = "models/current_best_model.pkl"
        self.model_metadata_path = "models/current_best_model_metadata.json"
        
    async def dispatch(self, request, call_next):
        # Vérifier si il faut synchroniser le modèle
        current_time = datetime.now().timestamp()
        
        if current_time - self.last_check > self.check_interval:
            await self._sync_model_if_needed()
            self.last_check = current_time
        
        response = await call_next(request)
        return response
    
    async def _sync_model_if_needed(self):
        """Synchroniser le modèle depuis Azure si nécessaire"""
        try:
            success = ensure_best_model_available(self.current_model_path)
            if success:
                print(f"[✅] Modèle synchronisé depuis Azure: {self.current_model_path}")
            else:
                print(f"[⚠️] Échec synchronisation Azure, utilisation modèle local")
        except Exception as e:
            print(f"[❌] Erreur sync modèle: {e}")


# Endpoints à ajouter à votre FastAPI
def add_model_endpoints(app: FastAPI):
    """
    Ajouter les endpoints de gestion des modèles à votre FastAPI
    """
    
    @app.get("/model/info")
    async def get_model_info():
        """Récupérer les informations du modèle actuellement utilisé"""
        try:
            metadata_path = "models/current_best_model_metadata.json"
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                return {
                    "status": "active",
                    "model_name": metadata.get("model_name", "Unknown"),
                    "r2_test": metadata.get("r2_test", 0),
                    "mae_test": metadata.get("mae_test", 0),
                    "rmse_test": metadata.get("rmse_test", 0),
                    "n_features": metadata.get("n_features", 0),
                    "upload_timestamp": metadata.get("upload_timestamp"),
                    "azure_url": metadata.get("azure_blob_name", "local")
                }
            else:
                return {
                    "status": "no_metadata",
                    "message": "Modèle disponible mais métadonnées manquantes"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Erreur récupération info modèle: {str(e)}"
            }
    
    @app.post("/model/sync")
    async def force_model_sync(background_tasks: BackgroundTasks):
        """Forcer la synchronisation du modèle depuis Azure"""
        def sync_task():
            try:
                success = ensure_best_model_available("models/current_best_model.pkl")
                print(f"[SYNC] Synchronisation forcée: {'✅ Succès' if success else '❌ Échec'}")
            except Exception as e:
                print(f"[SYNC] Erreur: {e}")
        
        background_tasks.add_task(sync_task)
        return {"message": "Synchronisation modèle lancée en arrière-plan"}
    
    @app.get("/model/health")
    async def model_health_check():
        """Vérifier l'état du modèle"""
        model_path = "models/current_best_model.pkl"
        metadata_path = "models/current_best_model_metadata.json"
        
        checks = {
            "model_file_exists": os.path.exists(model_path),
            "metadata_exists": os.path.exists(metadata_path),
            "model_age_hours": 0,
            "azure_connection": False
        }
        
        if checks["model_file_exists"]:
            file_age = datetime.now().timestamp() - os.path.getmtime(model_path)
            checks["model_age_hours"] = round(file_age / 3600, 1)
        
        try:
            from utils.azure_model_storage import AzureModelStorage
            storage = AzureModelStorage()
            best_model = storage.get_best_model_info()
            checks["azure_connection"] = best_model is not None
        except Exception:
            checks["azure_connection"] = False
        
        # Déterminer le statut global
        if all([checks["model_file_exists"], checks["metadata_exists"]]):
            if checks["model_age_hours"] < 24:
                status = "healthy"
            else:
                status = "outdated"
        else:
            status = "unhealthy"
        
        return {
            "status": status,
            "checks": checks,
            "timestamp": datetime.now().isoformat()
        }


# Code d'initialisation pour votre main.py
def setup_model_integration(app: FastAPI):
    """
    Configuration complète de l'intégration modèles pour FastAPI
    """
    # Ajouter le middleware de synchronisation
    app.add_middleware(ModelSyncMiddleware, check_interval=3600)  # 1h
    
    # Ajouter les endpoints
    add_model_endpoints(app)
    
    # Synchronisation initiale au démarrage
    @app.on_event("startup")
    async def startup_model_sync():
        print("[STARTUP] Synchronisation initiale du modèle...")
        try:
            success = ensure_best_model_available("models/current_best_model.pkl")
            if success:
                print("[✅] Modèle prêt pour FastAPI")
            else:
                print("[⚠️] Modèle local utilisé (Azure indisponible)")
        except Exception as e:
            print(f"[❌] Erreur startup sync: {e}")
    
    print("[✅] Intégration Azure modèles configurée pour FastAPI")


# Exemple d'utilisation dans main.py:
"""
from fastapi import FastAPI
from utils.fastapi_model_integration import setup_model_integration

app = FastAPI(title="Real Estate Price Prediction API")

# Configuration automatique de l'intégration modèles
setup_model_integration(app)

# Vos autres routes...
"""
