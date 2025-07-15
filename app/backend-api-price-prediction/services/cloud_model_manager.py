# Enhanced Model Manager with Cloud Integration
# This extends your existing model_manager.py with cloud capabilities

import os
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# Azure imports (optional)
try:
    from azure.storage.blob import BlobServiceClient
    from azure.cosmos import CosmosClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

logger = logging.getLogger(__name__)


class CloudModelManager:
    """Enhanced model manager with cloud synchronization"""
    
    def __init__(self, local_models_dir: str = "ml_models"):
        self.local_models_dir = Path(local_models_dir)
        self.local_models_dir.mkdir(exist_ok=True)
        
        # Cloud clients
        self.blob_client = None
        self.cosmos_client = None
        self._init_cloud_clients()
        
        # Model cache
        self._models_cache = {}
        self._metadata_cache = {}
        
        self.refresh_registry()
    
    def _init_cloud_clients(self):
        """Initialize Azure clients if credentials available"""
        if not AZURE_AVAILABLE:
            logger.info("Azure SDK not available - running in local mode")
            return
            
        storage_conn = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        cosmos_conn = os.getenv('AZURE_COSMOS_CONNECTION_STRING')
        
        if storage_conn:
            try:
                self.blob_client = BlobServiceClient.from_connection_string(storage_conn)
                logger.info("✅ Connected to Azure Blob Storage")
            except Exception as e:
                logger.warning(f"Failed to connect to Blob Storage: {e}")
        
        if cosmos_conn:
            try:
                self.cosmos_client = CosmosClient.from_connection_string(cosmos_conn)
                logger.info("✅ Connected to Azure Cosmos DB")
            except Exception as e:
                logger.warning(f"Failed to connect to Cosmos DB: {e}")
    
    def refresh_registry(self):
        """Refresh model registry from local and cloud sources"""
        logger.info("🔄 Refreshing model registry...")
        
        # Load local models
        local_models = self._scan_local_models()
        
        # Sync with cloud if available
        cloud_models = {}
        if self.cosmos_client:
            cloud_models = self._fetch_cloud_models()
        
        # Merge local and cloud model information
        self._metadata_cache = {**local_models, **cloud_models}
        
        logger.info(f"📦 Registry updated: {len(self._metadata_cache)} models found")
    
    def _scan_local_models(self) -> Dict[str, Dict]:
        """Scan local directory for models"""
        local_models = {}
        
        for pkl_file in self.local_models_dir.glob("*.pkl"):
            model_name = pkl_file.stem
            
            # Load metadata if exists
            metadata_file = pkl_file.with_suffix('_metadata.json')
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            else:
                # Create basic metadata
                metadata = {
                    "name": model_name,
                    "version": model_name,
                    "created_date": datetime.fromtimestamp(pkl_file.stat().st_mtime).isoformat(),
                    "source": "local",
                    "status": "unknown"
                }
            
            metadata["local_path"] = str(pkl_file)
            metadata["source"] = "local"
            local_models[model_name] = metadata
        
        return local_models
    
    def _fetch_cloud_models(self) -> Dict[str, Dict]:
        """Fetch model metadata from Cosmos DB"""
        cloud_models = {}
        
        try:
            database = self.cosmos_client.get_database_client("realestate")
            container = database.get_container_client("model_registry")
            
            models = list(container.query_items(
                query="SELECT * FROM c ORDER BY c.created_at DESC",
                enable_cross_partition_query=True
            ))
            
            for model in models:
                model_id = model["id"]
                model["source"] = "cloud"
                cloud_models[model_id] = model
                
        except Exception as e:
            logger.error(f"Failed to fetch cloud models: {e}")
        
        return cloud_models
    
    def get_models_list(self) -> List[Dict[str, Any]]:
        """Get comprehensive list of all models"""
        models_list = []
        
        for model_id, metadata in self._metadata_cache.items():
            # Determine availability
            has_local = "local_path" in metadata and Path(metadata["local_path"]).exists()
            has_cloud = metadata.get("source") == "cloud" and self.blob_client is not None
            
            model_info = {
                "id": model_id,
                "name": metadata.get("name", model_id),
                "version": metadata.get("version", model_id),
                "created_date": metadata.get("created_date", "unknown"),
                "status": metadata.get("status", "unknown"),
                "metrics": metadata.get("performance", metadata.get("metrics", {})),
                "source": metadata.get("source", "unknown"),
                "has_local_copy": has_local,
                "has_cloud_copy": has_cloud,
                "is_loaded": model_id in self._models_cache,
                "deployment_ready": metadata.get("deployment_ready", False)
            }
            
            models_list.append(model_info)
        
        # Sort by creation date
        models_list.sort(key=lambda x: x["created_date"], reverse=True)
        return models_list
    
    def load_model(self, model_id: str):
        """Load model from cache, local file, or cloud"""
        
        # Return cached model if available
        if model_id in self._models_cache:
            return self._models_cache[model_id]
        
        if model_id not in self._metadata_cache:
            raise ValueError(f"Model {model_id} not found in registry")
        
        metadata = self._metadata_cache[model_id]
        
        # Try loading from local file first
        if "local_path" in metadata:
            local_path = Path(metadata["local_path"])
            if local_path.exists():
                try:
                    with open(local_path, 'rb') as f:
                        model = pickle.load(f)
                    self._models_cache[model_id] = model
                    logger.info(f"📦 Loaded model {model_id} from local file")
                    return model
                except Exception as e:
                    logger.error(f"Failed to load local model {model_id}: {e}")
        
        # Try downloading from cloud
        if self.blob_client and metadata.get("source") == "cloud":
            model = self._download_from_cloud(model_id)
            if model:
                return model
        
        raise RuntimeError(f"Cannot load model {model_id}")
    
    def _download_from_cloud(self, model_id: str):
        """Download model from Azure Blob Storage"""
        
        try:
            blob_path = f"models/{model_id}/model.pkl"
            
            # Download model
            blob_data = self.blob_client.download_blob(
                container="models",
                blob=blob_path
            ).readall()
            
            # Load model
            model = pickle.loads(blob_data)
            
            # Cache in memory
            self._models_cache[model_id] = model
            
            # Save locally for future use
            local_path = self.local_models_dir / f"{model_id}.pkl"
            with open(local_path, 'wb') as f:
                f.write(blob_data)
            
            # Update metadata
            self._metadata_cache[model_id]["local_path"] = str(local_path)
            
            logger.info(f"☁️ Downloaded model {model_id} from cloud")
            return model
            
        except Exception as e:
            logger.error(f"Failed to download model {model_id}: {e}")
            return None
    
    def get_production_model(self):
        """Get current production model"""
        
        # Find production model
        production_models = [
            model_id for model_id, metadata in self._metadata_cache.items()
            if metadata.get("status") == "production"
        ]
        
        if production_models:
            return self.load_model(production_models[0])
        
        # Fallback: get best model by R²
        best_model = self.get_best_model("r2_score")
        if best_model:
            return self.load_model(best_model["id"])
        
        # Final fallback: any available model
        if self._metadata_cache:
            first_model = next(iter(self._metadata_cache))
            return self.load_model(first_model)
        
        return None
    
    def get_best_model(self, metric: str = "r2_score") -> Optional[Dict]:
        """Get model with best performance for given metric"""
        
        best_model = None
        best_score = float('-inf') if metric in ['r2_score', 'accuracy'] else float('inf')
        
        for model_id, metadata in self._metadata_cache.items():
            metrics = metadata.get("performance", metadata.get("metrics", {}))
            
            if metric in metrics:
                score = metrics[metric]
                
                if metric in ['r2_score', 'accuracy']:
                    # Higher is better
                    if score > best_score:
                        best_score = score
                        best_model = {"id": model_id, "score": score, **metadata}
                else:
                    # Lower is better (mae, rmse, etc.)
                    if score < best_score:
                        best_score = score
                        best_model = {"id": model_id, "score": score, **metadata}
        
        return best_model
    
    async def promote_to_production(self, model_id: str) -> bool:
        """Promote model to production status"""
        
        if model_id not in self._metadata_cache:
            return False
        
        try:
            # Update local metadata
            for mid, metadata in self._metadata_cache.items():
                if mid == model_id:
                    metadata["status"] = "production"
                    metadata["promoted_at"] = datetime.now().isoformat()
                elif metadata.get("status") == "production":
                    metadata["status"] = "rollback_ready"
            
            # Update cloud if available
            if self.cosmos_client:
                await self._update_cloud_status(model_id, "production")
            
            logger.info(f"🚀 Promoted model {model_id} to production")
            return True
            
        except Exception as e:
            logger.error(f"Failed to promote model {model_id}: {e}")
            return False
    
    async def _update_cloud_status(self, model_id: str, status: str):
        """Update model status in Cosmos DB"""
        
        try:
            database = self.cosmos_client.get_database_client("realestate")
            container = database.get_container_client("model_registry")
            
            # Read current document
            doc = container.read_item(item=model_id, partition_key=model_id)
            
            # Update status
            doc["status"] = status
            doc["last_updated"] = datetime.now().isoformat()
            
            # Replace document
            container.replace_item(item=model_id, body=doc)
            
        except Exception as e:
            logger.error(f"Failed to update cloud status: {e}")
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Get detailed information about a specific model"""
        
        if model_id not in self._metadata_cache:
            return None
        
        metadata = self._metadata_cache[model_id]
        
        # Add runtime information
        info = dict(metadata)
        info.update({
            "is_loaded": model_id in self._models_cache,
            "has_local_copy": "local_path" in metadata and Path(metadata["local_path"]).exists(),
            "has_cloud_copy": metadata.get("source") == "cloud",
            "cloud_available": self.blob_client is not None,
            "last_accessed": None  # Could track this
        })
        
        return info


# Create global instance for backward compatibility
cloud_model_manager = CloudModelManager()

# For your existing API to use
def get_model_registry():
    """Get the cloud-enabled model registry"""
    return cloud_model_manager
