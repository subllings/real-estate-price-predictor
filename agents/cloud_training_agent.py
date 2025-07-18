"""
Enhanced Training Agent with Cloud Integration
Bridges the gap between local training and cloud deployment
"""

import os
import json
import pickle
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration des logs Azure (doit être fait avant les imports Azure)
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from utils.configure_logging import configure_azure_logging

# Azure imports (to be installed)
try:
    from azure.storage.blob import BlobServiceClient
    from azure.cosmos import CosmosClient
    AZURE_AVAILABLE = True
except ImportError:
    print("⚠️  Azure SDK not installed. Install with: pip install azure-storage-blob azure-cosmos")
    AZURE_AVAILABLE = False

# ML imports
try:
    import optuna
    from catboost import CatBoostRegressor
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
    ML_AVAILABLE = True
except ImportError:
    print("⚠️  ML packages not installed. Install with: pip install optuna catboost scikit-learn")
    ML_AVAILABLE = False


class CloudTrainingAgent:
    """Enhanced training agent with cloud integration"""
    
    def __init__(self, config_path: str = "configs/training_config.json"):
        self.config = self._load_config(config_path)
        self.local_models_dir = Path("ml_models")
        self.local_models_dir.mkdir(exist_ok=True)
        
        # Cloud clients (initialize only if credentials available)
        self.blob_client = None
        self.cosmos_client = None
        self._init_cloud_clients()
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load training configuration"""
        default_config = {
            "optuna": {
                "n_trials": 100,
                "direction": "maximize",
                "study_name": "real_estate_optimization"
            },
            "model": {
                "type": "catboost",
                "cv_folds": 5,
                "test_size": 0.2
            },
            "cloud": {
                "enable_blob_storage": True,
                "enable_cosmos_logging": True,
                "production_threshold": 0.84
            },
            "azure": {
                "storage_container": "models",
                "cosmos_database": "realestate",
                "cosmos_container": "model_registry"
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        else:
            # Create default config file
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
                
        return default_config
    
    def _init_cloud_clients(self):
        """Initialize Azure clients if credentials available"""
        if not AZURE_AVAILABLE:
            return
            
        # Try to get connection strings from environment
        storage_conn = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        cosmos_conn = os.getenv('AZURE_COSMOS_CONNECTION_STRING')
        
        if storage_conn:
            try:
                self.blob_client = BlobServiceClient.from_connection_string(storage_conn)
                print("✅ Azure Blob Storage connected")
            except Exception as e:
                print(f"⚠️  Blob Storage connection failed: {e}")
        
        if cosmos_conn:
            try:
                self.cosmos_client = CosmosClient.from_connection_string(cosmos_conn)
                print("✅ Azure Cosmos DB connected")
            except Exception as e:
                print(f"⚠️  Cosmos DB connection failed: {e}")
    
    async def train_with_cloud_logging(self, X_train, y_train, X_val, y_val, feature_names):
        """Train model with comprehensive cloud logging"""
        
        print("🚀 Starting training with cloud integration...")
        
        # 1. Create Optuna study
        if AZURE_AVAILABLE and self.cosmos_client:
            # TODO: Use cloud-backed storage for Optuna
            study = optuna.create_study(
                direction=self.config["optuna"]["direction"],
                study_name=f"{self.config['optuna']['study_name']}_{datetime.now().strftime('%Y%m%d')}"
            )
        else:
            study = optuna.create_study(direction=self.config["optuna"]["direction"])
        
        # 2. Define objective with logging
        def objective(trial):
            return self._objective_with_logging(trial, X_train, y_train, X_val, y_val, feature_names)
        
        # 3. Run optimization
        study.optimize(objective, n_trials=self.config["optuna"]["n_trials"])
        
        # 4. Train best model
        best_model = self._train_best_model(study.best_params, X_train, y_train, feature_names)
        
        # 5. Generate comprehensive metadata
        metadata = self._generate_model_metadata(best_model, study, X_val, y_val, feature_names)
        
        # 6. Save locally first
        local_version = self._save_local_model(best_model, metadata, feature_names)
        
        # 7. Upload to cloud if available
        if self.blob_client and self.config["cloud"]["enable_blob_storage"]:
            cloud_version = await self._upload_to_cloud(best_model, metadata, feature_names, study)
        else:
            cloud_version = local_version
            print("ℹ️  Cloud storage not available, model saved locally only")
        
        # 8. Register in model registry
        if self.cosmos_client and self.config["cloud"]["enable_cosmos_logging"]:
            await self._register_model(cloud_version, metadata)
        else:
            print("ℹ️  Model registry not available, model metadata saved locally only")
        
        print(f"✅ Training completed! Model version: {cloud_version}")
        print(f"📊 Best R²: {metadata['performance']['r2_score']:.4f}")
        print(f"💰 Best MAE: €{metadata['performance']['mae']:.0f}")
        
        return {
            "version": cloud_version,
            "model": best_model,
            "metadata": metadata,
            "local_path": self.local_models_dir / f"{local_version}.pkl"
        }
    
    def _objective_with_logging(self, trial, X_train, y_train, X_val, y_val, feature_names):
        """Optuna objective with detailed logging"""
        
        # Suggest hyperparameters
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 10), 
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'random_seed': 42
        }
        
        # Train model
        model = CatBoostRegressor(**params, verbose=False)
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_val)
        r2 = r2_score(y_val, predictions)
        mae = mean_absolute_error(y_val, predictions)
        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        
        # Log trial details locally (since cloud MLflow isn't set up yet)
        trial_log = {
            "trial_number": trial.number,
            "params": params,
            "metrics": {"r2": r2, "mae": mae, "rmse": rmse},
            "timestamp": datetime.now().isoformat()
        }
        
        # Save trial log locally
        trials_dir = self.local_models_dir / "trials"
        trials_dir.mkdir(exist_ok=True)
        
        with open(trials_dir / f"trial_{trial.number:04d}.json", 'w') as f:
            json.dump(trial_log, f, indent=2)
        
        return r2
    
    def _train_best_model(self, best_params, X_train, y_train, feature_names):
        """Train final model with best parameters"""
        
        model = CatBoostRegressor(**best_params, verbose=False)
        model.fit(X_train, y_train)
        
        return model
    
    def _generate_model_metadata(self, model, study, X_val, y_val, feature_names):
        """Generate comprehensive model metadata"""
        
        predictions = model.predict(X_val)
        
        # Performance metrics
        performance = {
            "r2_score": float(r2_score(y_val, predictions)),
            "mae": float(mean_absolute_error(y_val, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y_val, predictions))),
            "validation_samples": len(y_val)
        }
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(feature_names, model.feature_importances_.tolist()))
        else:
            feature_importance = {}
        
        # Optuna study info
        study_info = {
            "n_trials": len(study.trials),
            "best_trial": study.best_trial.number,
            "best_params": study.best_params,
            "study_name": study.study_name
        }
        
        metadata = {
            "model_name": "real_estate_predictor",
            "model_type": "catboost",
            "version": f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "created_at": datetime.now().isoformat(),
            "performance": performance,
            "feature_importance": feature_importance,
            "features": feature_names,
            "n_features": len(feature_names),
            "optuna_study": study_info,
            "status": "candidate",
            "deployment_ready": performance["r2_score"] >= self.config["cloud"]["production_threshold"]
        }
        
        return metadata
    
    def _save_local_model(self, model, metadata, feature_names):
        """Save model locally with metadata"""
        
        version = metadata["version"]
        
        # Save model
        model_path = self.local_models_dir / f"{version}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata_path = self.local_models_dir / f"{version}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save features
        features_path = self.local_models_dir / f"{version}_features.json"
        with open(features_path, 'w') as f:
            json.dump({"features": feature_names}, f, indent=2)
        
        print(f"💾 Model saved locally: {model_path}")
        
        return version
    
    async def _upload_to_cloud(self, model, metadata, feature_names, study):
        """Upload model artifacts to Azure Blob Storage"""
        
        if not self.blob_client:
            return metadata["version"]
        
        version = metadata["version"]
        container = self.config["azure"]["storage_container"]
        
        try:
            # Ensure container exists
            try:
                self.blob_client.create_container(container)
            except:
                pass  # Container might already exist

            # Get container client
            container_client = self.blob_client.get_container_client(container)

            # Upload model binary
            model_blob = f"models/{version}/model.pkl"
            model_bytes = pickle.dumps(model)
            container_client.upload_blob(
                name=model_blob,
                data=model_bytes,
                overwrite=True
            )

            # Upload metadata
            metadata_blob = f"models/{version}/metadata.json"
            metadata_enhanced = {
                **metadata,
                "cloud_artifacts": {
                    "model_blob": model_blob,
                    "metadata_blob": metadata_blob,
                    "features_blob": f"models/{version}/features.json",
                    "trials_blob": f"models/{version}/trials.json"
                }
            }

            container_client.upload_blob(
                name=metadata_blob,
                data=json.dumps(metadata_enhanced, indent=2),
                overwrite=True
            )

            # Upload features
            features_blob = f"models/{version}/features.json"
            container_client.upload_blob(
                name=features_blob,
                data=json.dumps({"features": feature_names}),
                overwrite=True
            )

            # Upload trials data
            trials_blob = f"models/{version}/trials.json"
            trials_data = {
                "study_name": study.study_name,
                "n_trials": len(study.trials),
                "best_params": study.best_params,
                "best_value": study.best_value,
                "trials": [
                    {
                        "number": trial.number,
                        "value": trial.value,
                        "params": trial.params,
                        "state": trial.state.name
                    }
                    for trial in study.trials
                ]
            }

            container_client.upload_blob(
                name=trials_blob,
                data=json.dumps(trials_data, indent=2),
                overwrite=True
            )
            
            print(f"☁️  Model artifacts uploaded to Azure Blob Storage")
            return version
            
        except Exception as e:
            print(f"⚠️  Failed to upload to cloud: {e}")
            return metadata["version"]
    
    async def _register_model(self, version, metadata):
        """Register model in Cosmos DB registry"""
        
        if not self.cosmos_client:
            return
        
        try:
            database = self.cosmos_client.get_database_client(self.config["azure"]["cosmos_database"])
            container = database.get_container_client(self.config["azure"]["cosmos_container"])
            
            # Create model record
            model_record = {
                "id": version,
                "partition_key": version,
                **metadata,
                "registered_at": datetime.now().isoformat()
            }
            
            container.create_item(model_record)
            print(f"📝 Model registered in Cosmos DB: {version}")
            
        except Exception as e:
            print(f"⚠️  Failed to register model: {e}")
    
    def _upload_to_cloud_sync(self, model, metadata, feature_names, study):
        """Version synchrone de l'upload vers Azure"""
        
        if not self.blob_client:
            return metadata["version"]
        
        version = metadata["version"]
        container = self.config["azure"]["storage_container"]
        
        try:
            # Ensure container exists
            try:
                self.blob_client.create_container(container)
            except:
                pass  # Container might already exist
            
            # Upload model binary
            model_blob = f"models/{version}/model.pkl"
            model_bytes = pickle.dumps(model)
            blob_client = self.blob_client.get_blob_client(container=container, blob=model_blob)
            blob_client.upload_blob(
                data=model_bytes,
                overwrite=True
            )
            
            # Upload metadata
            metadata_blob = f"models/{version}/metadata.json"
            metadata_enhanced = {
                **metadata,
                "cloud_artifacts": {
                    "model_blob": model_blob,
                    "metadata_blob": metadata_blob,
                    "features_blob": f"models/{version}/features.json"
                }
            }
            
            blob_client = self.blob_client.get_blob_client(container=container, blob=metadata_blob)
            blob_client.upload_blob(
                data=json.dumps(metadata_enhanced, indent=2),
                overwrite=True
            )
            
            # Upload features
            features_blob = f"models/{version}/features.json"
            blob_client = self.blob_client.get_blob_client(container=container, blob=features_blob)
            blob_client.upload_blob(
                data=json.dumps({"features": feature_names}),
                overwrite=True
            )
            
            print(f"☁️  Model artifacts uploaded to Azure Blob Storage: {version}")
            return version
            
        except Exception as e:
            print(f"⚠️  Failed to upload to cloud: {e}")
            return metadata["version"]
    
    def _register_model_sync(self, version, metadata):
        """Version synchrone de l'enregistrement dans CosmosDB"""
        
        if not self.cosmos_client:
            return
        
        try:
            database = self.cosmos_client.get_database_client(self.config["azure"]["cosmos_database"])
            container = database.get_container_client(self.config["azure"]["cosmos_container"])
            
            # Create model record
            model_record = {
                "id": version,
                "partition_key": version,
                **metadata,
                "registered_at": datetime.now().isoformat()
            }
            
            container.create_item(model_record)
            print(f"📝 Model registered in Cosmos DB: {version}")
            
        except Exception as e:
            print(f"⚠️  Failed to register model: {e}")


# Usage example and test function
async def test_training_agent():
    """Test the training agent with sample data"""
    
    if not ML_AVAILABLE:
        print("❌ ML packages not available. Please install requirements.")
        return
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    
    # Split data
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Create training agent
    agent = CloudTrainingAgent()
    
    # Run training
    result = await agent.train_with_cloud_logging(
        X_train, y_train, X_val, y_val, feature_names
    )
    
    print("\n🎉 Training completed!")
    print(f"Model version: {result['version']}")
    print(f"Local path: {result['local_path']}")


if __name__ == "__main__":
    # Run test
    asyncio.run(test_training_agent())
