# services/model_manager.py
import os
import json
import joblib
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd

logger = logging.getLogger(__name__)

class ModelRegistry:
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.pkl_dir = self.models_dir / "pkl"
        self.features_dir = self.models_dir / "features"
        self.metrics_dir = self.models_dir / "metrics"
        
        # Cache des modèles chargés
        self._loaded_models: Dict[str, Any] = {}
        self._model_metadata: Dict[str, Dict] = {}
        
        self.refresh_registry()
    
    def refresh_registry(self):
        """Scan le dossier models et met à jour le registre"""
        self._model_metadata = {}
        
        if not self.pkl_dir.exists():
            logger.warning(f"Models directory {self.pkl_dir} does not exist")
            return
        
        for pkl_file in self.pkl_dir.glob("*.pkl"):
            try:
                model_info = self._extract_model_info(pkl_file)
                self._model_metadata[model_info["model_id"]] = model_info
                logger.info(f"Registered model: {model_info['model_id']}")
            except Exception as e:
                logger.error(f"Failed to register model {pkl_file}: {e}")
    
    def _extract_model_info(self, pkl_file: Path) -> Dict:
        """Extrait les métadonnées d'un fichier de modèle"""
        model_name = pkl_file.stem
        
        # Chercher les fichiers associés
        features_file = self.features_dir / f"{model_name}.json"
        metrics_file = self.metrics_dir / f"{model_name}_metrics.json"
        
        model_info = {
            "model_id": model_name,
            "name": model_name,
            "pkl_path": str(pkl_file),
            "created_at": datetime.fromtimestamp(pkl_file.stat().st_mtime).isoformat(),
            "file_size": pkl_file.stat().st_size,
            "status": "available"
        }
        
        # Charger les features si disponibles
        if features_file.exists():
            with open(features_file, 'r') as f:
                features_data = json.load(f)
                model_info["features"] = features_data
                model_info["feature_count"] = len(features_data)
        
        # Charger les métriques si disponibles
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
                model_info["metrics"] = metrics_data
                
                # Extraire les métriques principales
                if "mae" in metrics_data:
                    model_info["mae"] = metrics_data["mae"]
                if "rmse" in metrics_data:
                    model_info["rmse"] = metrics_data["rmse"]
                if "r2" in metrics_data:
                    model_info["r2"] = metrics_data["r2"]
        
        # Déterminer le type de modèle
        if "catboost" in model_name.lower():
            model_info["type"] = "CatBoost"
        elif "xgboost" in model_name.lower():
            model_info["type"] = "XGBoost"
        else:
            model_info["type"] = "Unknown"
            
        # Déterminer la variante (all features vs top30)
        if "top" in model_name.lower() or "30" in model_name:
            model_info["variant"] = "top_features"
        else:
            model_info["variant"] = "all_features"
        
        return model_info
    
    def list_models(self) -> List[Dict]:
        """Retourne la liste de tous les modèles disponibles"""
        return list(self._model_metadata.values())
    
    def get_model_info(self, model_id: str) -> Optional[Dict]:
        """Retourne les infos d'un modèle spécifique"""
        return self._model_metadata.get(model_id)
    
    def load_model(self, model_id: str):
        """Charge un modèle en mémoire"""
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]
        
        model_info = self.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found in registry")
        
        try:
            model = joblib.load(model_info["pkl_path"])
            self._loaded_models[model_id] = model
            logger.info(f"Loaded model: {model_id}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise
    
    def get_best_model(self, metric: str = "r2") -> Optional[str]:
        """Retourne l'ID du meilleur modèle selon une métrique"""
        best_model = None
        best_score = -float('inf') if metric == "r2" else float('inf')
        
        for model_id, info in self._model_metadata.items():
            if metric in info:
                score = info[metric]
                
                if metric == "r2" and score > best_score:
                    best_score = score
                    best_model = model_id
                elif metric in ["mae", "rmse"] and score < best_score:
                    best_score = score
                    best_model = model_id
        
        return best_model
    
    def set_production_model(self, model_id: str, variant: str = "all_features"):
        """Marque un modèle comme étant en production"""
        if model_id not in self._model_metadata:
            raise ValueError(f"Model {model_id} not found")
        
        # Reset tous les autres modèles
        for info in self._model_metadata.values():
            info["status"] = "available"
        
        # Marquer le modèle sélectionné comme production
        self._model_metadata[model_id]["status"] = "production"
        self._model_metadata[model_id]["promoted_at"] = datetime.now().isoformat()
        
        logger.info(f"Set model {model_id} as production model")

# Instance globale
model_registry = ModelRegistry()
