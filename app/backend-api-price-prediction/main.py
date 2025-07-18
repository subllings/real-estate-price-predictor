from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List
import logging

from services.model_manager import model_registry

# Initialize FastAPI app
app = FastAPI(title="Real Estate Price Prediction API", version="2.0.0")

# Configure logging (important pour afficher le contenu reçu)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable CORS for frontend on localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React local dev
        "https://realestate-react-ui.azurewebsites.net",  # React in production Azure
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Smart model loading - use available models or fallback
try:
    available_models = model_registry.list_models()
    if available_models:
        # Try to find best models for each variant
        best_all_features = None
        best_top_features = None
        
        for model in available_models:
            if model["variant"] == "all_features":
                if not best_all_features or (model.get("r2", 0) > best_all_features.get("r2", 0)):
                    best_all_features = model
            elif model["variant"] == "top_features":
                if not best_top_features or (model.get("r2", 0) > best_top_features.get("r2", 0)):
                    best_top_features = model
        
        # Load the best available models
        if best_all_features:
            model_all = model_registry.load_model(best_all_features["model_id"])
            model_all_info = best_all_features
            logger.info(f"Loaded best all-features model: {best_all_features['model_id']}")
        else:
            logger.warning("No all-features model found")
            model_all = None
            model_all_info = None
        
        if best_top_features:
            model_top30 = model_registry.load_model(best_top_features["model_id"])
            model_top30_info = best_top_features
            logger.info(f"Loaded best top-features model: {best_top_features['model_id']}")
        else:
            logger.warning("No top-features model found")
            model_top30 = None
            model_top30_info = None
    else:
        logger.error("No models found in registry")
        model_all = model_top30 = None
        model_all_info = model_top30_info = None
        
except Exception as e:
    logger.error(f"Failed to load models from registry: {e}")
    # Fallback to hardcoded paths if available
    MODEL_ALL_PATH = "models/pkl/catboost_optuna_all_20250703_0914.pkl"
    MODEL_TOP30_PATH = "models/pkl/catboost_optuna_top30_20250703_0914.pkl"
    
    try:
        with open(MODEL_ALL_PATH, "rb") as f:
            model_all = joblib.load(f)
        with open(MODEL_TOP30_PATH, "rb") as f:
            model_top30 = joblib.load(f)
        model_all_info = {"model_id": "fallback_all", "name": "Fallback All Features"}
        model_top30_info = {"model_id": "fallback_top30", "name": "Fallback Top 30"}
        logger.info("Loaded fallback models")
    except Exception as fallback_error:
        logger.error(f"Fallback loading also failed: {fallback_error}")
        model_all = model_top30 = None
        model_all_info = model_top30_info = None

print(f"Model loading complete. All-features: {'✓' if model_all else '✗'}, Top-features: {'✓' if model_top30 else '✗'}")

# Input schema for full feature model
class InputDataAll(BaseModel):
    bedroomCount: float
    bathroomCount: float
    postCode: float
    habitableSurface: float
    buildingConstructionYear: float
    facedeCount: float
    toiletCount: float
    room_count: float
    surface_per_room: float
    building_age: float
    type_APARTMENT: float
    type_HOUSE: float
    subtype_APARTMENT: float
    subtype_APARTMENT_BLOCK: float
    subtype_DUPLEX: float
    subtype_GROUND_FLOOR: float
    subtype_HOUSE: float
    subtype_MIXED_USE_BUILDING: float
    subtype_PENTHOUSE: float
    subtype_TOWN_HOUSE: float
    subtype_VILLA: float
    province_Antwerp: float
    province_Brussels: float
    province_East_Flanders: float = Field(..., alias="province_East Flanders")
    province_Flemish_Brabant: float = Field(..., alias="province_Flemish Brabant")
    province_Hainaut: float
    province_Limburg: float
    province_Liège: float = Field(..., alias="province_Liège")
    province_Luxembourg: float
    province_Namur: float
    province_Walloon_Brabant: float = Field(..., alias="province_Walloon Brabant")
    province_West_Flanders: float = Field(..., alias="province_West Flanders")
    locality_Anderlecht: float
    locality_Antwerpen: float
    locality_Bruxelles: float
    locality_Gent: float
    locality_Ixelles: float
    locality_Knokke_Heist: float = Field(..., alias="locality_Knokke-Heist")
    locality_Liège: float = Field(..., alias="locality_Liège")
    locality_Uccle: float
    buildingCondition_AS_NEW: float
    buildingCondition_GOOD: float
    buildingCondition_JUST_RENOVATED: float
    buildingCondition_TO_BE_DONE_UP: float
    buildingCondition_TO_RENOVATE: float
    buildingCondition_nan: float
    floodZoneType_NON_FLOOD_ZONE: float
    floodZoneType_POSSIBLE_FLOOD_ZONE: float
    floodZoneType_RECOGNIZED_FLOOD_ZONE: float
    floodZoneType_nan: float
    heatingType_ELECTRIC: float
    heatingType_FUELOIL: float
    heatingType_GAS: float
    heatingType_PELLET: float
    heatingType_nan: float
    kitchenType_HYPER_EQUIPPED: float
    kitchenType_INSTALLED: float
    kitchenType_NOT_INSTALLED: float
    kitchenType_SEMI_EQUIPPED: float
    kitchenType_USA_HYPER_EQUIPPED: float
    kitchenType_USA_INSTALLED: float
    kitchenType_nan: float
    epcScore_A: float
    epcScore_A_plus: float = Field(..., alias="epcScore_A+")
    epcScore_B: float
    epcScore_C: float
    epcScore_D: float
    epcScore_E: float
    epcScore_F: float
    epcScore_G: float
    hasLivingRoom: float
    hasTerrace: float

    class Config:
        populate_by_name = True


# Input schema for top 30 feature model
class InputDataTop30(BaseModel):
    habitableSurface: float
    bathroomCount: float
    postCode: float
    toiletCount: float
    buildingConstructionYear: float
    locality_Knokke_Heist: float = Field(..., alias="locality_Knokke-Heist")
    building_age: float
    surface_per_room: float
    facedeCount: float
    kitchenType_HYPER_EQUIPPED: float
    buildingCondition_AS_NEW: float
    province_West_Flanders: float = Field(..., alias="province_West Flanders")
    subtype_VILLA: float
    subtype_HOUSE: float
    province_Hainaut: float
    room_count: float
    bedroomCount: float
    buildingCondition_TO_RENOVATE: float
    epcScore_B: float
    hasTerrace: float
    subtype_PENTHOUSE: float
    epcScore_C: float
    buildingCondition_GOOD: float
    heatingType_nan: float
    hasLivingRoom: float
    locality_Ixelles: float
    kitchenType_INSTALLED: float
    epcScore_A: float
    epcScore_F: float
    locality_Gent: float

    class Config:
        populate_by_name = True



# Endpoint for full feature model
"""
@app.post("/predict_all")
def predict_all(data: InputDataAll):
    try:
        input_dict = data.dict(by_alias=True)
        for k, v in input_dict.items():
            if isinstance(v, dict):
                raise ValueError(f"Invalid value for key '{k}': nested dict detected ({v})")
        input_df = pd.DataFrame([input_dict])
        prediction = model_all.predict(input_df)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
"""
@app.post("/predict_all")
async def predict_all(data: Dict[str, Any]):
    if not model_all:
        raise HTTPException(status_code=503, detail="All-features model not available")
        
    try:
        logger.info("----")
        logger.info("Received payload for prediction:")
        logger.info(data)

        # Detect nested dictionaries
        for k, v in data.items():
            if isinstance(v, dict):
                raise ValueError(f"Invalid value for key '{k}': nested dict detected ({v})")

        # Check for missing features
        missing_cols = [col for col in model_all.feature_names_ if col not in data]
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Missing required features",
                    "missing_features": missing_cols
                }
            )

        # Create input DataFrame and reorder columns
        input_df = pd.DataFrame([data])
        input_df = input_df[model_all.feature_names_]
        logger.info(f"DataFrame created with shape {input_df.shape}")

        # Prediction
        prediction = model_all.predict(input_df)
        logger.info("Prediction: %s", prediction)
        return {
            "prediction": float(prediction[0]),
            "model_info": {
                "model_id": model_all_info.get("model_id") if model_all_info else "unknown",
                "model_name": model_all_info.get("name") if model_all_info else "Unknown Model",
                "r2_score": model_all_info.get("r2") if model_all_info else None,
                "mae": model_all_info.get("mae") if model_all_info else None
            }
        }

    except HTTPException as he:
        raise he  # Let FastAPI handle it properly
    except Exception as e:
        logger.exception("Prediction failed:")
        raise HTTPException(status_code=400, detail={"error": "Prediction failed", "message": str(e)})


# Endpoint for top 30 features model
@app.post("/predict_top30")
def predict_top30(data: InputDataTop30):
    if not model_top30:
        raise HTTPException(status_code=503, detail="Top-features model not available")
        
    try:
        input_dict = data.dict(by_alias=True)
        for k, v in input_dict.items():
            if isinstance(v, dict):
                raise ValueError(f"Invalid value for key '{k}': nested dict detected ({v})")
        input_df = pd.DataFrame([input_dict])
        prediction = model_top30.predict(input_df)
        return {
            "prediction": float(prediction[0]),
            "model_info": {
                "model_id": model_top30_info.get("model_id") if model_top30_info else "unknown",
                "model_name": model_top30_info.get("name") if model_top30_info else "Unknown Model",
                "r2_score": model_top30_info.get("r2") if model_top30_info else None,
                "mae": model_top30_info.get("mae") if model_top30_info else None
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/")
def read_root():
    return {"message": "Real Estate Prediction API", "version": "2.0.0", "models_available": len(model_registry.list_models())}

@app.post("/echo")
def echo_input(data: dict):
    return {"received": data}


@app.post("/get_model_parameters")
def get_model_parameters():
    if not model_all:
        raise HTTPException(status_code=503, detail="Model not available")
        
    try:
        parameters = model_all.feature_names_
        logger.info(f"Sending model parameters to agent. Count: {len(parameters)}")
        return {
            "model_name": model_all_info.get("model_id") if model_all_info else "unknown",
            "feature_count": len(parameters),
            "features": parameters,
        }
    except Exception as e:
        logger.exception("Failed to get model parameters")
        raise HTTPException(status_code=500, detail=str(e))

# === NEW MODEL MANAGEMENT ENDPOINTS ===

@app.get("/models")
def list_models():
    """Liste tous les modèles disponibles"""
    try:
        model_registry.refresh_registry()
        models = model_registry.list_models()
        return {
            "models": models,
            "count": len(models)
        }
    except Exception as e:
        logger.exception("Failed to list models")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_id}")
def get_model_details(model_id: str):
    """Obtient les détails d'un modèle spécifique"""
    try:
        model_info = model_registry.get_model_info(model_id)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
        return model_info
    except Exception as e:
        logger.exception(f"Failed to get model details for {model_id}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/{model_id}/promote")
def promote_model(model_id: str, variant: str = "all_features"):
    """Promeut un modèle en production"""
    try:
        model_registry.set_production_model(model_id, variant)
        
        # Recharger les modèles actifs
        global model_all, model_top30, model_all_info, model_top30_info
        
        model_info = model_registry.get_model_info(model_id)
        if model_info["variant"] == "all_features":
            model_all = model_registry.load_model(model_id)
            model_all_info = model_info
        elif model_info["variant"] == "top_features":
            model_top30 = model_registry.load_model(model_id)
            model_top30_info = model_info
        
        return {"message": f"Model {model_id} promoted to production", "variant": variant}
    except Exception as e:
        logger.exception(f"Failed to promote model {model_id}")
        raise HTTPException(status_code=500, detail=str(e))


# === TRAINING EXPERIMENTS ENDPOINTS ===

@app.get("/health")
async def health_check():
    """Point de contrôle de santé de l'API"""
    return {"status": "healthy", "service": "Real Estate Price Prediction API"}


@app.get("/experiments")
async def get_experiments():
    """Récupère tous les expériences de training depuis CosmosDB avec métriques structurées"""
    try:
        import sys
        import os
        
        # Ajouter le répertoire parent au path pour accéder à utils
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        
        from utils.cosmosdb_logger import CosmosDbLogger
        cosmos_logger = CosmosDbLogger()
        
        # Essayer d'abord la nouvelle méthode avec ModelMetrics
        formatted_experiments = []
        try:
            print("🔍 Trying to fetch from ModelMetrics container...")
            
            # Accéder directement au container ModelMetrics
            container = cosmos_logger.database.get_container_client('ModelMetrics')
            query = """
            SELECT TOP 100 * FROM c 
            WHERE c.model_type = 'catboost' 
            ORDER BY c.timestamp DESC
            """
            experiments = list(container.query_items(query=query, enable_cross_partition_query=True))
            
            for exp in experiments:
                formatted_exp = {
                    "id": exp.get("id", ""),
                    "trial_number": exp.get("trial_number", 0),
                    "experiment_name": exp.get("experiment_name", ""),
                    "model_type": exp.get("model_type", "catboost"),
                    "model_name": exp.get("model_name", "CatBoost CV (All Features)"),
                    "timestamp": exp.get("timestamp", ""),
                    
                    # Métriques de performance
                    "r2_train": exp.get("r2_train", 0),
                    "r2_test": exp.get("r2_test", 0),
                    "mae_train": exp.get("mae_train", 0),
                    "mae_test": exp.get("mae_test", 0),
                    "rmse_train": exp.get("rmse_train", 0),
                    "rmse_test": exp.get("rmse_test", 0),
                    
                    # Analyse de généralisation
                    "r2_gap": exp.get("r2_gap", 0),
                    "generalization_status": exp.get("generalization_status", "Unknown"),
                    "feature_count": exp.get("n_features", 2885),
                    
                    # Métadonnées importantes
                    "training_time": exp.get("training_time", 0),
                    "hyperparameters": exp.get("hyperparameters", {}),
                    "feature_importance": exp.get("feature_importance", []),
                    "status": exp.get("status", "completed")
                }
                formatted_experiments.append(formatted_exp)
                
            print(f"✅ Found {len(formatted_experiments)} experiments in ModelMetrics")
            
        except Exception as e:
            print(f"⚠️ ModelMetrics fetch failed: {e}")
            print("🔄 Falling back to legacy container...")
            
            # Fallback vers l'ancienne méthode
            experiments = cosmos_logger.get_trials_for_model("catboost", limit=100)
            
            # Fonction pour calculer le diagnostic de généralisation
            def calculate_generalization_status(r2_train, r2_test):
                if not r2_train or not r2_test:
                    return "Unknown"
                
                r2_gap = r2_train - r2_test
                
                # Logique alignée avec train_test_metrics_logger.py et CatBoost tuner
                if r2_gap < 0:
                    return "Possible underfitting"
                elif r2_gap < 0.05:
                    return "Excellent generalization"
                elif r2_gap < 0.08:
                    return "Good generalization"
                elif r2_gap < 0.12:
                    return "Moderate overfitting"
                else:
                    return "Strong overfitting"
            
            for exp in experiments:
                # Support des métriques structurées ou format legacy
                structured_metrics = exp.get("structured_metrics", {})
                
                # Extraire les valeurs R²
                r2_train = structured_metrics.get("r2_train") or exp.get("r2_score", 0)
                r2_test = structured_metrics.get("r2_test") or exp.get("r2_test", 0)
                
                # Calculer R² gap et diagnostic
                r2_gap = (r2_train - r2_test) if (r2_train and r2_test) else 0
                generalization_status = calculate_generalization_status(r2_train, r2_test)
                
                formatted_exp = {
                    "id": exp.get("id", ""),
                    "trial_number": exp.get("trial_number", 0),
                    "experiment_name": exp.get("experiment_name", ""),
                    "model_type": structured_metrics.get("model_type") or exp.get("model_name", "catboost"),
                    "model_name": structured_metrics.get("model_name") or "CatBoost CV (All Features)",
                    "timestamp": exp.get("timestamp", ""),
                    
                    # Métriques avec support structured_metrics ou format legacy
                    "r2_train": r2_train,
                    "r2_test": r2_test,
                    "mae_train": structured_metrics.get("mae_train") or exp.get("mae", 0),
                    "mae_test": structured_metrics.get("mae_test") or exp.get("mae_test", 0),
                    "rmse_train": structured_metrics.get("rmse_train") or exp.get("rmse", 0),
                    "rmse_test": structured_metrics.get("rmse_test") or exp.get("rmse_test", 0),
                    
                    # Diagnostics calculés dynamiquement
                    "r2_gap": r2_gap,
                    "generalization_status": generalization_status,
                    "feature_count": structured_metrics.get("feature_count") or 2885,
                    
                    # Données supplémentaires
                    "hyperparameters": exp.get("hyperparameters", {}),
                    "feature_importance": exp.get("feature_importance", []),
                    "training_time": exp.get("training_time", 0),
                    "status": exp.get("status", "completed")
                }
                formatted_experiments.append(formatted_exp)
            
            print(f"✅ Found {len(formatted_experiments)} experiments in legacy container")
        
        # Trier par R² test décroissant
        formatted_experiments.sort(key=lambda x: x.get("r2_test", 0), reverse=True)
        
        print(f"📊 Returning {len(formatted_experiments)} experiments to frontend")
        return {"experiments": formatted_experiments}
        
    except Exception as e:
        logger.exception("Failed to fetch experiments")
        raise HTTPException(status_code=500, detail=f"Failed to fetch experiments: {str(e)}")


@app.get("/experiments/summary")
async def get_experiments_summary():
    """Récupère un résumé des expériences de training avec métriques structurées"""
    try:
        import sys
        import os
        
        # Ajouter le répertoire parent au path pour accéder à utils
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        
        from utils.cosmosdb_logger import CosmosDbLogger
        cosmos_logger = CosmosDbLogger()
        
        # Essayer d'abord la nouvelle méthode avec ModelMetrics
        experiments = []
        try:
            print("🔍 Trying to fetch summary from ModelMetrics container...")
            experiments = cosmos_logger.get_model_metrics("catboost", limit=100, container_name="ModelMetrics")
            print(f"✅ Found {len(experiments)} experiments in ModelMetrics for summary")
            
        except Exception as e:
            print(f"⚠️ ModelMetrics fetch failed: {e}")
            print("🔄 Falling back to legacy container for summary...")
            
            # Fallback vers l'ancienne méthode
            experiments = cosmos_logger.get_trials_for_model("catboost", limit=100)
            print(f"✅ Found {len(experiments)} experiments in legacy container for summary")
        
        if not experiments:
            return {
                "total_experiments": 0,
                "best_r2_score": 0,
                "average_r2_score": 0,
                "latest_experiment": None,
                "best_generalization": None,
                "average_r2_gap": 0
            }
        
        # Extraire les R² test avec support des nouvelles métriques
        r2_scores = []
        r2_gaps = []
        
        for exp in experiments:
            # Support format direct ou structured_metrics
            structured_metrics = exp.get("structured_metrics", {})
            
            # R² test
            r2_test = exp.get("r2_test") or structured_metrics.get("r2_test") or exp.get("r2_score", 0)
            if r2_test > 0:
                r2_scores.append(r2_test)
            
            # R² gap
            r2_gap = exp.get("r2_gap") or structured_metrics.get("r2_gap", 0)
            if r2_gap:
                r2_gaps.append(abs(r2_gap))
        
        # Trouver la meilleure expérience
        best_experiment = max(experiments, key=lambda x: 
            x.get("r2_test") or 
            x.get("structured_metrics", {}).get("r2_test") or 
            x.get("r2_score", 0))
        
        # Trouver la dernière expérience
        latest_experiment = max(experiments, key=lambda x: x.get("timestamp", ""))
        
        # Trouver la meilleure généralisation (plus petit gap absolu)
        best_generalization_exp = min(experiments, key=lambda x: 
            abs(x.get("r2_gap") or 
                x.get("structured_metrics", {}).get("r2_gap", 1.0)))
        
        # Préparer les données pour latest_experiment
        latest_structured = latest_experiment.get("structured_metrics", {})
        latest_r2 = (latest_experiment.get("r2_test") or 
                    latest_structured.get("r2_test") or 
                    latest_experiment.get("r2_score", 0))
        
        # Préparer les données pour best_generalization
        best_gen_structured = best_generalization_exp.get("structured_metrics", {})
        best_gen_gap = (best_generalization_exp.get("r2_gap") or 
                       best_gen_structured.get("r2_gap", 0))
        best_gen_status = (best_generalization_exp.get("generalization_status") or 
                          best_gen_structured.get("generalization_status", "Unknown"))
        
        summary = {
            "total_experiments": len(experiments),
            "best_r2_score": max(r2_scores) if r2_scores else 0,
            "average_r2_score": sum(r2_scores) / len(r2_scores) if r2_scores else 0,
            "average_r2_gap": sum(r2_gaps) / len(r2_gaps) if r2_gaps else 0,
            "latest_experiment": {
                "id": latest_experiment.get("id", ""),
                "model_type": latest_experiment.get("model_type") or latest_structured.get("model_type") or latest_experiment.get("model_name", "catboost"),
                "r2_score": latest_r2,
                "timestamp": latest_experiment.get("timestamp", "")
            },
            "best_generalization": {
                "id": best_generalization_exp.get("id", ""),
                "r2_gap": best_gen_gap,
                "generalization_status": best_gen_status
            }
        }
        
        print(f"📊 Returning summary with {summary['total_experiments']} experiments")
        return summary
        
    except Exception as e:
        logger.exception("Failed to fetch experiments summary")
        raise HTTPException(status_code=500, detail=f"Failed to fetch experiments summary: {str(e)}")
    except Exception as e:
        logger.exception("Failed to fetch experiments summary")
        raise HTTPException(status_code=500, detail=f"Failed to fetch experiments summary: {str(e)}")


@app.get("/experiments/{experiment_id}")
async def get_experiment_detail(experiment_id: str):
    """Récupère les détails d'une expérience spécifique"""
    try:
        import sys
        import os
        
        # Ajouter le répertoire parent au path pour accéder à utils
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if parent_dir not in sys.path:
            sys.path.append(parent_dir)
        
        from utils.cosmosdb_logger import CosmosDbLogger
        cosmos_logger = CosmosDbLogger()
        
        # Récupérer tous les trials et chercher celui avec l'ID demandé
        experiments = cosmos_logger.get_trials_for_model("catboost", limit=100)
        experiment = next((exp for exp in experiments if exp.get("id") == experiment_id), None)
        
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found")
        
        return experiment
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to fetch experiment {experiment_id}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch experiment: {str(e)}")