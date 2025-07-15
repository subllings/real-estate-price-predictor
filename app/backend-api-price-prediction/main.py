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