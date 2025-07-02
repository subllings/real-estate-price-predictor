from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Enable CORS for frontend on localhost
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load both models at startup
MODEL_ALL_PATH = "models/pkl/catboost_optuna_all_20250702_1417_TEST.pkl"
MODEL_TOP30_PATH = "models/pkl/catboost_optuna_top30_20250702_1417_TEST.pkl"

with open(MODEL_ALL_PATH, "rb") as f:
    model_all = joblib.load(f)

with open(MODEL_TOP30_PATH, "rb") as f:
    model_top30 = joblib.load(f)

print("Both models loaded.")

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
    province_East_Flanders: float
    province_Flemish_Brabant: float
    province_Hainaut: float
    province_Limburg: float
    province_Liège: float
    province_Luxembourg: float
    province_Namur: float
    province_Walloon_Brabant: float
    province_West_Flanders: float
    locality_Anderlecht: float
    locality_Antwerpen: float
    locality_Bruxelles: float
    locality_Gent: float
    locality_Ixelles: float
    locality_Knokke_Heist: float
    locality_Liège: float
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
    epcScore_A_: float
    epcScore_B: float
    epcScore_C: float
    epcScore_D: float
    epcScore_E: float
    epcScore_F: float
    epcScore_G: float
    hasLivingRoom: float
    hasTerrace: float

# Input schema for top 30 feature model
class InputDataTop30(BaseModel):
    habitableSurface: float
    bathroomCount: float
    postCode: float
    toiletCount: float
    buildingConstructionYear: float
    locality_Knokke_Heist: float
    building_age: float
    surface_per_room: float
    facedeCount: float
    kitchenType_HYPER_EQUIPPED: float
    buildingCondition_AS_NEW: float
    province_West_Flanders: float
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

# Endpoint for full feature model
@app.post("/predict_all")
def predict_all(data: InputDataAll):
    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model_all.predict(input_df)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Endpoint for top 30 features model
@app.post("/predict_top30")
def predict_top30(data: InputDataTop30):
    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model_top30.predict(input_df)
        return {"prediction": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
