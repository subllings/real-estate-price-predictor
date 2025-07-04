
import json
import streamlit as st
import os
import sys
from streamlit.web import cli as stcli
import requests
import time

# Set up Streamlit page config
st.set_page_config(page_title="Real Estate Price Predictor", layout="wide")
st.title("Real Estate Price Prediction")

# API endpoints (ALL vs TOP 30 features)
#API_URL = os.getenv("API_URL", "http://localhost:8000")
# Force usage of Azure API endpoints — NO fallback to localhost
API_URL = os.getenv("API_URL", "http://realestate-api.azurewebsites.net")

# Detect environment port (used in Docker/Azure)
port = os.environ.get("PORT", "8501")



# Categorical options
property_types = ["HOUSE", "APARTMENT"]
subtypes = [
    "HOUSE", "VILLA", "APARTMENT", "APARTMENT_BLOCK", "DUPLEX",
    "GROUND_FLOOR", "MIXED_USE_BUILDING", "PENTHOUSE", "TOWN_HOUSE"
]
provinces = [
    "Antwerp", "Brussels", "East Flanders", "Flemish Brabant", "Hainaut",
    "Limburg", "Liège", "Luxembourg", "Namur", "Walloon Brabant", "West Flanders"
]
localities = [
    "Anderlecht", "Antwerpen", "Bruxelles", "Gent", "Ixelles",
    "Knokke-Heist", "Liège", "Uccle"
]
locality_to_postcode = {
    "Anderlecht": 1070,
    "Antwerpen": 2000,
    "Bruxelles": 1000,
    "Gent": 9000,
    "Ixelles": 1050,
    "Knokke-Heist": 8300,
    "Liège": 4000,
    "Uccle": 1180
}
building_conditions = [
    "AS_NEW", "GOOD", "JUST_RENOVATED", "TO_BE_DONE_UP", "TO_RENOVATE", "nan"
]
kitchen_types = [
    "HYPER_EQUIPPED", "INSTALLED", "NOT_INSTALLED", "SEMI_EQUIPPED",
    "USA_HYPER_EQUIPPED", "USA_INSTALLED", "nan"
]
heating_types = ["ELECTRIC", "FUELOIL", "GAS", "PELLET", "nan"]
flood_zones = ["NON_FLOOD_ZONE", "POSSIBLE_FLOOD_ZONE", "RECOGNIZED_FLOOD_ZONE", "nan"]
epc_labels = ["A+", "A", "B", "C", "D", "E", "F", "G"]

# Form layout
with st.form("property_form"):
    st.subheader("Property Information")
    col1, col2 = st.columns(2)

    with col1:
        prop_type = st.selectbox("Property Type", property_types)
        subtype = st.selectbox("Subtype", subtypes)
        province = st.selectbox("Province", provinces)
        locality = st.selectbox("Locality", localities)
        bedroomCount = st.number_input("Bedrooms", min_value=0, value=3)
        bathroomCount = st.number_input("Bathrooms", min_value=0, value=1)
        toiletCount = st.number_input("Toilets", min_value=0, value=1)
        room_count = st.number_input("Total Rooms", min_value=1, value=5)
        habitableSurface = st.number_input("Habitable Surface (m²)", min_value=10, value=110)

    with col2:
        facedeCount = st.number_input("Facades", min_value=1, value=2)
        buildingConstructionYear = st.number_input("Construction Year", min_value=1800, value=2000)
        buildingCondition = st.selectbox("Building Condition", building_conditions)
        kitchenType = st.selectbox("Kitchen Type", kitchen_types)
        heatingType = st.selectbox("Heating Type", heating_types)
        floodZoneType = st.selectbox("Flood Zone Type", flood_zones)
        epcScore = st.selectbox("EPC Label", epc_labels)
        hasLivingRoom = st.checkbox("Has Living Room", value=True)
        hasTerrace = st.checkbox("Has Terrace", value=True)
   

    submitted = st.form_submit_button("Predict")

# Feature engineering
def encode_inputs():
    year_now = 2024
    building_age = year_now - buildingConstructionYear
    surface_per_room = habitableSurface / room_count if room_count > 0 else 0
    postCode = locality_to_postcode.get(str(locality), 1000)

    payload = {
        "bedroomCount": bedroomCount,
        "bathroomCount": bathroomCount,
        "postCode": postCode,
        "habitableSurface": habitableSurface,
        "buildingConstructionYear": buildingConstructionYear,
        "facedeCount": facedeCount,
        "toiletCount": toiletCount,
        "room_count": room_count,
        "surface_per_room": surface_per_room,
        "building_age": building_age,
        "hasLivingRoom": int(hasLivingRoom),
        "hasTerrace": int(hasTerrace),
    }

    # One-hot encode categorical fields
    for t in property_types:
        payload[f"type_{t}"] = int(t == prop_type)

    for stype in subtypes:
        payload[f"subtype_{stype}"] = int(stype == subtype)

    for prov in provinces:
        payload[f"province_{prov}"] = int(prov == province)

    for loc in localities:
        payload[f"locality_{loc}"] = int(loc == locality)

    for cond in building_conditions:
        payload[f"buildingCondition_{cond}"] = int(cond == buildingCondition)

    for kt in kitchen_types:
        payload[f"kitchenType_{kt}"] = int(kt == kitchenType)

    for ht in heating_types:
        payload[f"heatingType_{ht}"] = int(ht == heatingType)

    for fz in flood_zones:
        payload[f"floodZoneType_{fz}"] = int(fz == floodZoneType)

    for label in epc_labels:
        payload[f"epcScore_{label}"] = int(label == epcScore)

    # Final validation: ensure all values are numeric and not dicts
    for key, value in payload.items():
        if isinstance(value, dict):
            raise ValueError(f"Invalid value for '{key}': dict found instead of number")
        if isinstance(value, bool):
            payload[key] = int(value)
        if not isinstance(payload[key], (int, float)):
            raise ValueError(f"Invalid type for '{key}': {type(value)}")

    return payload


# API call logic
if submitted:
    try:
        input_data = encode_inputs()

        # DEBUG – détecter les erreurs de type dans l'input
        for k, v in input_data.items():
            if isinstance(v, dict):
                st.error(f"Key '{k}' has a dict instead of a number → {v}")
            elif not isinstance(v, (int, float)):
                st.warning(f"Key '{k}' has type {type(v)} with value: {v}")

        #st.subheader("Input JSON sent to API")
        #st.json(input_data) # Display the input data as JSON for debugging

        with st.spinner("Sending data to prediction API..."):
            time.sleep(0.8)
            res_all = requests.post(f"{API_URL}/predict_all", json=input_data)
            res_top = requests.post(f"{API_URL}/predict_top30", json=input_data)

            if res_all.ok and res_top.ok:
                price_all = res_all.json()["prediction"]
                price_top = res_top.json()["prediction"]

                col1, col2 = st.columns(2)
                with col1:
                    st.success("Prediction using all features")
                    st.metric("Estimated Price (€)", f"{int(price_all):,}".replace(",", " "))
                with col2:
                    st.success("Prediction using top 30 features")
                    st.metric("Estimated Price (€)", f"{int(price_top):,}".replace(",", " "))
            else:
                st.error(f"Prediction failed. API responses:\n/predict_all: {res_all.text}\n/predict_top30: {res_top.text}")
    except Exception as e:
        st.error(f"Error during API call: {e}")
