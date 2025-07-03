import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Real Estate Price Predictor", layout="wide")
st.title("Real Estate Price Prediction")

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

    return payload

# API call logic
if submitted:
    input_data = encode_inputs()

    with st.spinner("Sending data to prediction API..."):
        time.sleep(0.8)  # show spinner briefly
        try:
            res_all = requests.post(f"{API_URL}/predict_all", json=input_data)
            #st.write(res_all.json()) # Debugging line to see the response structure
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
                st.error("Prediction failed. Please check the API response.")
        except Exception as e:
            st.error(f"Error during API call: {e}")
