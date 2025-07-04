from typing import Union
import pandas as pd

class InputFeatureConverter:
    """
    Converts raw user input into a DataFrame with all expected features (one-hot + numeric)
    required by the model.
    """

    def __init__(self):
        self.expected_features = [
            "bedroomCount", "bathroomCount", "postCode", "habitableSurface", "buildingConstructionYear",
            "facedeCount", "toiletCount", "room_count", "surface_per_room", "building_age",
            "type_APARTMENT", "type_HOUSE",
            "subtype_APARTMENT", "subtype_APARTMENT_BLOCK", "subtype_DUPLEX", "subtype_GROUND_FLOOR",
            "subtype_HOUSE", "subtype_MIXED_USE_BUILDING", "subtype_PENTHOUSE", "subtype_TOWN_HOUSE", "subtype_VILLA",
            "province_Antwerp", "province_Brussels", "province_East Flanders", "province_Flemish Brabant",
            "province_Hainaut", "province_Limburg", "province_Liège", "province_Luxembourg",
            "province_Namur", "province_Walloon Brabant", "province_West Flanders",
            "locality_Anderlecht", "locality_Antwerpen", "locality_Bruxelles", "locality_Gent", "locality_Ixelles",
            "locality_Knokke-Heist", "locality_Liège", "locality_Uccle",
            "buildingCondition_AS_NEW", "buildingCondition_GOOD", "buildingCondition_JUST_RENOVATED",
            "buildingCondition_TO_BE_DONE_UP", "buildingCondition_TO_RENOVATE", "buildingCondition_nan",
            "floodZoneType_NON_FLOOD_ZONE", "floodZoneType_POSSIBLE_FLOOD_ZONE",
            "floodZoneType_RECOGNIZED_FLOOD_ZONE", "floodZoneType_nan",
            "heatingType_ELECTRIC", "heatingType_FUELOIL", "heatingType_GAS", "heatingType_PELLET", "heatingType_nan",
            "kitchenType_HYPER_EQUIPPED", "kitchenType_INSTALLED", "kitchenType_NOT_INSTALLED",
            "kitchenType_SEMI_EQUIPPED", "kitchenType_USA_HYPER_EQUIPPED", "kitchenType_USA_INSTALLED", "kitchenType_nan",
            "epcScore_A", "epcScore_A+", "epcScore_B", "epcScore_C", "epcScore_D", "epcScore_E", "epcScore_F", "epcScore_G",
            "hasLivingRoom", "hasTerrace"
        ]

    def convert(self, raw_input: dict) -> pd.DataFrame:
        features = {feat: 0 for feat in self.expected_features}

        # Numeric and derived features
        try:
            features["bedroomCount"] = raw_input["bedroomCount"]
            features["bathroomCount"] = raw_input["bathroomCount"]
            features["postCode"] = raw_input["postCode"]
            features["habitableSurface"] = raw_input["habitableSurface"]
            features["buildingConstructionYear"] = raw_input["buildingConstructionYear"]
            features["facedeCount"] = raw_input["facedeCount"]
            features["toiletCount"] = raw_input["toiletCount"]

            features["room_count"] = raw_input["room_count"]
            features["surface_per_room"] = (
                raw_input["habitableSurface"] / raw_input["room_count"]
                if raw_input["room_count"] > 0 else 0
            )
            features["building_age"] = 2025 - raw_input["buildingConstructionYear"]
        except Exception as e:
            print("Error computing numeric fields:", e)

        # One-hot encoding for categorical fields
        for prefix in [
            "type", "subtype", "province", "locality",
            "buildingCondition", "floodZoneType",
            "heatingType", "kitchenType", "epcScore"
        ]:
            raw_value = raw_input.get(prefix)

            # Skip dicts or invalid types (this is your bug)
            if isinstance(raw_value, dict):
                print(f"Skipped invalid dict for {prefix}: {raw_value}")
                continue

            # Proceed only if value is a valid string
            if isinstance(raw_value, str):
                key = f"{prefix}_{raw_value}"
                if key in features:
                    features[key] = 1
            else:
                print(f"Skipping invalid value for {prefix}: {raw_value} (type={type(raw_value)})")

