export default function encodeInputs(formData) {
  // Étape 1 : mapping entre noms encodés (React) et vrais noms utilisés par le modèle FastAPI
  const ALIAS_MAPPING = {
    province_West_Flanders: "province_West Flanders",
    province_East_Flanders: "province_East Flanders",
    province_Flemish_Brabant: "province_Flemish Brabant",
    province_Walloon_Brabant: "province_Walloon Brabant",
    locality_Knokke_Heist: "locality_Knokke-Heist",
    locality_Liège: "locality_Liège",
    epcScore_A_plus: "epcScore_A+"
  };

  const allFeatures = {
      // Property type
    "type_APARTMENT": 0,
    "type_HOUSE": 0,

    // Subtypes (ajuste selon ton modèle exact)
    "subtype_APARTMENT": 0,
    "subtype_HOUSE": 0,
    "subtype_VILLA": 0,
    "subtype_PENTHOUSE": 0,

    // Provinces (liste complète selon ton modèle)
    "province_Antwerp": 0,
    "province_East Flanders": 0,
    "province_Flemish Brabant": 0,
    "province_Liège": 0,
    "province_Walloon Brabant": 0,
    "province_West Flanders": 0,

    // Localities (exemples — complète selon tes one-hot encodings)
    "locality_Gent": 0,
    "locality_Ixelles": 0,
    "locality_Knokke-Heist": 0,
    "locality_Liège": 0,
    "locality_Uccle": 0,
    "locality_Anderlecht": 0,

    // Building condition
    "buildingCondition_AS_NEW": 0,
    "buildingCondition_GOOD": 0,
    "buildingCondition_JUST_RENOVATED": 0,
    "buildingCondition_TO_BE_DONE_UP": 0,
    "buildingCondition_TO_RENOVATE": 0,
    "buildingCondition_TO_RESTORE": 0,

    // Flood zones
    "floodZoneType_NON_FLOOD_ZONE": 0,
    "floodZoneType_POSSIBLE_FLOOD_ZONE": 0,
    "floodZoneType_RECOGNIZED_FLOOD_ZONE": 0,

    // Heating
    "heatingType_ELECTRIC": 0,
    "heatingType_GAS": 0,
    "heatingType_OIL": 0,
    "heatingType_PELLET": 0,
    "heatingType_SOLAR": 0,
    "heatingType_NONE": 0,

    // Kitchen types
    "kitchenType_HYPER_EQUIPPED": 0,
    "kitchenType_INSTALLED": 0,
    "kitchenType_SEMI_EQUIPPED": 0,
    "kitchenType_USA_HYPER_EQUIPPED": 0,
    "kitchenType_USA_INSTALLED": 0,
    "kitchenType_nan": 0,

    // EPC
    "epcScore_A+": 0,
    "epcScore_A": 0,
    "epcScore_B": 0,
    "epcScore_C": 0,
    "epcScore_D": 0,
    "epcScore_E": 0,
    "epcScore_F": 0,
    "epcScore_G": 0,

    // Booléens
    "hasLivingRoom": 0,
    "hasTerrace": 0,

    // Numériques
    "bedroomCount": 0,
    "bathroomCount": 0,
    "toiletCount": 0,
    "facedeCount": 0,
    "habitableSurface": 0,
    "buildingConstructionYear": 0,
    "building_age": 0,
    "postCode": 0,
    "room_count": 0,
    "surface_per_room": 0
  };
  
  allFeatures.bedroomCount = Number(formData.bedroomCount) || 0;
  allFeatures.bathroomCount = Number(formData.bathroomCount) || 0;
  allFeatures.postCode = Number(formData.postCode) || 0;
  allFeatures.habitableSurface = Number(formData.habitableSurface) || 0;
  allFeatures.buildingConstructionYear = Number(formData.buildingConstructionYear) || 0;
  allFeatures.facedeCount = Number(formData.facedeCount) || 0;
  allFeatures.toiletCount = Number(formData.toiletCount) || 0;
  allFeatures.room_count = Number(formData.roomCount) || 0;

  allFeatures.surface_per_room = allFeatures.room_count > 0
    ? allFeatures.habitableSurface / allFeatures.room_count
    : 0;

  allFeatures.building_age = allFeatures.buildingConstructionYear > 0
    ? 2025 - allFeatures.buildingConstructionYear
    : 0;

  const normalize = (text) => {
    if (!text) return "";
    return text.trim().replace(/\s+/g, "_");
  };

  const setOneHot = (prefix, value) => {
    const key = `${prefix}_${normalize(value)}`;
    if (key in allFeatures) {
      allFeatures[key] = 1;
    } else {
      console.warn(`⚠️ Warning: ${key} not found in allFeatures`);
    }
  };

  setOneHot("type", formData.propertyType);
  setOneHot("subtype", formData.subtype);
  setOneHot("province", formData.province);
  setOneHot("locality", formData.locality);
  setOneHot("buildingCondition", formData.buildingCondition);
  setOneHot("floodZoneType", formData.floodZoneType);
  setOneHot("heatingType", formData.heatingType);
  setOneHot("kitchenType", formData.kitchenType);
  setOneHot("epcScore", formData.epcScore);

  allFeatures.hasLivingRoom = formData.hasLivingRoom ? 1 : 0;
  allFeatures.hasTerrace = formData.hasTerrace ? 1 : 0;

  // Étape 2 : appliquer les alias pour correspondre aux noms attendus par FastAPI
  for (const [alias, realKey] of Object.entries(ALIAS_MAPPING)) {
    if (allFeatures[alias] !== undefined) {
      allFeatures[realKey] = allFeatures[alias];
      delete allFeatures[alias];
    }
  }

  console.log("✅ Features sent to API:", allFeatures);
  return allFeatures;
}
