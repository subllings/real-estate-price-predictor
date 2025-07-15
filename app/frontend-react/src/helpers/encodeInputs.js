export default function encodeInputs(formData) {
  

const allFeatures = {
  // Property type
  "type_APARTMENT": 0,
  "type_HOUSE": 0,

  // Subtypes
  "subtype_APARTMENT_BLOCK": 0,
  "subtype_DUPLEX": 0,
  "subtype_GROUND_FLOOR": 0,
  "subtype_HOUSE": 0,
  "subtype_PENTHOUSE": 0,
  "subtype_TOWN_HOUSE": 0,
  "subtype_VILLA": 0,
  "subtype_MIXED_USE_BUILDING": 0,
  "subtype_APARTMENT": 0,

  // Provinces
  "province_Antwerp": 0,
  "province_East Flanders": 0,
  "province_Flemish Brabant": 0,
  "province_Liège": 0,
  "province_Walloon Brabant": 0,
  "province_West Flanders": 0,
  "province_Brussels": 0,
  "province_Hainaut": 0,
  "province_Limburg": 0,
  "province_Luxembourg": 0,
  "province_Namur": 0,

  // Localities
  "locality_Gent": 0,
  "locality_Ixelles": 0,
  "locality_Knokke-Heist": 0,
  "locality_Liège": 0,
  "locality_Uccle": 0,
  "locality_Anderlecht": 0,
  "locality_Antwerpen": 0,
  "locality_Bruxelles": 0,

  // Building condition
  "buildingCondition_AS_NEW": 0,
  "buildingCondition_GOOD": 0,
  "buildingCondition_JUST_RENOVATED": 0,
  "buildingCondition_TO_BE_DONE_UP": 0,
  "buildingCondition_TO_RENOVATE": 0,
  "buildingCondition_RENOVATION_NEEDED": 0,
  "buildingCondition_TO_RESTORE": 0,
  "buildingCondition_nan": 0,

  // Flood zones
  "floodZoneType_NON_FLOOD_ZONE": 0,
  "floodZoneType_FLOOD_ZONE": 0,
  "floodZoneType_POSSIBLE_FLOOD_ZONE": 0,
  "floodZoneType_RECOGNIZED_FLOOD_ZONE": 0,
  "floodZoneType_nan": 0,

  // Heating
  "heatingType_ELECTRIC": 0,
  "heatingType_GAS": 0,
  "heatingType_OIL": 0,
  "heatingType_PELLET": 0,
  "heatingType_SOLAR": 0,
  "heatingType_NONE": 0,
  "heatingType_FUELOIL": 0,
  "heatingType_nan": 0,

  // Kitchen types
  "kitchenType_HYPER_EQUIPPED": 0,
  "kitchenType_EQUIPPED": 0,
  "kitchenType_SIMPLE": 0,
  "kitchenType_INSTALLED": 0,
  "kitchenType_SEMI_EQUIPPED": 0,
  "kitchenType_USA_HYPER_EQUIPPED": 0,
  "kitchenType_USA_INSTALLED": 0,
  "kitchenType_NOT_INSTALLED": 0,
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

  // Booleans
  "hasLivingRoom": 0,
  "hasTerrace": 0,

  // Numeric features
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
  return text.trim()
    .replace(/\s+/g, "_")
    .replace("+", "_plus");  
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

  

  console.log("Features sent to API:", allFeatures);
  return allFeatures;
}
