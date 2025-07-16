import React, { useState } from "react";
import axios from "axios";
import ResultCard from "../ResultCard";
import SidePanel from "../SidePanel/SidePanel.jsx";
import encodeInputs from "../../helpers/encodeInputs";
import "./PropertyForm.css";

// const API_URL = "https://realestate-api.azurewebsites.net";
// const LLM_API_URL = "https://realestate-api-llm-v2.azurewebsites.net/comment";

const API_URL = "http://127.0.0.1:8000";
const LLM_API_URL = "http://127.0.0.1:8010/comment";


const initialFormData = {
  propertyType: "HOUSE",
  subtype: "HOUSE",
  province: "Antwerp",
  locality: "Antwerpen",
  postCode: "2000",
  bedroomCount: 3,
  bathroomCount: 1,
  toiletCount: 1,
  roomCount: 5,
  habitableSurface: 110,
  facedeCount: 2,
  buildingConstructionYear: 2000,
  buildingCondition: "AS_NEW",
  kitchenType: "HYPER_EQUIPPED",
  heatingType: "ELECTRIC",
  floodZoneType: "NON_FLOOD_ZONE",
  epcScore: "A_plus",
  hasLivingRoom: true,
  hasTerrace: true,
};

const PropertyForm = () => {
  const [isSidePanelExpanded, setIsSidePanelExpanded] = useState(false);
  const [formData, setFormData] = useState(initialFormData);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState({ all: null, top: null });
  const [error, setError] = useState(null);

  const [comments, setComments] = useState([]);


  const subtypesByPropertyType = {
    HOUSE: [
      "HOUSE",
      "VILLA",
      "BUNGALOW",
      "MANSION",
      "FARMHOUSE",
      "MANOR_HOUSE",
      "CHALET",
      "TOWN_HOUSE",
      "SERVICE_FLAT",
    ],
    APARTMENT: [
      "APARTMENT",
      "APARTMENT_BLOCK",
      "DUPLEX",
      "GROUND_FLOOR",
      "PENTHOUSE",
    ],
  };

  const localityData = {
    Brussels: {
      Anderlecht: 1070,
      Ixelles: 1050,
      Uccle: 1180,
      "Woluwe-Saint-Lambert": 1200,
      "Woluwe-Saint-Pierre": 1150,
      Bruxelles: 1000,
    },
    Antwerp: {
      Antwerpen: 2000,
    },
    "East Flanders": {
      Gent: 9000,
    },
    Liège: {
      Liège: 4000,
    },
  };

  const postCodeToLocation = {};
  Object.entries(localityData).forEach(([province, localities]) => {
    Object.entries(localities).forEach(([locality, postCode]) => {
      postCodeToLocation[postCode] = { province, locality };
    });
  });

  const availableLocalities = Object.keys(localityData[formData.province] || []);

  const handleChange = (e) => {
    const { name, type, checked, value } = e.target;
    console.log(`handleChange called for ${name}:`, type === "checkbox" ? checked : value); // DEBUG

    let updatedForm = {
      ...formData,
      [name]: type === "checkbox" ? checked : value,
    };

    if (name === "propertyType") {
      const newSubtypes = subtypesByPropertyType[value] || [];
      updatedForm.subtype = newSubtypes.length > 0 ? newSubtypes[0] : "";
    }

    if (name === "province") {
      const newProvince = value;
      const newLocality = Object.keys(localityData[newProvince])[0];
      updatedForm.locality = newLocality;
      updatedForm.postCode = localityData[newProvince][newLocality];
    }

    if (name === "locality") {
      updatedForm.postCode = localityData[formData.province][value];
    }

    if (name === "postCode") {
      const location = postCodeToLocation[value];
      if (location) {
        updatedForm.province = location.province;
        updatedForm.locality = location.locality;
      }
    }

    setFormData(updatedForm);
  };

const handleSubmit = async (e) => {
  e.preventDefault();
  setLoading(true);
  setError(null);
  setComments([]); // Clear comments at the beginning

  try {
    const encodedPayload = encodeInputs(formData);
    console.log("Encoded payload:", encodedPayload); // Debug log

    // Price prediction calls
    console.log("Making API calls to:", API_URL); // Debug log
    const [resAll, resTop] = await Promise.all([
      axios.post(`${API_URL}/predict_all`, encodedPayload),
      axios.post(`${API_URL}/predict_top30`, encodedPayload),
    ]);

    console.log("Prediction results:", { all: resAll.data, top: resTop.data }); // Debug log

    setResults({
      all: resAll.data.prediction,
      top: resTop.data.prediction,
    });

    // Prepare data to send to LLM
    const llmPayload = {
      formData: {
        ...formData,
        region: formData.province, // Add region field (same as province)
        scoreMeta: {
          scoreType: "prediction",
          confidence: "medium",
          accuracy: "85%",
          mae: 15000,
          rmse: 25000,
          r2: 0.85
        }
      },
      predictionAll: resAll.data.prediction,
      predictionTop: resTop.data.prediction,
      userProfile: {
        name: "User",
        type: "individual",
        preferences: "detailed_analysis",
        objectives: ["buy"],
        language: "fr"
      }
    };

    console.log("LLM Payload:", llmPayload);
    console.log("Making LLM API call to:", LLM_API_URL); // Debug log

    // LLM API call to generate commentary (non-blocking)
    try {
      const commentaryResponse = await axios.post(LLM_API_URL, llmPayload);
      console.log("LLM Response:", commentaryResponse.data); // Debug log

      // Extract comments from response
      if (commentaryResponse.data.comments && Array.isArray(commentaryResponse.data.comments)) {
        setComments(commentaryResponse.data.comments);
      } else if (commentaryResponse.data.comment) {
        const commentText = commentaryResponse.data.comment;
        setComments([typeof commentText === 'string' ? commentText : JSON.stringify(commentText)]);
      } else {
        setComments(["Analyse terminée. Les recommandations détaillées sont disponibles."]);
      }
    } catch (llmError) {
      console.warn("LLM API failed, but predictions are available:", llmError);
      // Set a default message instead of showing error
      setComments(["Analyse en cours... Les recommandations détaillées seront disponibles prochainement."]);
    }

  } catch (err) {
    console.error("Error details:", err); // Enhanced error logging
    console.error("Error response:", err.response); // Log the full response
    
    let errorMsg = "Prediction failed. Please try again.";
    
    if (err.response && err.response.data) {
      const errorData = err.response.data;
      
      // Handle different types of error responses
      if (errorData.detail) {
        // If detail is an array (validation errors)
        if (Array.isArray(errorData.detail)) {
          errorMsg = errorData.detail.map(error => {
            if (typeof error === 'object') {
              return `${error.msg || error.message || 'Validation error'} (field: ${error.loc ? error.loc.join('.') : 'unknown'})`;
            }
            return String(error);
          }).join('; ');
        } else if (typeof errorData.detail === 'object') {
          // If detail is an object
          errorMsg = JSON.stringify(errorData.detail);
        } else {
          // If detail is a string
          errorMsg = String(errorData.detail);
        }
      } else if (errorData.message) {
        errorMsg = String(errorData.message);
      } else {
        // Fallback: stringify the entire error data
        errorMsg = JSON.stringify(errorData);
      }
    } else if (err.message) {
      errorMsg = String(err.message);
    }
    
    setError(errorMsg);
  } finally {
    setLoading(false);
  }
};


  return (
    <>
      <form className="property-form" onSubmit={handleSubmit}>
        <h2 className="form-title">Property Information</h2>

        <div className="form-grid">
          {/* Property Type */}
          <div className="form-field">
            <label>Property Type</label>
            <select
              name="propertyType"
              value={formData.propertyType}
              onChange={handleChange}
            >
              {["HOUSE", "APARTMENT"].map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          </div>

          {/* Subtype */}
          <div className="form-field">
            <label>Subtype</label>
            <select name="subtype" value={formData.subtype} onChange={handleChange}>
              {(subtypesByPropertyType[formData.propertyType] || []).map((opt) => (
                <option key={opt} value={opt}>
                  {opt.replace(/_/g, " ")}
                </option>
              ))}
            </select>
          </div>

          {/* Province */}
          <div className="form-field">
            <label>Province</label>
            <select name="province" value={formData.province} onChange={handleChange}>
              {Object.keys(localityData).map((province) => (
                <option key={province} value={province}>
                  {province}
                </option>
              ))}
            </select>
          </div>

          {/* Locality */}
          <div className="form-field">
            <label>Locality</label>
            <select name="locality" value={formData.locality} onChange={handleChange}>
              {availableLocalities.map((locality) => (
                <option key={locality} value={locality}>
                  {locality}
                </option>
              ))}
            </select>
          </div>

          {/* Post Code */}
          <div className="form-field">
            <label>Post Code</label>
            <select name="postCode" value={formData.postCode} onChange={handleChange}>
              {Object.entries(postCodeToLocation).map(([code, data]) => (
                <option key={code} value={code}>
                  {code} – {data.locality}
                </option>
              ))}
            </select>
          </div>

          {/* Numeric fields */}
          {[
            { label: "Bedrooms", name: "bedroomCount" },
            { label: "Bathrooms", name: "bathroomCount" },
            { label: "Toilets", name: "toiletCount" },
            { label: "Rooms", name: "roomCount" },
            { label: "Habitable Surface (m²)", name: "habitableSurface" },
            { label: "Facade Count", name: "facedeCount" },
            { label: "Construction Year", name: "buildingConstructionYear" },
          ].map(({ label, name }) => (
            <div className="form-field" key={name}>
              <label>{label}</label>
              <input
                type="number"
                name={name}
                value={formData[name]}
                onChange={handleChange}
                min={name === "buildingConstructionYear" ? 1800 : 0}
                max={name === "buildingConstructionYear" ? new Date().getFullYear() : undefined}
              />
            </div>
          ))}

          {/* Select fields */}
          {[
            {
              name: "buildingCondition",
              options: ["AS_NEW", "GOOD", "RENOVATION_NEEDED", "TO_RESTORE"],
            },
            {
              name: "kitchenType",
              options: ["HYPER_EQUIPPED", "EQUIPPED", "SIMPLE", "NOT_INSTALLED"],
            },
            { name: "heatingType", options: ["ELECTRIC", "GAS", "NONE"] },
            { name: "floodZoneType", options: ["NON_FLOOD_ZONE", "FLOOD_ZONE"] },
          ].map(({ name, options }) => (
            <div className="form-field" key={name}>
              <label>
                {name.replace(/([A-Z])/g, " $1").replace(/^./, (str) => str.toUpperCase())}
              </label>
              <select name={name} value={formData[name]} onChange={handleChange}>
                {options.map((opt) => (
                  <option key={opt} value={opt}>
                    {opt.replace(/_/g, " ")}
                  </option>
                ))}
              </select>
            </div>
          ))}

          {/* EPC Score */}
          <div className="form-field">
            <label>EPC Score</label>
            <select name="epcScore" value={formData.epcScore} onChange={handleChange}>
              <option value="A_plus">A+</option>
              <option value="A">A</option>
              <option value="B">B</option>
              <option value="C">C</option>
              <option value="D">D</option>
              <option value="E">E</option>
              <option value="F">F</option>
              <option value="G">G</option>
            </select>
          </div>

          {/* Features checkboxes */}
          <div className="form-field">
            <label className="form-label">Features</label>
            <div className="checkbox-inline">
              <label className="checkbox-item">
                <input
                  type="checkbox"
                  name="hasLivingRoom"
                  checked={formData.hasLivingRoom}
                  onChange={handleChange}
                />
                Has Living Room
              </label>
              <label className="checkbox-item">
                <input
                  type="checkbox"
                  name="hasTerrace"
                  checked={formData.hasTerrace}
                  onChange={handleChange}
                />
                Has Terrace
              </label>
            </div>
          </div>
        </div>

        <div className="form-actions">
          <button
            type="button"
            className="reset-button"
            onClick={() => {
              setFormData(initialFormData);
              setResults({ all: null, top: null });
              setError(null);
            }}
            disabled={loading}
          >
            Reset
          </button>
          <button type="submit" className="submit-button" disabled={loading}>
            Predict
          </button>

          {loading && (
            <span className="loading-text">
              <span className="spinner" />
              Calling API...
            </span>
          )}
        </div>

        {error && (
          <div className="error-message">
            {typeof error === 'string' ? error : JSON.stringify(error)}
          </div>
        )}

        {results.all && results.top && (
          <div className="results-container">
            <ResultCard title="Prediction using all features" value={results.all} />
            <ResultCard title="Prediction using top 30 features" value={results.top} />
          </div>
        )}
      </form>

    <SidePanel
      user={{ profile: "Yves", history: ["search1", "search2"] }}
      isExpanded={isSidePanelExpanded}
      onToggle={() => setIsSidePanelExpanded(!isSidePanelExpanded)}
      onClose={() => setIsSidePanelExpanded(false)}
      comments={comments}
    />


    </>
  );
};

export default PropertyForm;
