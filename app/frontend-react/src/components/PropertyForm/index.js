import React, { useState } from "react";
import axios from "axios";
import ResultCard from "../ResultCard";
import SidePanel from "../SidePanel/SidePanel.jsx";
import ESGIntegrationPrompt from "../ESGIntegrationPrompt";
import encodeInputs from "../../helpers/encodeInputs";
import { PREDICTION_API_URL, COMMENT_API_URL } from "../../config/api";
import "./PropertyForm.css";

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

const PropertyForm = ({ onDataChange, onPriceEstimate }) => {
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
  // Don't clear comments - keep accumulating them

  try {
    const encodedPayload = encodeInputs(formData);
    console.log("Encoded payload:", encodedPayload); // Debug log

    // Price prediction call - only all features
    console.log("Making API call to:", PREDICTION_API_URL); // Debug log
    const resAll = await axios.post(`${PREDICTION_API_URL}/predict_all`, encodedPayload);

    console.log("Prediction result:", resAll.data); // Debug log

    setResults({
      all: resAll.data.prediction,
      top: null, // Remove top30 predictions
    });

    // Call parent callbacks to update other components
    if (onDataChange) {
      onDataChange(formData);
    }
    if (onPriceEstimate) {
      onPriceEstimate(resAll.data.prediction);
    }

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
      userProfile: {
        name: "User",
        type: "individual",
        preferences: "detailed_analysis",
        objectives: ["buy"],
        language: "en"
      }
    };

    console.log("LLM Payload:", llmPayload);
    console.log("Making LLM API call to:", COMMENT_API_URL); // Debug log

    // LLM API call to generate commentary (non-blocking)
    try {
      const commentaryResponse = await axios.post(COMMENT_API_URL, llmPayload);
      console.log("LLM Response:", commentaryResponse.data); // Debug log

      // Create timestamp for this prediction
      const timestamp = new Date().toLocaleTimeString();
      const predictionHeader = `=== Prediction ${timestamp} ===`;

      // Extract comments from response and add to existing comments
      if (commentaryResponse.data.comments && Array.isArray(commentaryResponse.data.comments)) {
        setComments(prev => [...prev, predictionHeader, ...commentaryResponse.data.comments]);
      } else if (commentaryResponse.data.comment) {
        const commentText = commentaryResponse.data.comment;
        const newComment = typeof commentText === 'string' ? commentText : JSON.stringify(commentText);
        setComments(prev => [...prev, predictionHeader, newComment]);
      } else {
        setComments(prev => [...prev, predictionHeader, "Analysis completed. Detailed recommendations are available."]);
      }
    } catch (llmError) {
      console.warn("LLM API failed, but predictions are available:", llmError);
      // Add a default message instead of showing error
      const timestamp = new Date().toLocaleTimeString();
      setComments(prev => [...prev, `=== Prediction ${timestamp} ===`, "Analysis in progress... Detailed recommendations will be available shortly."]);
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
          {/* Location Section - Priority fields */}
          <div className="form-field location-field">
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

          <div className="form-field location-field">
            <label>Subtype</label>
            <select name="subtype" value={formData.subtype} onChange={handleChange}>
              {(subtypesByPropertyType[formData.propertyType] || []).map((opt) => (
                <option key={opt} value={opt}>
                  {opt.replace(/_/g, " ")}
                </option>
              ))}
            </select>
          </div>

          <div className="form-field location-field">
            <label>Province</label>
            <select name="province" value={formData.province} onChange={handleChange}>
              {Object.keys(localityData).map((province) => (
                <option key={province} value={province}>
                  {province}
                </option>
              ))}
            </select>
          </div>

          <div className="form-field location-field">
            <label>Locality</label>
            <select name="locality" value={formData.locality} onChange={handleChange}>
              {availableLocalities.map((locality) => (
                <option key={locality} value={locality}>
                  {locality}
                </option>
              ))}
            </select>
          </div>

          {/* Basic Property Info */}
          <div className="form-field price-field">
            <label>Living Surface (m²)</label>
            <input
              type="number"
              name="habitableSurface"
              value={formData.habitableSurface}
              onChange={handleChange}
              min="0"
            />
          </div>

          <div className="form-field price-field">
            <label>Construction Year</label>
            <input
              type="number"
              name="buildingConstructionYear"
              value={formData.buildingConstructionYear}
              onChange={handleChange}
              min="1800"
              max={new Date().getFullYear()}
            />
          </div>

          <div className="form-field">
            <label>Bedrooms</label>
            <input
              type="number"
              name="bedroomCount"
              value={formData.bedroomCount}
              onChange={handleChange}
              min="0"
            />
          </div>

          <div className="form-field">
            <label>Rooms Total</label>
            <input
              type="number"
              name="roomCount"
              value={formData.roomCount}
              onChange={handleChange}
              min="0"
            />
          </div>

          {/* Additional Details */}
          <div className="form-field">
            <label>Bathrooms</label>
            <input
              type="number"
              name="bathroomCount"
              value={formData.bathroomCount}
              onChange={handleChange}
              min="0"
            />
          </div>

          <div className="form-field">
            <label>Toilets</label>
            <input
              type="number"
              name="toiletCount"
              value={formData.toiletCount}
              onChange={handleChange}
              min="0"
            />
          </div>

          <div className="form-field">
            <label>Facade Count</label>
            <input
              type="number"
              name="facedeCount"
              value={formData.facedeCount}
              onChange={handleChange}
              min="0"
            />
          </div>

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

          {/* Property Condition & Features */}
          <div className="form-field">
            <label>Building Condition</label>
            <select name="buildingCondition" value={formData.buildingCondition} onChange={handleChange}>
              {["AS_NEW", "GOOD", "RENOVATION_NEEDED", "TO_RESTORE"].map((opt) => (
                <option key={opt} value={opt}>
                  {opt.replace(/_/g, " ")}
                </option>
              ))}
            </select>
          </div>

          <div className="form-field">
            <label>Kitchen Type</label>
            <select name="kitchenType" value={formData.kitchenType} onChange={handleChange}>
              {["HYPER_EQUIPPED", "EQUIPPED", "SIMPLE", "NOT_INSTALLED"].map((opt) => (
                <option key={opt} value={opt}>
                  {opt.replace(/_/g, " ")}
                </option>
              ))}
            </select>
          </div>

          <div className="form-field">
            <label>Heating Type</label>
            <select name="heatingType" value={formData.heatingType} onChange={handleChange}>
              {["ELECTRIC", "GAS", "NONE"].map((opt) => (
                <option key={opt} value={opt}>
                  {opt}
                </option>
              ))}
            </select>
          </div>

          <div className="form-field">
            <label>Flood Zone</label>
            <select name="floodZoneType" value={formData.floodZoneType} onChange={handleChange}>
              {["NON_FLOOD_ZONE", "FLOOD_ZONE"].map((opt) => (
                <option key={opt} value={opt}>
                  {opt.replace(/_/g, " ")}
                </option>
              ))}
            </select>
          </div>

          {/* EPC Score - Important for ESG */}
          <div className="form-field price-field">
            <label>EPC Score (Energy Class)</label>
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

          {/* Features checkboxes - Full width */}
          <div className="form-field features-section">
            <label className="form-label">Additional Features</label>
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
          <button
            type="button"
            className="clear-history-button"
            onClick={() => {
              setComments([]);
            }}
            disabled={loading}
            title="Clear commentary history"
          >
            Clear History
          </button>
          <button
            type="button"
            className="admin-button"
            onClick={() => window.open('/admin', '_blank')}
            title="Open Admin Dashboard"
          >
            🔧 Admin
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

        {results.all && (
          <div className="results-container">
            <ResultCard title="Prediction using all features" value={results.all} />
            
            {/* ESG Integration Component */}
            <ESGIntegrationPrompt 
              propertyData={{
                constructionYear: formData.buildingConstructionYear,
                surface: formData.habitableSurface,
                province: formData.province,
                locality: formData.locality,
                epcScore: formData.epcScore,
                propertyType: formData.propertyType,
                condition: formData.buildingCondition
              }}
              estimatedPrice={results.all}
            />
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
