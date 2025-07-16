import React, { useState } from "react";
import axios from "axios";
import ResultCard from "../ResultCard";
import EsgSummary from "../EsgSummary/EsgSummary";
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

const PropertyForm = ({ onPredictionComment, onToggleSidePanel }) => {
  const [formData, setFormData] = useState(initialFormData);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState({ all: null, top: null });
  const [error, setError] = useState(null);

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

  const handleViewDetailedESGReport = () => {
    // Ouvrir le side panel avec un message ESG détaillé
    if (onToggleSidePanel) {
      onToggleSidePanel();
    }
    
    if (onPredictionComment) {
      const timestamp = new Date().toLocaleString('fr-FR');
      const esgMessage = `📊 Detailed ESG Analysis - ${timestamp}`;
      const esgDetails = `Based on the property features in ${formData.locality}, ${formData.province}, here's a comprehensive ESG analysis:

**Environmental Impact:**
• Energy Class: ${formData.epcScore.replace('_', '')}
• Heating System: ${formData.heatingType}
• Flood Risk: ${formData.floodZoneType.replace('_', ' ')}
• Surface Efficiency: ${formData.habitableSurface} m²

**Social Benefits:**
• Location: Urban area with good accessibility
• Family Capacity: ${formData.bedroomCount} bedrooms, ${formData.roomCount} total rooms
• Quality of Life: ${formData.hasLivingRoom ? 'Living room included' : 'No living room'}, ${formData.hasTerrace ? 'Terrace available' : 'No terrace'}

**Governance Standards:**
• Construction Year: ${formData.buildingConstructionYear} (meets modern standards)
• Building Condition: ${formData.buildingCondition.replace('_', ' ')}
• Transparency: Complete property data available

This property demonstrates strong ESG credentials with particular strength in environmental efficiency and social accessibility.`;

      const comments = [esgMessage, esgDetails];
      onPredictionComment(comments);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    // Ouvrir le sidebar de gauche lors du clic sur Predict
    if (onToggleSidePanel) {
      onToggleSidePanel();
    }

    try {
      const encodedPayload = encodeInputs(formData);
      
      const response = await axios.post(`${PREDICTION_API_URL}/predict_all`, encodedPayload);

      setResults({
        all: response.data.prediction,
        top: null,
      });

      // Créer un commentaire de prédiction pour le SidePanel
      if (onPredictionComment && response.data.prediction) {
        const timestamp = new Date().toLocaleTimeString('en-US', { 
          hour12: true, 
          hour: 'numeric', 
          minute: '2-digit', 
          second: '2-digit' 
        });
        const predictionComment = `Prediction for ${formData.propertyType.toLowerCase()} in ${formData.locality}, ${formData.province} - ${timestamp}`;
        const priceComment = `Predicted price: ${Math.round(response.data.prediction).toLocaleString('fr-FR').replace(/,/g, ' ')} €`;
        
        const comments = [predictionComment, priceComment];

        // Appeler l'API des commentaires LLM pour obtenir des commentaires automatiques
        try {
          const commentPayload = {
            formData: {
              ...formData,
              region: formData.province,
              scoreMeta: {
                epcScore: formData.epcScore,
                buildingCondition: formData.buildingCondition,
                mae: 25000,
                rmse: 35000,
                r2: 0.85
              }
            },
            predictionAll: response.data.prediction,
            predictionTop: 0,
            userProfile: {
              name: "Yves",
              profile: "Investisseur",
              type: "investor",
              objectives: ["regional_market_trends", "property_characteristics_analysis", "investment_return_strategy"],
              language: "en"
            }
          };

          const commentResponse = await axios.post(COMMENT_API_URL, commentPayload);

          // Ajouter les commentaires LLM aux commentaires existants
          if (commentResponse.data && commentResponse.data.comments) {
            comments.push(...commentResponse.data.comments);
          }
        } catch (commentError) {
          console.warn("Failed to get LLM comments:", commentError);
          // Continuer sans les commentaires LLM si l'API échoue
        }

        onPredictionComment(comments);
      }

    } catch (err) {
      console.error("Error:", err);
      setError("Prediction failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="property-form" style={{ position: 'relative' }}>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
        <div style={{ display: 'flex', gap: '10px' }}>
        <button
          type="button"
          className="reset-button"
          style={{ 
            height: '35px', 
            padding: '8px 16px', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            fontSize: '14px'
          }}
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
          type="submit" 
          className="submit-button" 
          style={{ 
            height: '35px', 
            padding: '8px 16px', 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            fontSize: '14px'
          }}
          disabled={loading}
          onClick={handleSubmit}
        >
          Predict
        </button>

        {loading && (
          <span className="loading-text" style={{ marginLeft: '15px' }}>
            <span 
              style={{
                display: 'inline-block',
                width: '18px',
                height: '18px',
                border: '1px solid rgba(59, 130, 246, 0.3)',
                borderTop: '1px solid #3b82f6',
                borderRadius: '50%',
                animation: 'spin 0.8s linear infinite',
                marginRight: '3px'
              }}
            />
            Calling API...
          </span>
        )}
        </div>
        
        {/* Affichage du prix prédit aligné avec les boutons */}
        {results.all && (
          <div style={{ marginTop: '0px' }}>
            <ResultCard title="" value={results.all} />
          </div>
        )}
      </div>

      <hr style={{ border: 'none', borderTop: '1px solid #e5e7eb', margin: '20px 0' }} />

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

        {/* Living Surface */}
        <div className="form-field">
          <label>Living Surface (M²)</label>
          <input
            type="number"
            name="habitableSurface"
            value={formData.habitableSurface}
            onChange={handleChange}
            min="0"
          />
        </div>

        {/* Construction Year */}
        <div className="form-field">
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

        {/* Bedrooms */}
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

        {/* Rooms Total */}
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

        {/* Bathrooms */}
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

        {/* Toilets */}
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

        {/* Facade Count */}
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

        {/* Building Condition */}
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

        {/* Kitchen Type */}
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

        {/* Heating Type */}
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

        {/* Flood Zone */}
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

        {/* EPC Score */}
        <div className="form-field">
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

        {/* Has Living Room */}
        <div className="form-field">
          <label>Additional Features</label>
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
          </div>
        </div>

        {/* Has Terrace */}
        <div className="form-field">
          <label>&nbsp;</label>
          <div className="checkbox-inline">
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

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {/* ESG Summary - affichage tout en bas après le formulaire complet */}
      <EsgSummary 
        formData={formData} 
        onViewDetailedReport={handleViewDetailedESGReport} 
      />
    </div>
  );
};

export default PropertyForm;