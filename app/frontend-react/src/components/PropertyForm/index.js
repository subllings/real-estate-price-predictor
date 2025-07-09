import React, { useState } from "react";
import axios from "axios";
import ResultCard from "../ResultCard";
import encodeInputs from "../../helpers/encodeInputs";
import "./PropertyForm.css";

const API_URL = "http://localhost:8000";
// const API_URL = "https://realestate-api.azurewebsites.net";

const PropertyForm = () => {
  const [formData, setFormData] = useState({
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
  });

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

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState({ all: null, top: null });
  const [error, setError] = useState(null);

  const availableLocalities = Object.keys(localityData[formData.province] || []);
  const currentPostCode = localityData[formData.province]?.[formData.locality] || "";

  const handleChange = (e) => {
    const { name, type, checked, value } = e.target;
    let updatedForm = {
      ...formData,
      [name]: type === "checkbox" ? checked : value,
    };

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
    try {
      const encodedPayload = encodeInputs(formData);
      const [resAll, resTop] = await Promise.all([
        axios.post(`${API_URL}/predict_all`, encodedPayload),
        axios.post(`${API_URL}/predict_top30`, encodedPayload),
      ]);
      setResults({
        all: resAll.data.prediction,
        top: resTop.data.prediction,
      });
    } catch (err) {
      setError("Prediction failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <form className="property-form" onSubmit={handleSubmit}>
      <h2 className="form-title">Property Information</h2>

      <div className="form-grid">
        {/* Property type */}
        <div className="form-field">
          <label>Property Type</label>
          <select name="propertyType" value={formData.propertyType} onChange={handleChange}>
            {["HOUSE", "APARTMENT"].map((opt) => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        </div>

        {/* Subtype */}
        <div className="form-field">
          <label>Subtype</label>
          <select name="subtype" value={formData.subtype} onChange={handleChange}>
            {[
              "HOUSE", "VILLA", "BUNGALOW", "MANSION", "PENTHOUSE", "STUDIO", "LOFT", "DUPLEX",
              "TRIPLEX", "FARMHOUSE", "MANOR_HOUSE", "CHALET", "TOWN_HOUSE", "SERVICE_FLAT"
            ].map((opt) => (
              <option key={opt} value={opt}>{opt.replace(/_/g, " ")}</option>
            ))}
          </select>
        </div>

        {/* Province */}
        <div className="form-field">
          <label>Province</label>
          <select name="province" value={formData.province} onChange={handleChange}>
            {Object.keys(localityData).map((province) => (
              <option key={province} value={province}>{province}</option>
            ))}
          </select>
        </div>

        {/* Locality */}
        <div className="form-field">
          <label>Locality</label>
          <select name="locality" value={formData.locality} onChange={handleChange}>
            {availableLocalities.map((locality) => (
              <option key={locality} value={locality}>{locality}</option>
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
          { label: "Construction Year", name: "buildingConstructionYear" }
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
          { name: "buildingCondition", options: ["AS_NEW", "GOOD", "RENOVATION_NEEDED", "TO_RESTORE"] },
          { name: "kitchenType", options: ["HYPER_EQUIPPED", "EQUIPPED", "SIMPLE", "NOT_INSTALLED"] },
          { name: "heatingType", options: ["ELECTRIC", "GAS", "NONE"] },
          { name: "floodZoneType", options: ["NON_FLOOD_ZONE", "FLOOD_ZONE"] }
        ].map(({ name, options }) => (
          <div className="form-field" key={name}>
            <label>{name.replace(/([A-Z])/g, " $1").replace(/^./, str => str.toUpperCase())}</label>
            <select name={name} value={formData[name]} onChange={handleChange}>
              {options.map((opt) => (
                <option key={opt} value={opt}>{opt.replace(/_/g, " ")}</option>
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

      {/* Submit */}
      <div className="form-actions">
        <button type="submit" className="submit-button" disabled={loading}>
          Predict
        </button>
        {loading && <span className="loading-text">Calling API...</span>}
      </div>

      {error && <div className="error-message">{error}</div>}

      {results.all && results.top && (
        <div className="results-container">
          <ResultCard title="Prediction using all features" value={results.all} />
          <ResultCard title="Prediction using top 30 features" value={results.top} />
        </div>
      )}
    </form>
  );
};

export default PropertyForm;
