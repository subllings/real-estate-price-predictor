import React, { useState } from "react";
import axios from "axios";
import ResultCard from "../ResultCard";
import encodeInputs from "../../helpers/encodeInputs";

const API_URL = "http://localhost:8000";

const PropertyForm = () => {
  const [formData, setFormData] = useState({
    propertyType: "HOUSE",
    subtype: "HOUSE",
    province: "Antwerp",
    locality: "Anderlecht",
    postCode: "1050",
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

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState({ all: null, top: null });
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, type, checked, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === "checkbox" ? checked : value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      
      const encodedPayload = encodeInputs(formData);
      console.log("encoded inputs sent to backend:", encodedPayload);


      const [resAll, resTop] = await Promise.all([
        axios.post(`${API_URL}/predict_all`, encodedPayload )
        //axios.post(`${API_URL}/predict_top30`, encodedPayload ),
      ]);
      console.log("Encoded payload to send:", encodedPayload);

      setResults({
        all: resAll.data.prediction,
        top: resTop.data.prediction,
      });
    } catch (err) {
      setError("Prediction failed. Please try again.");
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      style={{ maxWidth: "700px", margin: "auto", padding: "20px", fontFamily: "Arial, sans-serif" }}
    >
      <h2>Property Information</h2>

      <label>
        Property Type:
        <select name="propertyType" value={formData.propertyType} onChange={handleChange} style={{ marginLeft: "10px" }}>
          <option value="HOUSE">House</option>
          <option value="APARTMENT">Apartment</option>
        </select>
      </label>
      <br /><br />

      <label>
        Subtype:
        <select name="subtype" value={formData.subtype} onChange={handleChange} style={{ marginLeft: "10px" }}>
          <option value="HOUSE">House</option>
          <option value="APARTMENT">Apartment</option>
        </select>
      </label>
      <br /><br />

      <label>
        Province:
        <input type="text" name="province" value={formData.province} onChange={handleChange} style={{ marginLeft: "10px" }} />
      </label>
      <br /><br />

      <label>
        Locality:
        <input type="text" name="locality" value={formData.locality} onChange={handleChange} style={{ marginLeft: "10px" }} />
      </label>
      <br /><br />

      <label>
        Post Code:
        <input type="text" name="postCode" value={formData.postCode} onChange={handleChange} style={{ marginLeft: "10px" }} />
      </label>
      <br /><br />

      <label>
        Bedrooms:
        <input type="number" name="bedroomCount" value={formData.bedroomCount} onChange={handleChange} min={0} style={{ marginLeft: "10px", width: "50px" }} />
      </label>
      <br /><br />

      <label>
        Bathrooms:
        <input type="number" name="bathroomCount" value={formData.bathroomCount} onChange={handleChange} min={0} style={{ marginLeft: "10px", width: "50px" }} />
      </label>
      <br /><br />

      <label>
        Toilets:
        <input type="number" name="toiletCount" value={formData.toiletCount} onChange={handleChange} min={0} style={{ marginLeft: "10px", width: "50px" }} />
      </label>
      <br /><br />

      <label>
        Rooms:
        <input type="number" name="roomCount" value={formData.roomCount} onChange={handleChange} min={0} style={{ marginLeft: "10px", width: "50px" }} />
      </label>
      <br /><br />

      <label>
        Habitable Surface (m²):
        <input type="number" name="habitableSurface" value={formData.habitableSurface} onChange={handleChange} min={0} style={{ marginLeft: "10px", width: "80px" }} />
      </label>
      <br /><br />

      <label>
        Facade Count:
        <input type="number" name="facedeCount" value={formData.facedeCount} onChange={handleChange} min={0} style={{ marginLeft: "10px", width: "50px" }} />
      </label>
      <br /><br />

      <label>
        Construction Year:
        <input
          type="number"
          name="buildingConstructionYear"
          value={formData.buildingConstructionYear}
          onChange={handleChange}
          min={1800}
          max={new Date().getFullYear()}
          style={{ marginLeft: "10px", width: "80px" }}
        />
      </label>
      <br /><br />

      <label>
        Building Condition:
        <select name="buildingCondition" value={formData.buildingCondition} onChange={handleChange} style={{ marginLeft: "10px" }}>
          <option value="AS_NEW">As New</option>
          <option value="GOOD">Good</option>
          <option value="RENOVATION_NEEDED">Renovation Needed</option>
          <option value="TO_RESTORE">To Restore</option>
        </select>
      </label>
      <br /><br />

      <label>
        Kitchen Type:
        <select name="kitchenType" value={formData.kitchenType} onChange={handleChange} style={{ marginLeft: "10px" }}>
          <option value="HYPER_EQUIPPED">Hyper Equipped</option>
          <option value="EQUIPPED">Equipped</option>
          <option value="SIMPLE">Simple</option>
          <option value="NOT_INSTALLED">Not Installed</option>
        </select>
      </label>
      <br /><br />

      <label>
        Heating Type:
        <select name="heatingType" value={formData.heatingType} onChange={handleChange} style={{ marginLeft: "10px" }}>
          <option value="ELECTRIC">Electric</option>
          <option value="GAS">Gas</option>
          <option value="NONE">None</option>
        </select>
      </label>
      <br /><br />

      <label>
        Flood Zone Type:
        <select name="floodZoneType" value={formData.floodZoneType} onChange={handleChange} style={{ marginLeft: "10px" }}>
          <option value="NON_FLOOD_ZONE">Non Flood Zone</option>
          <option value="FLOOD_ZONE">Flood Zone</option>
        </select>
      </label>
      <br /><br />

      <label>
        EPC Score:
        <select name="epcScore" value={formData.epcScore} onChange={handleChange} style={{ marginLeft: "10px" }}>
          <option value="A_plus">A+</option>
          <option value="A">A</option>
          <option value="B">B</option>
          <option value="C">C</option>
          <option value="D">D</option>
          <option value="E">E</option>
          <option value="F">F</option>
          <option value="G">G</option>
        </select>
      </label>
      <br /><br />

      <label style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <input type="checkbox" name="hasLivingRoom" checked={formData.hasLivingRoom} onChange={handleChange} />
        Has Living Room
      </label>
      <br />

      <label style={{ display: "flex", alignItems: "center", gap: "8px" }}>
        <input type="checkbox" name="hasTerrace" checked={formData.hasTerrace} onChange={handleChange} />
        Has Terrace
      </label>
      <br /><br />

      <button
        type="submit"
        disabled={loading}
        style={{ padding: "10px 20px", fontWeight: "bold", cursor: "pointer" }}
      >
        {loading ? "Predicting..." : "Predict"}
      </button>

      {error && <p style={{ color: "red" }}>{error}</p>}

      {results.all && results.top && (
        <div style={{ marginTop: "20px", display: "flex", gap: "20px", flexWrap: "wrap" }}>
          <ResultCard title="Prediction using all features" value={results.all} />
          <ResultCard title="Prediction using top 30 features" value={results.top} />
        </div>
      )}
    </form>
  );
};

export default PropertyForm;
