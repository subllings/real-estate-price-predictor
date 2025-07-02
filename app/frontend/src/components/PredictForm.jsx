import React, { useState } from 'react';
import PredictChart from './PredictChart';

const initialState = {
  longitude: -122.23,
  latitude: 37.88,
  housing_median_age: 41,
  total_rooms: 5,
  total_bedrooms: 129,
  population: 322,
  households: 126,
  median_income: 8.3252,
  ocean_proximity: 'NEAR BAY',
};

function PredictForm() {
  const [formData, setFormData] = useState(initialState);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Erreur inconnue');

      setPrediction(data.prediction);
      setError(null);
    } catch (err) {
      setError(err.message);
      setPrediction(null);
    }
  };

  return (
    <div style={{
      maxWidth: '500px',
      margin: '40px auto',
      padding: '30px',
      border: '1px solid #ccc',
      borderRadius: '8px',
      boxShadow: '0 0 10px rgba(0,0,0,0.1)',
      fontFamily: 'Arial'
    }}>
      <h2 style={{ textAlign: 'center' }}>🏡 House Price Predictor</h2>
      <form onSubmit={handleSubmit}>
        {Object.keys(formData).map((key) => (
          <div key={key} style={{ marginBottom: '15px' }}>
            <label style={{ display: 'block', fontWeight: 'bold', marginBottom: '5px' }}>
              {key.replace(/_/g, ' ')}:
            </label>
            {key === 'ocean_proximity' ? (
              <select
                name={key}
                value={formData[key]}
                onChange={handleChange}
                style={{ width: '100%', padding: '8px' }}
              >
                <option value="NEAR BAY">NEAR BAY</option>
                <option value="<1H OCEAN">&lt;1H OCEAN</option>
                <option value="INLAND">INLAND</option>
                <option value="NEAR OCEAN">NEAR OCEAN</option>
                <option value="ISLAND">ISLAND</option>
              </select>
            ) : (
              <input
                type="number"
                name={key}
                value={formData[key]}
                onChange={handleChange}
                step="any"
                style={{ width: '100%', padding: '8px' }}
              />
            )}
            
          </div>
        ))}

        <button type="submit" style={{
          width: '100%',
          padding: '10px',
          backgroundColor: '#007bff',
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: 'pointer'
        }}>
          Predict
        </button>
      </form>

      {prediction && (
        <p style={{ color: 'green', fontWeight: 'bold', textAlign: 'center', marginTop: '20px' }}>
          ✅ Predicted House Price: {Math.round(prediction)} €
        </p>
      )}
      {error && (
        <p style={{ color: 'red', textAlign: 'center', marginTop: '20px' }}>
          ❌ {error}
        </p>
      )}
      <PredictChart data={formData} />
    </div>
  );
}

export default PredictForm;
