import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { STRATEGIC_SUMMARY_API_URL } from '../../config/api.js';
import './StrategicSummary.css';

const StrategicSummary = ({ propertyData, predictionData, esgData }) => {
  const [strategicData, setStrategicData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (propertyData && predictionData && esgData) {
      fetchStrategicSummary();
    }
  }, [propertyData, predictionData, esgData]);

  const fetchStrategicSummary = async () => {
    setLoading(true);
    setError(null);

    try {
      // Prepare the request data according to the backend schema
      const requestData = {
        price_prediction: predictionData.prediction || predictionData.predictionAll || 0,
        esg_summary: {
          environment: esgData?.esg_scores?.environmental || 7.0,
          social: esgData?.esg_scores?.social || 7.0,
          governance: esgData?.esg_scores?.governance || 7.0,
          overall: esgData?.overall_grade || "B+"
        },
        property_features: {
          propertyType: propertyData.propertyType || "HOUSE",
          subtype: propertyData.subtype || "villa",
          province: propertyData.province || "",
          locality: propertyData.locality || "",
          postCode: propertyData.postCode || "",
          bedroomCount: propertyData.bedroomCount || 3,
          bathroomCount: propertyData.bathroomCount || 1,
          toiletCount: propertyData.toiletCount || 1,
          roomCount: propertyData.roomCount || 6,
          habitableSurface: propertyData.habitableSurface || 150,
          facedeCount: propertyData.facedeCount || 2,
          buildingConstructionYear: propertyData.buildingConstructionYear || 1990,
          buildingCondition: propertyData.buildingCondition || "GOOD",
          kitchenType: propertyData.kitchenType || "INSTALLED",
          heatingType: propertyData.heatingType || "GAS",
          floodZoneType: propertyData.floodZoneType || "NO_FLOOD_ZONE",
          epcScore: propertyData.epcScore || "C",
          hasLivingRoom: propertyData.hasLivingRoom !== false,
          hasTerrace: propertyData.hasTerrace !== false
        },
        strategic_goals: "invest", // Default goal, could be made dynamic
        agent_insights: [
          {
            agent: "PriceAnalyzer",
            summary: `Estimated property value: €${(predictionData.prediction || predictionData.predictionAll || 0).toLocaleString('en-EU')}`
          },
          {
            agent: "ESGAnalyzer", 
            summary: `ESG Rating: ${esgData?.overall_grade || "B+"} (${esgData?.esg_scores?.overall || 7.0}/10)`
          }
        ]
      };

      console.log('Sending strategic summary request:', requestData);

      const response = await axios.post(STRATEGIC_SUMMARY_API_URL, requestData, {
        headers: {
          'Content-Type': 'application/json',
        },
      });

      setStrategicData(response.data);
    } catch (err) {
      console.error('Strategic summary error:', err);
      setError('Failed to generate strategic summary. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestionClick = (suggestion) => {
    console.log('Suggestion clicked:', suggestion);
    // In future, could trigger specific actions based on suggestion.action
    alert(`Feature coming soon: ${suggestion.title}`);
  };

  if (loading) {
    return (
      <div className="strategic-summary loading">
        <div className="strategic-header">
          <h3>Strategic Analysis</h3>
        </div>
        <div className="loading-content">
          <div className="spinner"></div>
          <p>Analyzing market position and generating strategic recommendations...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="strategic-summary error">
        <div className="strategic-section">
          <h4>Strategic Analysis</h4>
        </div>
        <div className="error-content">
          <p>{error}</p>
          <button onClick={fetchStrategicSummary} className="retry-btn">
            Retry Analysis
          </button>
        </div>
      </div>
    );
  }

  if (!strategicData) {
    return (
      <div className="strategic-summary">
        <div className="strategic-section">
          <h4>Strategic Analysis</h4>
        </div>
        <p className="no-data">Complete property assessment to see strategic recommendations.</p>
      </div>
    );
  }

  return (
    <div className="strategic-summary">
      <div className="strategic-section">
        <h4>Strategic Analysis</h4>
        <div className="confidence-badge">
          Confidence: {Math.round(strategicData.confidence_score * 100)}%
        </div>
      </div>

      <div className="strategic-content">
        <section className="strategic-section">
          <h4>Market Position</h4>
          <p>{strategicData.strategic_positioning}</p>
        </section>

        <section className="strategic-section">
          <h4>Recommended Actions</h4>
          <div className="action-list">
            <div className="action-item">
              <span className="action-number">1.</span>
              <div className="action-content">
                <strong>Market Analysis</strong>
                <p>Property positioned competitively with estimated value of €{(predictionData.prediction || predictionData.predictionAll || 0).toLocaleString('en-EU')} in current market conditions.</p>
              </div>
            </div>
            <div className="action-item">
              <span className="action-number">2.</span>
              <div className="action-content">
                <strong>Investment Strategy</strong>
                <p>Focus on value-add improvements and market positioning to maximize returns while maintaining competitive advantages.</p>
              </div>
            </div>
            <div className="action-item">
              <span className="action-number">3.</span>
              <div className="action-content">
                <strong>Future Planning</strong>
                <p>Develop long-term maintenance schedule and consider market trends for sustained property value growth.</p>
              </div>
            </div>
          </div>
        </section>

        <section className="strategic-section">
          <h4>Next Steps</h4>
          <div className="suggestions">
            {strategicData.clickable_suggestions?.map((suggestion) => (
              <button
                key={suggestion.id}
                className="suggestion-btn"
                onClick={() => handleSuggestionClick(suggestion)}
              >
                <span className="suggestion-title">{suggestion.title}</span>
                <span className="suggestion-desc">{suggestion.description}</span>
              </button>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
};

export default StrategicSummary;
