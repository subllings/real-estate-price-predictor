import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './EsgSummary.css';

const StrategicAnalysisConclusion = ({ formData, detailedEsgData, esgAnalysisAvailable }) => {
  const [strategicSummary, setStrategicSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // API endpoint for strategic analysis summary
  const STRATEGIC_ANALYSIS_API_URL = 'https://api-azure-llm-v2.azurewebsites.net/strategic-analysis-summary';

  // Fetch strategic analysis summary from API
  useEffect(() => {
    const fetchStrategicSummary = async () => {
      if (!formData || !detailedEsgData) return;

      setLoading(true);
      setError(null);

      try {
        const requestData = {
          price_prediction: detailedEsgData.estimated_price || 400000,
          esg_summary: {
            environment: detailedEsgData.esg_scores?.environmental || 7.0,
            social: detailedEsgData.esg_scores?.social || 7.0,
            governance: detailedEsgData.esg_scores?.governance || 7.0,
            overall: detailedEsgData.esg_scores?.overall || 7.0
          },
          property_features: {
            propertyType: formData.propertyType || 'HOUSE',
            subtype: formData.subtype || 'HOUSE',
            province: formData.province || 'Unknown',
            locality: formData.locality || 'Unknown',
            postCode: formData.postCode || '1000',
            bedroomCount: formData.bedroomCount || 1,
            bathroomCount: formData.bathroomCount || 1,
            toiletCount: formData.toiletCount || 1,
            roomCount: formData.roomCount || 1,
            habitableSurface: formData.habitableSurface || 100,
            facedeCount: formData.facedeCount || 1,
            buildingConstructionYear: formData.buildingConstructionYear || 1990,
            buildingCondition: formData.buildingCondition || 'GOOD',
            kitchenType: formData.kitchenType || 'EQUIPPED',
            heatingType: formData.heatingType || 'GAS',
            floodZoneType: formData.floodZoneType || 'NON_FLOOD_ZONE',
            epcScore: formData.epcScore || 'C',
            hasLivingRoom: formData.hasLivingRoom || true,
            hasTerrace: formData.hasTerrace || false
          },
          strategic_goals: 'invest',
          agent_insights: [
            {
              agent: 'Energy Analysis',
              summary: `EPC ${formData.epcScore} property with ${formData.heatingType} heating`
            },
            {
              agent: 'Market Analysis',
              summary: `Property in ${formData.locality}, ${formData.province}`
            }
          ]
        };

        const response = await axios.post(STRATEGIC_ANALYSIS_API_URL, requestData);
        setStrategicSummary(response.data);
      } catch (err) {
        console.error('Strategic analysis API error:', err);
        setError('Failed to fetch strategic analysis. Using fallback data.');
      } finally {
        setLoading(false);
      }
    };

    fetchStrategicSummary();
  }, [formData, detailedEsgData]);

  // Render Financial Impact Dashboard
  const renderFinancialImpactDashboard = () => {
    if (!detailedEsgData || !detailedEsgData.financial_impact) return null;

    const { financial_impact } = detailedEsgData;
    
    return (
      <div className="strategic-dashboard-card">
        <h3>Financial Impact Dashboard</h3>
        <div className="financial-metrics-grid">
          <div className="metric-card">
            <div className="metric-value">{financial_impact.energy_cost_annual || 'N/A'}</div>
            <div className="metric-label">Annual Energy Cost</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{financial_impact.improvement_cost_estimate || 'N/A'}</div>
            <div className="metric-label">Improvement Investment</div>
          </div>
          <div className="metric-card">
            <div className="metric-value">{financial_impact.roi_potential || 'N/A'}</div>
            <div className="metric-label">ROI Potential</div>
          </div>
        </div>
      </div>
    );
  };

  // Render Compliance Status Dashboard
  const renderComplianceStatusDashboard = () => {
    if (!detailedEsgData || !detailedEsgData.compliance_status) return null;

    const { compliance_status } = detailedEsgData;
    
    const getStatusColor = (status) => {
      switch (status) {
        case 'Compliant': return '#28a745';
        case 'Needs Review': return '#ffc107';
        case 'Non-Compliant': return '#dc3545';
        default: return '#6c757d';
      }
    };

    return (
      <div className="strategic-dashboard-card">
        <h3>Compliance Status Dashboard</h3>
        <div className="compliance-grid">
          {Object.entries(compliance_status).map(([key, status]) => (
            <div key={key} className="compliance-item">
              <div className="compliance-header">
                <span className="compliance-label">{key.replace(/_/g, ' ').toUpperCase()}</span>
                <span 
                  className="compliance-status"
                  style={{ backgroundColor: getStatusColor(status) }}
                >
                  {status}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Render Key Recommendations Dashboard
  const renderKeyRecommendationsDashboard = () => {
    if (!strategicSummary || !strategicSummary.key_insights) return null;

    return (
      <div className="strategic-dashboard-card">
        <h3>Key Recommendations Dashboard</h3>
        <div className="recommendations-grid">
          {strategicSummary.key_insights.map((insight, index) => (
            <div key={index} className="recommendation-item">
              <div className="recommendation-priority">
                <span className="priority-badge">P{index + 1}</span>
              </div>
              <div className="recommendation-content">
                <p>{insight}</p>
              </div>
              <div className="recommendation-actions">
                <button className="action-btn primary">Execute</button>
                <button className="action-btn secondary">Learn More</button>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  // Render Confidence & Timeline Dashboard
  const renderConfidenceTimelineDashboard = () => {
    const confidenceScore = strategicSummary?.confidence_score || 0;
    const confidencePercentage = Math.round(confidenceScore * 100);
    
    const getConfidenceColor = (score) => {
      if (score >= 0.8) return '#28a745';
      if (score >= 0.6) return '#ffc107';
      return '#dc3545';
    };

    return (
      <div className="strategic-dashboard-card">
        <h3>Analysis Quality & Timeline</h3>
        <div className="confidence-timeline-grid">
          <div className="confidence-section">
            <div className="confidence-score">
              <div 
                className="confidence-circle"
                style={{ borderColor: getConfidenceColor(confidenceScore) }}
              >
                <span className="confidence-percentage">{confidencePercentage}%</span>
              </div>
              <div className="confidence-label">Analysis Confidence</div>
            </div>
          </div>
          <div className="timeline-section">
            <div className="timeline-item">
              <div className="timeline-badge short-term">ST</div>
              <div className="timeline-content">
                <div className="timeline-title">Short Term (1-3 months)</div>
                <div className="timeline-desc">Market analysis, documentation review</div>
              </div>
            </div>
            <div className="timeline-item">
              <div className="timeline-badge medium-term">MT</div>
              <div className="timeline-content">
                <div className="timeline-title">Medium Term (3-12 months)</div>
                <div className="timeline-desc">Energy improvements, compliance updates</div>
              </div>
            </div>
            <div className="timeline-item">
              <div className="timeline-badge long-term">LT</div>
              <div className="timeline-content">
                <div className="timeline-title">Long Term (1-5 years)</div>
                <div className="timeline-desc">Strategic repositioning, major renovations</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  };

  // Fallback ESG calculation for initial display
  const getFallbackEsgData = () => {
    if (!formData) return null;

    const epcScores = {
      'A_plus': 9.0, 'A': 8.5, 'B': 7.5, 'C': 6.5, 'D': 5.5, 'E': 4.5, 'F': 3.5, 'G': 2.5
    };
    
    const environmental = epcScores[formData.epcScore] || 6.0;
    const social = formData.locality && ['Antwerpen', 'Brussels', 'Gent', 'Brugge', 'Leuven'].includes(formData.locality) ? 8.0 : 7.0;
    const governance = formData.buildingConstructionYear > 2000 ? 7.5 : 6.5;
    const overall = ((environmental + social + governance) / 3).toFixed(1);

    return {
      environment: environmental,
      social: social,
      governance: governance,
      overall: parseFloat(overall)
    };
  };

  // Get ESG scores from API or fallback
  const esgScores = strategicSummary ? {
    environment: detailedEsgData?.esg_scores?.environmental || 7.0,
    social: detailedEsgData?.esg_scores?.social || 7.0,
    governance: detailedEsgData?.esg_scores?.governance || 7.0,
    overall: detailedEsgData?.esg_scores?.overall || 7.0
  } : getFallbackEsgData();

  // Get insights from API or fallback
  const insights = strategicSummary ? {
    environment: strategicSummary.key_insights.filter(insight => 
      insight.toLowerCase().includes('energy') || 
      insight.toLowerCase().includes('environmental') ||
      insight.toLowerCase().includes('epc')
    ),
    social: strategicSummary.key_insights.filter(insight => 
      insight.toLowerCase().includes('social') || 
      insight.toLowerCase().includes('location') ||
      insight.toLowerCase().includes('community')
    ),
    governance: strategicSummary.key_insights.filter(insight => 
      insight.toLowerCase().includes('governance') || 
      insight.toLowerCase().includes('compliance') ||
      insight.toLowerCase().includes('regulation')
    )
  } : {
    environment: [`EPC ${formData?.epcScore?.replace('_', '+') || 'N/A'} energy rating`, `${formData?.heatingType?.replace('_', ' ') || 'Standard'} heating system`],
    social: [`Located in ${formData?.locality || 'Unknown'}, ${formData?.province || 'Belgium'}`, `${formData?.bedroomCount || 1} bedroom property`],
    governance: [`Built in ${formData?.buildingConstructionYear || 'N/A'}`, `${formData?.buildingCondition?.replace('_', ' ') || 'Good'} condition`]
  };

  // Function to convert score to letter grade
  const getScoreLetter = (score) => {
    if (score >= 8.5) return 'A+';
    if (score >= 7.5) return 'A';
    if (score >= 6.5) return 'B+';
    if (score >= 5.5) return 'B';
    if (score >= 4.5) return 'C+';
    if (score >= 3.5) return 'C';
    return 'D';
  };

  if (!formData || !esgScores) {
    return (
      <div className="strategic-analysis-conclusion">
        <div className="strategic-header">
          <h2>Conclusion of Strategic Analysis</h2>
          <div className="strategic-loading">Fill the form to see strategic analysis</div>
        </div>
      </div>
    );
  }

  const scoreLetter = getScoreLetter(esgScores.overall);

  return (
    <div className="strategic-analysis-conclusion">
      <div className="strategic-header">
        <h2>Conclusion of Strategic Analysis</h2>
        <p className="strategic-subtitle">
          Comprehensive analysis combining market data, ESG assessment, and strategic recommendations
        </p>
        {strategicSummary && (
          <div className="analysis-timestamp">
            <span>Analysis completed: {strategicSummary.timestamp}</span>
          </div>
        )}
      </div>

      {loading && (
        <div className="strategic-analysis-loading">
          <div className="loading-spinner"></div>
          <p>Generating strategic analysis summary...</p>
        </div>
      )}

      {error && (
        <div className="strategic-analysis-error">
          <span className="error-icon">Error:</span>
          <span>{error}</span>
        </div>
      )}

      {strategicSummary && (
        <div className="strategic-summary-section">
          <div className="strategic-summary-card">
            <h3>Executive Summary</h3>
            <div className="summary-content">
              <p>{strategicSummary.summary}</p>
            </div>
          </div>
        </div>
      )}

      <div className="strategic-dashboards-grid">
        {renderFinancialImpactDashboard()}
        {renderComplianceStatusDashboard()}
        {renderKeyRecommendationsDashboard()}
        {renderConfidenceTimelineDashboard()}
      </div>

      {esgScores && (
        <div className="esg-scores-summary">
          <h3>ESG Scores Summary</h3>
          <div className="esg-scores-grid">
            <div className="esg-score-item">
              <div className="score-value">{esgScores.environment}/10</div>
              <div className="score-label">Environmental</div>
            </div>
            <div className="esg-score-item">
              <div className="score-value">{esgScores.social}/10</div>
              <div className="score-label">Social</div>
            </div>
            <div className="esg-score-item">
              <div className="score-value">{esgScores.governance}/10</div>
              <div className="score-label">Governance</div>
            </div>
            <div className="esg-score-item overall">
              <div className="score-value">{esgScores.overall}/10</div>
              <div className="score-label">Overall ESG</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default StrategicAnalysisConclusion;
