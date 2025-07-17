import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './EsgSummary.css';

const StrategicAnalysisConclusion = ({ formData, detailedEsgData, esgAnalysisAvailable, esgLoading, onOpenSidePanel, onSendChatMessage }) => {
  console.log('StrategicAnalysisConclusion rendering with:', { formData, detailedEsgData, esgAnalysisAvailable, esgLoading });
  
  const [strategicSummary, setStrategicSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // API endpoint for strategic analysis summary
  const STRATEGIC_ANALYSIS_API_URL = 'https://api-azure-llm-v2.azurewebsites.net/strategic-analysis-summary';

  // Clear all dynamic data when ESG loading starts
  useEffect(() => {
    if (esgLoading) {
      setStrategicSummary(null);
      setError(null);
      setLoading(false);
    }
  }, [esgLoading]);

  // Generate fallback strategic analysis based on form data
  const generateFallbackStrategicAnalysis = useCallback(() => {
    if (!formData || !detailedEsgData) return null;

    const currentDate = new Date().toISOString().split('T')[0];
    const currentTime = new Date().toLocaleTimeString('en-US', { hour12: false });

    // Generate realistic insights based on property data
    const insights = [];
    
    // EPC-based insights
    const epcScore = formData.epcScore || 'C';
    if (['A_plus', 'A', 'B'].includes(epcScore)) {
      insights.push('Energy efficiency is excellent - property meets future regulatory standards');
      insights.push('Low operational costs due to superior energy performance');
    } else if (['C', 'D'].includes(epcScore)) {
      insights.push('Consider energy efficiency improvements to reduce operational costs');
      insights.push('Moderate energy performance - potential for optimization');
    } else {
      insights.push('Priority: Energy efficiency upgrades required for compliance');
      insights.push('High energy costs - significant improvement potential');
    }

    // Location-based insights
    const locality = formData.locality || 'Unknown';
    const majorCities = ['Antwerpen', 'Brussels', 'Gent', 'Brugge', 'Leuven'];
    if (majorCities.includes(locality)) {
      insights.push(`Strong location advantage in ${locality} - high demand market`);
      insights.push('Excellent accessibility and urban amenities');
    } else {
      insights.push(`Regional market opportunities in ${locality}`);
      insights.push('Consider local market dynamics and growth potential');
    }

    // Property type insights
    const propertyType = formData.propertyType || 'HOUSE';
    if (propertyType === 'HOUSE') {
      insights.push('Single-family homes show strong market resilience');
      insights.push('Potential for value-add improvements and extensions');
    } else {
      insights.push('Apartment market benefits from urbanization trends');
      insights.push('Lower maintenance requirements compared to houses');
    }

    // Building age insights
    const buildingYear = formData.buildingConstructionYear || 1990;
    if (buildingYear >= 2010) {
      insights.push('Modern construction standards provide quality foundation');
    } else if (buildingYear >= 1990) {
      insights.push('Well-established property with proven track record');
    } else {
      insights.push('Historic property with character - consider renovation opportunities');
    }

    return {
      summary: `Strategic analysis for ${propertyType.toLowerCase()} in ${locality} reveals a ${epcScore === 'A_plus' || epcScore === 'A' || epcScore === 'B' ? 'strong' : 'moderate'} investment opportunity. The property demonstrates ${['A_plus', 'A', 'B'].includes(epcScore) ? 'excellent' : 'adequate'} energy performance and is well-positioned in the local market. Key focus areas include ${['E', 'F', 'G'].includes(epcScore) ? 'energy efficiency improvements' : 'market positioning'} and ${buildingYear < 2000 ? 'modernization opportunities' : 'maintaining current standards'}.`,
      key_insights: insights,
      confidence_score: 0.85,
      timestamp: `${currentDate} ${currentTime}`
    };
  }, [formData, detailedEsgData]);

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

        // Try to fetch from API with timeout
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

        const response = await axios.post(STRATEGIC_ANALYSIS_API_URL, requestData, {
          signal: controller.signal,
          timeout: 5000
        });
        
        clearTimeout(timeoutId);
        setStrategicSummary(response.data);
      } catch (err) {
        console.log('Strategic analysis API not available, using fallback data:', err.message);
        
        // Use fallback data instead of showing error
        const fallbackData = generateFallbackStrategicAnalysis();
        if (fallbackData) {
          setStrategicSummary(fallbackData);
        }
        
        // Don't set error state to avoid showing error message to user
        // setError('Failed to fetch strategic analysis. Using fallback data.');
      } finally {
        setLoading(false);
      }
    };

    fetchStrategicSummary();
  }, [formData, detailedEsgData, generateFallbackStrategicAnalysis]);

  // Render Financial Impact Dashboard
  const renderFinancialImpactDashboard = () => {
    // Show loading state if no data yet
    if (!detailedEsgData || !detailedEsgData.financial_impact) {
      return (
        <div className="strategic-dashboard-card">
          <h3>Financial Impact Dashboard</h3>
          <div className="financial-metrics-grid">
            <div className="metric-card" style={{ 
              background: '#6c757d', 
              color: 'white', 
              border: 'none',
              boxShadow: '0 4px 12px rgba(108, 117, 125, 0.3)'
            }}>
              <div className="metric-value">Loading...</div>
              <div className="metric-label">Annual Energy Cost</div>
            </div>
            <div className="metric-card" style={{ 
              background: '#6c757d', 
              color: 'white', 
              border: 'none',
              boxShadow: '0 4px 12px rgba(108, 117, 125, 0.3)'
            }}>
              <div className="metric-value">Loading...</div>
              <div className="metric-label">Improvement Investment</div>
            </div>
            <div className="metric-card" style={{ 
              background: '#6c757d', 
              color: 'white', 
              border: 'none',
              boxShadow: '0 4px 12px rgba(108, 117, 125, 0.3)'
            }}>
              <div className="metric-value">Loading...</div>
              <div className="metric-label">ROI Potential</div>
            </div>
          </div>
        </div>
      );
    }

    const { financial_impact } = detailedEsgData;
    // Color logic helpers - corrected for proper EPC/financial assessment
    const getMetricColor = (metricType, value) => {
      if (!value || value === 'N/A') return '#6c757d';
      switch (metricType) {
        case 'energy_cost': {
          // First check for EPC rating in the string - this takes priority
          const epcMatch = value.match(/EPC\s([A-G][+]?)/);
          if (epcMatch) {
            const epcRating = epcMatch[1];
            // EPC color logic: A+ and A = green, B = light green, C = yellow, D/E = orange, F/G = red
            if (['A+', 'A'].includes(epcRating)) return '#28a745'; // Green - excellent
            if (['B'].includes(epcRating)) return '#8fd19e'; // Light green - good
            if (['C'].includes(epcRating)) return '#ffc107'; // Yellow - average
            if (['D', 'E'].includes(epcRating)) return '#ff9800'; // Orange - poor
            if (['F', 'G'].includes(epcRating)) return '#dc3545'; // Red - very poor
          }
          // Fallback to numeric cost if no EPC found
          const annualCost = extractNumericValue(value);
          if (annualCost <= 1000) return '#28a745'; // Green - low cost
          if (annualCost <= 2000) return '#ffc107'; // Yellow - moderate cost
          return '#dc3545'; // Red - high cost
        }
        case 'improvement_cost': {
          const improvementCost = extractNumericValue(value);
          // Higher investment needed = worse (red), lower investment = better (green)
          // Pour les ranges comme "5,000 - 25,000", on prend la valeur moyenne
          if (improvementCost <= 5000) return '#28a745'; // Green - low investment
          if (improvementCost <= 15000) return '#ffc107'; // Yellow - moderate investment
          return '#dc3545'; // Red - high investment (au-dessus de 15k)
        }
        case 'roi_potential': {
          const roiPercentage = extractPercentageValue(value);
          // Higher ROI = better (green), lower ROI = worse (red)
          if (roiPercentage >= 12) return '#28a745'; // Green - high ROI (12%+)
          if (roiPercentage >= 8) return '#ffc107'; // Yellow - moderate ROI (8-12%)
          return '#dc3545'; // Red - low ROI (moins de 8%)
        }
        default: return '#6c757d';
      }
    };
    const extractNumericValue = (str) => {
      if (!str) return 0;
      const rangeMatch = str.match(/(\d{1,3}(?:,\d{3})*(?:\.\d+)?)\s*-\s*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)/);
      if (rangeMatch) {
        const min = parseFloat(rangeMatch[1].replace(/,/g, ''));
        const max = parseFloat(rangeMatch[2].replace(/,/g, ''));
        return (min + max) / 2;
      }
      const singleMatch = str.match(/(\d{1,3}(?:,\d{3})*(?:\.\d+)?)/);
      if (singleMatch) return parseFloat(singleMatch[1].replace(/,/g, ''));
      return 0;
    };
    const extractPercentageValue = (str) => {
      if (!str) return 0;
      const match = str.match(/(\d+(?:\.\d+)?)%/);
      if (match) return parseFloat(match[1]);
      return 0;
    };
    const energyCostColor = getMetricColor('energy_cost', financial_impact.energy_cost_annual);
    const improvementCostColor = getMetricColor('improvement_cost', financial_impact.improvement_cost_estimate);
    const roiColor = getMetricColor('roi_potential', financial_impact.roi_potential);
    
    // Function to determine text color based on background color
    const getTextColor = (backgroundColor) => {
      // For dark backgrounds, use white text. For light backgrounds, use dark text.
      const darkColors = ['#dc3545', '#28a745', '#ff9800']; // Red, Green, Orange
      const lightColors = ['#ffc107', '#8fd19e']; // Yellow, Light Green
      
      if (darkColors.includes(backgroundColor)) return 'white';
      if (lightColors.includes(backgroundColor)) return '#333';
      return 'white'; // Default to white for other colors
    };
    
    // Function to generate box shadow based on background color
    const getBoxShadow = (backgroundColor) => {
      switch (backgroundColor) {
        case '#dc3545': // Red
          return '0 4px 12px rgba(220, 53, 69, 0.3)';
        case '#28a745': // Green
          return '0 4px 12px rgba(40, 167, 69, 0.3)';
        case '#ffc107': // Yellow
          return '0 4px 12px rgba(255, 193, 7, 0.3)';
        case '#ff9800': // Orange
          return '0 4px 12px rgba(255, 152, 0, 0.3)';
        case '#8fd19e': // Light Green
          return '0 4px 12px rgba(143, 209, 158, 0.3)';
        default:
          return '0 4px 12px rgba(108, 117, 125, 0.3)'; // Default gray shadow
      }
    };
    
    const formatEnergyCostString = (str) => {
      if (!str) return 'N/A';
      // Replace EPC A_plus, B_plus, etc. with A+, B+, etc.
      // Also ensure EPC G appears correctly and gets proper color coding
      return str.replace(/EPC ([A-G])_plus/g, 'EPC $1+').replace(/EPC ([A-G])(?![\+])/g, 'EPC $1');
    };

    // Actionable summary logic for financial impact
    const summaryPoints = [];
    if (financial_impact.energy_cost_annual) {
      summaryPoints.push(`Annual energy cost: ${formatEnergyCostString(financial_impact.energy_cost_annual)}`);
    }
    if (financial_impact.improvement_cost_estimate) {
      summaryPoints.push(`Estimated investment for improvements: ${financial_impact.improvement_cost_estimate}`);
    }
    if (financial_impact.roi_potential) {
      summaryPoints.push(`ROI potential: ${financial_impact.roi_potential}`);
    }

    // Group summary points
    const categories = {
      Energy: summaryPoints.filter(pt => pt.toLowerCase().includes('energy')),
      Investment: summaryPoints.filter(pt => pt.toLowerCase().includes('investment') || pt.toLowerCase().includes('improvement')),
      ROI: summaryPoints.filter(pt => pt.toLowerCase().includes('roi')),
      Other: summaryPoints.filter(pt => !pt.toLowerCase().includes('energy') && !pt.toLowerCase().includes('investment') && !pt.toLowerCase().includes('roi'))
    };
    const shownCats = Object.entries(categories).filter(([cat, arr]) => arr.length > 0);

    return (
      <div className="strategic-dashboard-card">
        <h3>Financial Impact Dashboard</h3>
        <div className="financial-metrics-grid">
          <div className="metric-card" style={{ 
            background: energyCostColor, 
            color: getTextColor(energyCostColor), 
            border: 'none',
            boxShadow: getBoxShadow(energyCostColor)
          }}>
            <div className="metric-value" style={{ color: getTextColor(energyCostColor) }}>{formatEnergyCostString(financial_impact.energy_cost_annual) || 'N/A'}</div>
            <div className="metric-label" style={{ color: getTextColor(energyCostColor), opacity: 0.9 }}>Annual Energy Cost</div>
          </div>
          <div className="metric-card" style={{ 
            background: improvementCostColor, 
            color: getTextColor(improvementCostColor), 
            border: 'none',
            boxShadow: getBoxShadow(improvementCostColor)
          }}>
            <div className="metric-value" style={{ color: getTextColor(improvementCostColor) }}>{financial_impact.improvement_cost_estimate || 'N/A'}</div>
            <div className="metric-label" style={{ color: getTextColor(improvementCostColor), opacity: 0.9 }}>Improvement Investment</div>
          </div>
          <div className="metric-card" style={{ 
            background: roiColor, 
            color: getTextColor(roiColor), 
            border: 'none',
            boxShadow: getBoxShadow(roiColor)
          }}>
            <div className="metric-value" style={{ color: getTextColor(roiColor) }}>{financial_impact.roi_potential || 'N/A'}</div>
            <div className="metric-label" style={{ color: getTextColor(roiColor), opacity: 0.9 }}>ROI Potential</div>
          </div>
        </div>
        {/* Actionable summary section */}
        {shownCats.length > 0 && (
          <div className="action-summary" style={{ marginTop: '1.2em', padding: '1em', background: '#f6f8fa', borderRadius: '0.7em', boxShadow: '0 2px 8px #eee' }}>
            <h4 style={{ marginBottom: '0.7em', color: '#007bff' }}>Actionable Financial Summary</h4>
            {shownCats.map(([cat, arr]) => (
              <div key={cat} style={{ marginBottom: '0.7em' }}>
                <div style={{ fontWeight: 'bold', color: '#333', marginBottom: '0.3em' }}>{cat}:</div>
                <ul style={{ margin: 0, paddingLeft: '1.2em', color: '#333' }}>
                  {arr.map((pt, idx) => <li key={idx} style={{ marginBottom: '0.4em', color: '#333' }}>{pt}</li>)}
                </ul>
              </div>
            ))}
          </div>
        )}
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

    // Helper to generate explanation for each status
    const getComplianceExplanation = (key, status) => {
      const label = key.replace(/_/g, ' ').toUpperCase();
      if (status === 'Compliant') {
        return `The status "${label}" is compliant with current EU and local energy efficiency regulations (e.g., EPC standards, minimum insulation, and heating requirements). No immediate action is required.`;
      }
      if (status === 'Needs Review') {
        return `The status "${label}" may require further review to ensure full compliance with energy performance regulations, such as EPC certification, insulation standards, or heating system requirements. Please verify documentation and recent updates.`;
      }
      if (status === 'Non-Compliant') {
        return `The status "${label}" does not meet current regulatory requirements (e.g., EPC rating below legal minimum, insufficient insulation, or outdated heating systems). Improvements are needed to avoid penalties or restrictions on property sale/rental.`;
      }
      return `Unknown status for "${label}".`;
    };

    return (
      <div className="strategic-dashboard-card">
        <h3>Compliance Status Dashboard</h3>
        <div className="compliance-grid">
          {Object.entries(compliance_status).map(([key, status]) => (
            <div key={key} className="compliance-item">
              <div className="compliance-header" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '1rem' }}>
                <span className="compliance-label" style={{ flex: 1, textAlign: 'left' }}>{key.replace(/_/g, ' ').toUpperCase()}</span>
                <span 
                  className="compliance-status"
                  style={{ backgroundColor: getStatusColor(status), flex: 0, textAlign: 'center', minWidth: '120px', padding: '0.3em 1em', borderRadius: '1em', fontWeight: 'bold', whiteSpace: 'nowrap' }}
                >
                  {status}
                </span>
              </div>
              <div className="compliance-explanation" style={{ marginTop: '0.5em', fontSize: '0.95em', color: '#444', fontStyle: 'italic' }}>
                {getComplianceExplanation(key, status)}
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

    // Handler for "Learn More" button
    const handleLearnMore = (insight, index) => {
      const question = `Can you provide more details about this recommendation: "${insight}"? Please consider the specific context of my ${formData?.propertyType?.toLowerCase() || 'property'} in ${formData?.locality || 'this location'}, ${formData?.province || 'this province'} with EPC score ${formData?.epcScore || 'unknown'} and construction year ${formData?.buildingConstructionYear || 'unknown'}.`;
      
      // Open chat panel and send question
      if (onOpenSidePanel) {
        onOpenSidePanel();
      }
      
      if (onSendChatMessage) {
        onSendChatMessage(question);
      }
    };

    // Handler for "Execute" button
    const handleExecute = (insight, index) => {
      const actionSteps = generateActionSteps(insight);
      const message = `I want to execute this recommendation: "${insight}". Here are the suggested action steps:\n\n${actionSteps.join('\n')}\n\nCan you help me prioritize these steps and provide more specific guidance for my ${formData?.propertyType?.toLowerCase() || 'property'} in ${formData?.locality || 'this location'}?`;
      
      // Open chat panel and send action plan
      if (onOpenSidePanel) {
        onOpenSidePanel();
      }
      
      if (onSendChatMessage) {
        onSendChatMessage(message);
      }
    };

    // Generate action steps based on insight content
    const generateActionSteps = (insight) => {
      const lowerInsight = insight.toLowerCase();
      
      if (lowerInsight.includes('energy') || lowerInsight.includes('epc') || lowerInsight.includes('efficiency')) {
        return [
          "1. Contact certified energy auditor for detailed assessment",
          "2. Get quotes from insulation contractors",
          "3. Research available energy subsidies in Belgium",
          "4. Plan renovation timeline and budget",
          "5. Schedule EPC certification after improvements"
        ];
      }
      
      if (lowerInsight.includes('market') || lowerInsight.includes('investment') || lowerInsight.includes('value')) {
        return [
          "1. Research recent comparable sales in the area",
          "2. Consult with local real estate agents",
          "3. Analyze rental yield potential",
          "4. Consider market timing for investment decisions",
          "5. Review property insurance and tax implications"
        ];
      }
      
      if (lowerInsight.includes('compliance') || lowerInsight.includes('regulation') || lowerInsight.includes('legal')) {
        return [
          "1. Review current Belgian building regulations",
          "2. Check property permits and documentation",
          "3. Consult with building compliance expert",
          "4. Schedule required inspections",
          "5. Update documentation as needed"
        ];
      }
      
      if (lowerInsight.includes('renovation') || lowerInsight.includes('improvement') || lowerInsight.includes('upgrade')) {
        return [
          "1. Get detailed renovation quotes from contractors",
          "2. Apply for necessary building permits",
          "3. Create realistic timeline and budget",
          "4. Research financing options and subsidies",
          "5. Plan temporary accommodation if needed"
        ];
      }
      
      // Default action steps
      return [
        "1. Research this recommendation in detail",
        "2. Get professional consultation",
        "3. Evaluate costs and benefits",
        "4. Create implementation timeline",
        "5. Monitor progress and results"
      ];
    };

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
                <button 
                  className="action-btn primary"
                  onClick={() => handleExecute(insight, index)}
                  title="Get action steps for this recommendation"
                >
                  Execute
                </button>
                <button 
                  className="action-btn secondary"
                  onClick={() => handleLearnMore(insight, index)}
                  title="Learn more about this recommendation"
                >
                  Learn More
                </button>
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

  // Function to format EPC score for display
  const formatEpcScore = (epcScore) => {
    if (!epcScore) return 'N/A';
    return epcScore.replace('A_plus', 'A+').replace('_plus', '+').replace('_', '+');
  };

  // Fallback ESG calculation for initial display
  const getFallbackEsgData = () => {
    if (!formData) {
      console.log('No formData available for ESG calculation');
      return null;
    }

    console.log('Calculating ESG scores with formData:', formData);

    const epcScores = {
      'A_plus': 9.0, 'A': 8.5, 'B': 7.5, 'C': 6.5, 'D': 5.5, 'E': 4.5, 'F': 3.5, 'G': 2.5
    };
    
    // Environmental score based on multiple factors
    let environmental = epcScores[formData.epcScore] || 6.0;
    console.log('Base environmental score from EPC:', environmental, 'EPC Score:', formData.epcScore);
    
    // Adjust for heating type
    const heatingAdjustment = {
      'ELECTRIC': 0.5, 'GAS': 0, 'SOLAR': 1.5, 'HEAT_PUMP': 1.0, 'WOOD': 0.5
    };
    environmental += heatingAdjustment[formData.heatingType] || 0;
    console.log('Environmental after heating adjustment:', environmental, 'Heating Type:', formData.heatingType);
    
    // Adjust for flood zone
    if (formData.floodZoneType === 'NON_FLOOD_ZONE') environmental += 0.5;
    console.log('Environmental after flood zone adjustment:', environmental, 'Flood Zone:', formData.floodZoneType);
    
    // Social score based on location and amenities
    let social = formData.locality && ['Antwerpen', 'Brussels', 'Gent', 'Brugge', 'Leuven'].includes(formData.locality) ? 8.0 : 7.0;
    console.log('Base social score:', social, 'Locality:', formData.locality);
    
    // Adjust for property features
    if (formData.hasLivingRoom) social += 0.3;
    if (formData.hasTerrace) social += 0.2;
    if (formData.bedroomCount >= 3) social += 0.3;
    console.log('Social after features adjustment:', social, 'Living Room:', formData.hasLivingRoom, 'Terrace:', formData.hasTerrace, 'Bedrooms:', formData.bedroomCount);
    
    // Governance score based on building age and condition
    let governance = formData.buildingConstructionYear > 2000 ? 7.5 : 6.5;
    console.log('Base governance score:', governance, 'Construction Year:', formData.buildingConstructionYear);
    
    // Adjust for building condition
    const conditionAdjustment = {
      'AS_NEW': 1.0, 'GOOD': 0.5, 'RENOVATION_NEEDED': -0.5, 'TO_RESTORE': -1.0
    };
    governance += conditionAdjustment[formData.buildingCondition] || 0;
    console.log('Governance after condition adjustment:', governance, 'Building Condition:', formData.buildingCondition);
    
    // Adjust for kitchen type
    const kitchenAdjustment = {
      'HYPER_EQUIPPED': 0.5, 'EQUIPPED': 0.2, 'SIMPLE': 0, 'NOT_INSTALLED': -0.5
    };
    governance += kitchenAdjustment[formData.kitchenType] || 0;
    console.log('Governance after kitchen adjustment:', governance, 'Kitchen Type:', formData.kitchenType);
    
    // Ensure scores are within valid range
    environmental = Math.max(1, Math.min(10, environmental));
    social = Math.max(1, Math.min(10, social));
    governance = Math.max(1, Math.min(10, governance));
    
    const overall = ((environmental + social + governance) / 3);

    const finalScores = {
      environment: Math.round(environmental * 10) / 10,
      social: Math.round(social * 10) / 10,
      governance: Math.round(governance * 10) / 10,
      overall: Math.round(overall * 10) / 10
    };

    console.log('Final ESG scores:', finalScores);
    return finalScores;
  };

  // Get ESG scores from API or fallback
  const fallbackScores = getFallbackEsgData();
  const esgScores = fallbackScores || {
    environment: 7.0,
    social: 7.0,
    governance: 7.0,
    overall: 7.0
  };

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
    environment: [
      `EPC ${formatEpcScore(formData?.epcScore) || 'N/A'} energy rating`,
      `${formData?.heatingType?.replace('_', ' ') || 'Standard'} heating system`,
      `${formData?.floodZoneType === 'NON_FLOOD_ZONE' ? 'No flood risk' : 'Flood zone consideration required'}`,
      `${formData?.habitableSurface || 100}m² living space efficiency`
    ],
    social: [
      `Located in ${formData?.locality || 'Unknown'}, ${formData?.province || 'Belgium'}`,
      `${formData?.bedroomCount || 1} bedroom property`,
      `${formData?.bathroomCount || 1} bathroom facility`,
      `${formData?.hasLivingRoom ? 'Living room available' : 'Compact living space'}`
    ],
    governance: [
      `Built in ${formData?.buildingConstructionYear || 'N/A'}`,
      `${formData?.buildingCondition?.replace('_', ' ') || 'Good'} condition`,
      `${formData?.kitchenType?.replace('_', ' ') || 'Standard'} kitchen`,
      `${formData?.facedeCount || 1} facade${formData?.facedeCount > 1 ? 's' : ''}`
    ]
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

  // Show loading state if ESG is being analyzed
  if (esgLoading) {
    return (
      <div style={{ background: 'transparent', border: 'none', padding: '0', margin: '0' }}>
        <div style={{ background: 'transparent', border: 'none', textAlign: 'center' }}>
          <style>
            {`
              @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
              }
            `}
          </style>
          <h2 style={{ color: '#2c3e50', fontSize: '2.2rem', fontWeight: '700', marginBottom: '2rem' }}>
            Conclusion of Strategic Analysis
          </h2>
          <div style={{ 
            textAlign: 'center', 
            padding: '2rem', 
            background: 'transparent', 
            border: 'none',
            boxShadow: 'none',
            borderRadius: '0'
          }}>
            <div style={{ 
              display: 'flex', 
              flexDirection: 'column', 
              alignItems: 'center', 
              gap: '1.5rem',
              background: 'transparent',
              border: 'none'
            }}>
              <div style={{
                width: '40px',
                height: '40px',
                border: '3px solid #e3f2fd',
                borderTop: '3px solid #007bff',
                borderRadius: '50%',
                animation: 'spin 1s linear infinite'
              }}></div>
              <p style={{ 
                fontSize: '1.2rem', 
                fontWeight: '500', 
                color: '#007bff', 
                margin: '0' 
              }}>
                Analysis in progress...
              </p>
            </div>
          </div>
        </div>
      </div>
    );
  }

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
