import React, { useMemo } from 'react';
import './EsgSummary.css';

const EsgSummary = ({ formData, detailedEsgData, esgAnalysisAvailable }) => {
  
  // Calcul ESG en temps réel basé sur formData
  const esgCalculation = useMemo(() => {
    if (!formData) return null;

    let totalScore = 0;
    let maxScore = 100;

    // ENVIRONNEMENTAL (40 points max)
    let envScore = 0;
    
    // Classe énergétique (0-15 points)
    const epcScores = {
      'A_plus': 15, 'A': 13, 'B': 11, 'C': 8, 'D': 6, 'E': 4, 'F': 2, 'G': 0
    };
    envScore += epcScores[formData.epcScore] || 0;

    // Type de chauffage (0-10 points)
    const heatingScores = {
      'ELECTRIC': 10, 'GAS': 6, 'FUEL_OIL': 2, 'WOOD': 8, 'SOLAR': 12, 'NONE': 0
    };
    envScore += heatingScores[formData.heatingType] || 5;

    // Zone inondable (0-10 points)
    if (formData.floodZone === 'NON_FLOOD_ZONE') envScore += 10;
    else if (formData.floodZone === 'POSSIBLE_FLOOD_ZONE') envScore += 5;
    else envScore += 0;

    // Surface efficace (0-5 points)
    if (formData.livingSize <= 120) envScore += 5;
    else if (formData.livingSize <= 180) envScore += 3;
    else if (formData.livingSize <= 250) envScore += 1;
    else envScore += 0;

    // SOCIAL (35 points max)
    let socialScore = 0;
    
    // Localisation urbaine (0-15 points)
    const urbanCities = ['Antwerpen', 'Brussels', 'Gent', 'Brugge', 'Leuven', 'Namur', 'Mons', 'Charleroi', 'Liège'];
    if (urbanCities.includes(formData.locality)) socialScore += 15;
    else socialScore += 8;

    // Capacité familiale (0-10 points)
    if (formData.roomsCount >= 3) socialScore += 10;
    else if (formData.roomsCount >= 2) socialScore += 7;
    else if (formData.roomsCount >= 1) socialScore += 4;
    else socialScore += 0;

    // Commodités sociales (0-10 points)
    if (formData.hasLivingRoom) socialScore += 3;
    if (formData.hasTerrace) socialScore += 3;
    if (formData.bathroomsCount >= 2) socialScore += 2;
    if (formData.toiletsCount >= 2) socialScore += 2;

    // GOUVERNANCE (25 points max)
    let govScore = 0;
    
    // Conformité moderne (0-15 points)
    const currentYear = new Date().getFullYear();
    const age = currentYear - (formData.constructionYear || 1990);
    if (age <= 5) govScore += 15;
    else if (age <= 15) govScore += 12;
    else if (age <= 25) govScore += 8;
    else if (age <= 40) govScore += 5;
    else govScore += 2;

    // Condition bâtiment (0-10 points)
    const conditionScores = {
      'AS_NEW': 10, 'GOOD': 8, 'TO_RENOVATE': 4, 'TO_RESTORE': 2
    };
    govScore += conditionScores[formData.buildingCondition] || 5;

    // Calcul des scores finaux
    const envPercentage = Math.round((envScore / 40) * 100);
    const socialPercentage = Math.round((socialScore / 35) * 100);
    const govPercentage = Math.round((govScore / 25) * 100);
    const overallScore = Math.round(((envScore + socialScore + govScore) / 100) * 100);

    return {
      overall: overallScore,
      environment: envPercentage,
      social: socialPercentage,
      governance: govPercentage,
      rawScores: { envScore, socialScore, govScore }
    };
  }, [formData]);

  // Génération des insights en temps réel
  const insights = useMemo(() => {
    if (!formData) return { environment: [], social: [], governance: [] };

    const result = {
      environment: [],
      social: [],
      governance: []
    };

    // Insights environnementaux
    const epcScore = formData.epcScore?.replace('_', '+') || 'N/A';
    if (['A+', 'A', 'B'].includes(epcScore)) {
      result.environment.push(`Excellent energy efficiency (Class ${epcScore})`);
    } else if (['C', 'D'].includes(epcScore)) {
      result.environment.push(`Moderate energy efficiency (Class ${epcScore})`);
    } else {
      result.environment.push(`Energy efficiency needs improvement (Class ${epcScore})`);
    }

    if (formData.heatingType === 'ELECTRIC') {
      result.environment.push('Electric heating system');
    } else if (formData.heatingType === 'GAS') {
      result.environment.push('Gas heating system');
    } else if (formData.heatingType === 'SOLAR') {
      result.environment.push('Sustainable solar heating');
    }

    if (formData.floodZone === 'NON_FLOOD_ZONE') {
      result.environment.push('No flood risk zone');
    } else if (formData.floodZone === 'FLOOD_ZONE') {
      result.environment.push('Located in flood risk zone');
    }

    if (formData.livingSize <= 120) {
      result.environment.push('Energy-efficient surface area');
    }

    // Insights sociaux
    const urbanCities = ['Antwerpen', 'Brussels', 'Gent', 'Brugge', 'Leuven', 'Namur', 'Mons', 'Charleroi', 'Liège'];
    if (urbanCities.includes(formData.locality)) {
      result.social.push(`Prime urban location: ${formData.locality}`);
    } else {
      result.social.push(`Location: ${formData.locality || 'Not specified'}`);
    }

    if (formData.roomsCount >= 3) {
      result.social.push(`Family-friendly: ${formData.roomsCount} bedrooms`);
    } else if (formData.roomsCount >= 1) {
      result.social.push(`${formData.roomsCount} bedroom${formData.roomsCount > 1 ? 's' : ''}`);
    }

    const amenities = [];
    if (formData.hasLivingRoom) amenities.push('living room');
    if (formData.hasTerrace) amenities.push('terrace');
    if (amenities.length > 0) {
      result.social.push(`Quality amenities: ${amenities.join(' + ')}`);
    }

    if (formData.bathroomsCount >= 2) {
      result.social.push(`Multiple bathrooms (${formData.bathroomsCount})`);
    }

    // Insights gouvernance
    const currentYear = new Date().getFullYear();
    const age = currentYear - (formData.constructionYear || 1990);
    if (age <= 25) {
      result.governance.push(`Built in ${formData.constructionYear || 'N/A'} - meets modern standards`);
    } else {
      result.governance.push(`Building from ${formData.constructionYear || 'N/A'} (${age} years old)`);
    }

    const conditionText = {
      'AS_NEW': 'Excellent condition',
      'GOOD': 'Good condition',
      'TO_RENOVATE': 'Needs renovation',
      'TO_RESTORE': 'Major restoration needed'
    };
    result.governance.push(conditionText[formData.buildingCondition] || 'Condition status available');

    result.governance.push('Transparent property data available');

    return result;
  }, [formData]);

  // Fonction pour convertir le score en lettre
  const getScoreLetter = (score) => {
    if (score >= 90) return 'A+';
    if (score >= 80) return 'A';
    if (score >= 70) return 'B+';
    if (score >= 60) return 'B';
    if (score >= 50) return 'C+';
    if (score >= 40) return 'C';
    return 'D';
  };

  if (!esgCalculation) {
    return (
      <div className="esg-summary">
        <div className="esg-summary-header">
          <h3>ESG Quick Assessment</h3>
          <div className="esg-loading">Fill the form to see ESG analysis</div>
        </div>
      </div>
    );
  }

  const scoreLetter = getScoreLetter(esgCalculation.overall);

  // Function to generate compliance explanations
  const getComplianceExplanation = (complianceType, status, propertyData) => {
    const explanations = {
      energy_compliance: {
        compliant: `This property meets energy compliance standards due to its ${propertyData.epcScore || 'good'} energy rating. ${propertyData.buildingConstructionYear > 2010 ? 'Recent construction ensures compliance with modern energy efficiency standards.' : 'Despite its age, energy systems have been updated to meet current requirements.'} All European energy regulations are respected and the property demonstrates excellent energy performance.`,
        'non-compliant': `Non-compliance identified due to insufficient energy rating. Energy renovation work is required to meet current standards.`,
        'partial-compliance': `Partial compliance - some energy aspects meet standards but improvements are recommended for full compliance.`
      },
      building_codes: {
        compliant: `The building complies with all applicable construction codes and regulations. ${propertyData.buildingConstructionYear > 2010 ? 'Recent construction guarantees compliance with modern building standards.' : 'Despite its age, it has been brought up to current standards.'} All mandatory inspections are up to date and no violations have been identified. The structure meets safety, accessibility, and construction quality requirements.`,
        'non-compliant': `Building code violations have been identified requiring immediate corrections to meet regulatory standards.`,
        'partial-compliance': `Most building codes are met but some minor adjustments are necessary for full compliance.`
      },
      safety_standards: {
        compliant: `All safety standards are fully met and exceeded. ${propertyData.hasSecuritySystem ? 'Installed security systems enhance protection beyond basic requirements.' : 'Basic safety equipment meets all compliance standards.'} ${propertyData.floodZoneType === 'NON_FLOOD_ZONE' ? 'Location in a non-flood zone enhances overall safety profile.' : ''} Fire detectors, emergency exits, and fire safety equipment are compliant. The property demonstrates excellent safety preparedness and risk management.`,
        'non-compliant': `Safety issues have been identified requiring immediate attention to meet regulatory standards.`,
        'partial-compliance': `Most safety standards are met but some improvements are recommended for enhanced protection.`
      }
    };

    const typeExplanations = explanations[complianceType] || {};
    return typeExplanations[status.toLowerCase().replace(/\s+/g, '-')] || `Status ${status} for ${complianceType.replace(/_/g, ' ')}.`;
  };

  return (
    <div className="esg-summary">
      <div className="esg-summary-header">
        <h3>ESG Quick Assessment</h3>
        <div className="esg-overall-score">
          <span className="score-label">Overall ESG Score:</span>
          <span className={`score-text score-${scoreLetter.toLowerCase().replace('+', 'plus')}`}>
            {scoreLetter} ({esgCalculation.overall}/100)
          </span>
        </div>
      </div>

      <div className="esg-categories">
        <div className="esg-category environment">
          <div className="category-header">
            <h4>Environment ({esgCalculation.environment}/100)</h4>
          </div>
          <div className="category-insights">
            {insights.environment.map((insight, index) => (
              <div key={index} className="insight-item">• {insight}</div>
            ))}
          </div>
        </div>

        <div className="esg-category social">
          <div className="category-header">
            <h4>Social ({esgCalculation.social}/100)</h4>
          </div>
          <div className="category-insights">
            {insights.social.map((insight, index) => (
              <div key={index} className="insight-item">• {insight}</div>
            ))}
          </div>
        </div>

        <div className="esg-category governance">
          <div className="category-header">
            <h4>Governance ({esgCalculation.governance}/100)</h4>
          </div>
          <div className="category-insights">
            {insights.governance.map((insight, index) => (
              <div key={index} className="insight-item">• {insight}</div>
            ))}
          </div>
        </div>
      </div>

      {/* Financial Impact Section - shown after detailed analysis */}
      {esgAnalysisAvailable && detailedEsgData?.financial_impact && (
        <div className="esg-additional-section">
          <div className="section-header">
            <h4>Financial Impact</h4>
          </div>
          <div className="section-content">
            {Object.entries(detailedEsgData.financial_impact).map(([key, value], index) => (
              <div key={index} className="impact-item">
                <strong>{key.replace(/_/g, ' ').toUpperCase()}:</strong>
                <span className="impact-value">{value}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Compliance Status Section - shown after detailed analysis */}
      {esgAnalysisAvailable && detailedEsgData?.compliance_status && (
        <div className="esg-additional-section">
          <div className="section-header">
            <h4>Compliance Status</h4>
          </div>
          <div className="section-content">
            {Object.entries(detailedEsgData.compliance_status).map(([key, value], index) => (
              <div key={index} className="compliance-item">
                <div className="compliance-header">
                  <strong>{key.replace(/_/g, ' ').toUpperCase()}:</strong>
                  <span className={`compliance-status status-${value.toLowerCase().replace(/\s+/g, '-')}`}>
                    {value}
                  </span>
                </div>
                <div className="compliance-explanation">
                  {getComplianceExplanation(key, value, formData)}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Key Recommendations Section - shown after detailed analysis */}
      {esgAnalysisAvailable && detailedEsgData?.recommendations && detailedEsgData.recommendations.length > 0 && (
        <div className="esg-additional-section">
          <div className="section-header">
            <h4>Key Recommendations</h4>
          </div>
          <div className="section-content">
            {detailedEsgData.recommendations.slice(0, 5).map((recommendation, index) => (
              <div key={index} className="recommendation-item">
                <span className="recommendation-number">{index + 1}.</span>
                <span className="recommendation-text">{recommendation}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
export default EsgSummary;
