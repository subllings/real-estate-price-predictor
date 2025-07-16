import React from 'react';
import './EsgSummary.css';

const EsgSummary = ({ formData, onViewDetailedReport }) => {
  
  // Fonction pour calculer le score ESG basé sur les features
  const calculateESGScore = (data) => {
    let totalScore = 0;
    let maxScore = 0;

    // ENVIRONNEMENTAL (40% du score total)
    let envScore = 0;
    let envMaxScore = 40;

    // Classe énergétique (0-15 points)
    const epcScores = {
      'A_plus': 15, 'A': 12, 'B': 10, 'C': 8, 'D': 6, 'E': 4, 'F': 2, 'G': 0
    };
    envScore += epcScores[data.epcScore] || 0;

    // Type de chauffage (0-10 points)
    const heatingScores = {
      'ELECTRIC': 10, 'GAS': 6, 'FUEL_OIL': 2, 'WOOD': 8, 'SOLAR': 12
    };
    envScore += heatingScores[data.heatingType] || 5;

    // Zone inondable (0-10 points)
    if (data.floodZoneType === 'NON_FLOOD_ZONE') envScore += 10;
    else if (data.floodZoneType === 'POSSIBLE_FLOOD_ZONE') envScore += 5;

    // Surface raisonnable (0-5 points) - bonus pour efficacité énergétique
    if (data.habitableSurface <= 120) envScore += 5;
    else if (data.habitableSurface <= 180) envScore += 3;
    else if (data.habitableSurface <= 250) envScore += 1;

    totalScore += envScore;
    maxScore += envMaxScore;

    // SOCIAL (35% du score total)
    let socialScore = 0;
    let socialMaxScore = 35;

    // Localisation urbaine (0-15 points)
    const urbanCities = ['Antwerpen', 'Brussels', 'Gent', 'Brugge', 'Leuven'];
    if (urbanCities.includes(data.locality)) socialScore += 15;
    else socialScore += 8; // Autres localités

    // Capacité d'accueil familiale (0-10 points)
    if (data.bedroomCount >= 3) socialScore += 10;
    else if (data.bedroomCount >= 2) socialScore += 7;
    else socialScore += 4;

    // Commodités sociales (0-10 points)
    if (data.hasLivingRoom) socialScore += 3;
    if (data.hasTerrace) socialScore += 3;
    if (data.bathroomCount >= 2) socialScore += 2;
    if (data.toiletCount >= 2) socialScore += 2;

    totalScore += socialScore;
    maxScore += socialMaxScore;

    // GOUVERNANCE (25% du score total)
    let govScore = 0;
    let govMaxScore = 25;

    // Conformité aux normes modernes (0-15 points)
    const currentYear = new Date().getFullYear();
    const age = currentYear - data.buildingConstructionYear;
    if (age <= 5) govScore += 15;
    else if (age <= 15) govScore += 12;
    else if (age <= 25) govScore += 8;
    else if (age <= 40) govScore += 5;
    else govScore += 2;

    // Condition du bâtiment (0-10 points)
    const conditionScores = {
      'AS_NEW': 10, 'GOOD': 8, 'TO_RENOVATE': 4, 'TO_RESTORE': 2
    };
    govScore += conditionScores[data.buildingCondition] || 5;

    totalScore += govScore;
    maxScore += govMaxScore;

    // Calcul du score final sur 100
    const finalScore = Math.round((totalScore / maxScore) * 100);
    
    return {
      overall: finalScore,
      environment: Math.round((envScore / envMaxScore) * 100),
      social: Math.round((socialScore / socialMaxScore) * 100),
      governance: Math.round((govScore / govMaxScore) * 100),
      details: {
        env: { score: envScore, max: envMaxScore },
        social: { score: socialScore, max: socialMaxScore },
        gov: { score: govScore, max: govMaxScore }
      }
    };
  };

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

  // Générer les insights basés sur les features
  const generateInsights = (data) => {
    const insights = {
      environment: [],
      social: [],
      governance: []
    };

    // Insights environnementaux
    if (data.epcScore === 'A_plus' || data.epcScore === 'A') {
      insights.environment.push(`Excellent energy efficiency (Class ${data.epcScore.replace('_', '')})`);
    }
    if (data.heatingType === 'ELECTRIC') {
      insights.environment.push('Electric heating system');
    }
    if (data.floodZoneType === 'NON_FLOOD_ZONE') {
      insights.environment.push('No flood risk zone');
    }
    if (data.habitableSurface <= 120) {
      insights.environment.push('Energy-efficient surface area');
    }

    // Insights sociaux
    const urbanCities = ['Antwerpen', 'Brussels', 'Gent', 'Brugge', 'Leuven'];
    if (urbanCities.includes(data.locality)) {
      insights.social.push(`Prime urban location: ${data.locality}`);
    }
    if (data.bedroomCount >= 3) {
      insights.social.push(`Family-friendly: ${data.bedroomCount} bedrooms`);
    }
    if (data.hasLivingRoom && data.hasTerrace) {
      insights.social.push('Living room + terrace for quality of life');
    }

    // Insights gouvernance
    const currentYear = new Date().getFullYear();
    const age = currentYear - data.buildingConstructionYear;
    if (age <= 25) {
      insights.governance.push(`Built in ${data.buildingConstructionYear} - meets modern standards`);
    }
    if (data.buildingCondition === 'AS_NEW' || data.buildingCondition === 'GOOD') {
      insights.governance.push('Good building condition');
    }
    insights.governance.push('Transparent property data available');

    return insights;
  };

  const esgScore = calculateESGScore(formData);
  const insights = generateInsights(formData);
  const scoreLetter = getScoreLetter(esgScore.overall);

  return (
    <div className="esg-summary">
      <div className="esg-summary-header">
        <h3>ESG Summary</h3>
        <div className="esg-overall-score">
          <span className="score-label">Overall ESG Score:</span>
          <span className={`score-text score-${scoreLetter.toLowerCase().replace('+', 'plus')}`}>
            {scoreLetter} ({esgScore.overall}/100)
          </span>
        </div>
      </div>

      <div className="esg-categories">
        <div className="esg-category environment">
          <div className="category-header">
            <h4>Environment ({esgScore.environment}/100)</h4>
          </div>
          <div className="category-insights">
            {insights.environment.map((insight, index) => (
              <div key={index} className="insight-item">• {insight.replace('✔️ ', '')}</div>
            ))}
          </div>
        </div>

        <div className="esg-category social">
          <div className="category-header">
            <h4>Social ({esgScore.social}/100)</h4>
          </div>
          <div className="category-insights">
            {insights.social.map((insight, index) => (
              <div key={index} className="insight-item">• {insight.replace('✔️ ', '')}</div>
            ))}
          </div>
        </div>

        <div className="esg-category governance">
          <div className="category-header">
            <h4>Governance ({esgScore.governance}/100)</h4>
          </div>
          <div className="category-insights">
            {insights.governance.map((insight, index) => (
              <div key={index} className="insight-item">• {insight.replace('✔️ ', '')}</div>
            ))}
          </div>
        </div>
      </div>

      <div className="esg-summary-footer">
        <button 
          className="view-detailed-btn"
          onClick={onViewDetailedReport}
        >
          View Detailed ESG Report
        </button>
      </div>
    </div>
  );
};

export default EsgSummary;
