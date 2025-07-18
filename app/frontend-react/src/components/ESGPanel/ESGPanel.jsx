import React from 'react';
import './ESGPanel.css';

const ESGPanel = ({ isOpen, onClose, onToggle, esgAnalysis, propertyData, esgLoading }) => {
  // Color logic helpers
  const getEpcColor = (epcScore) => {
    if (!epcScore) return '#6c757d';
    if (['A+', 'A'].includes(epcScore)) return '#28a745'; // Green
    if (['B', 'B+'].includes(epcScore)) return '#8fd19e'; // Light green
    if (['C', 'C+'].includes(epcScore)) return '#ffc107'; // Yellow
    if (['D', 'E'].includes(epcScore)) return '#ff9800'; // Orange
    if (['F', 'G'].includes(epcScore)) return '#dc3545'; // Red
    return '#dc3545'; // Default to red for unknown/very poor EPC
  };

  const getInvestmentColor = (amount) => {
    if (amount <= 10000) return '#28a745'; // Green
    if (amount <= 25000) return '#ffc107'; // Yellow
    return '#dc3545'; // Red
  };

  const getTextColorForBackground = (backgroundColor) => {
    // For dark green and dark red backgrounds, use black text
    if (backgroundColor === '#28a745' || backgroundColor === '#dc3545') {
      return '#000000'; // Black text
    }
    return '#ffffff'; // White text for other backgrounds
  };

  const getEsgImprovementColor = (improvement) => {
    if (improvement >= 15) return '#28a745'; // High improvement - green
    if (improvement >= 10) return '#8fd19e'; // Moderate improvement - light green
    if (improvement >= 5) return '#ffc107'; // Low improvement - yellow
    return '#f8d7da'; // Very low - light red
  };

  // Action points summary generator
  const summarizeActionPoints = (analysisArr) => {
    if (!analysisArr || analysisArr.length === 0) return null;
    // Group recommendations by category
    const categories = {
      Energy: [],
      Investment: [],
      Compliance: [],
      ESG: [],
      Other: []
    };
    analysisArr.forEach(pt => {
      const lower = pt.toLowerCase();
      if (lower.includes('epc') || lower.includes('energy')) categories.Energy.push(pt);
      else if (lower.includes('investment') || lower.includes('cost') || lower.includes('upgrade')) categories.Investment.push(pt);
      else if (lower.includes('compliance') || lower.includes('regulation')) categories.Compliance.push(pt);
      else if (lower.includes('esg') || lower.includes('improvement')) categories.ESG.push(pt);
      else categories.Other.push(pt);
    });
    // Only show categories with content
    const shownCats = Object.entries(categories).filter(([cat, arr]) => arr.length > 0);
    if (shownCats.length === 0) return null;
    return (
      <div className="action-summary" style={{ marginTop: '1.5em', padding: '1em', background: '#f6f8fa', borderRadius: '0.7em', boxShadow: '0 2px 8px #eee' }}>
        <h4 style={{ marginBottom: '0.7em', color: '#007bff' }}>Actionable Summary</h4>
        {shownCats.map(([cat, arr], i) => (
          <div key={cat} style={{ marginBottom: '1em' }}>
            <div style={{ fontWeight: 'bold', color: '#333', marginBottom: '0.3em' }}>{cat} Recommendations:</div>
            <ul style={{ margin: 0, paddingLeft: '1.2em' }}>
              {arr.map((act, idx) => <li key={idx} style={{ marginBottom: '0.5em' }}>{act}</li>)}
            </ul>
          </div>
        ))}
      </div>
    );
  };
  const isLoadingState = esgAnalysis && esgAnalysis.length > 0 && 
    esgAnalysis[0].includes('Generating ESG analysis in progress');

  const formatAnalysisPoint = (point, index) => {
    // Check if this is a loading message - remove all emoji checks
    const isLoadingMessage = point.includes('Generating') || point.includes('analysis in progress') || 
      point.includes('Azure OpenAI') || point.includes('Processing');

    // Color logic for EPC, investment, ESG improvement
    let colorStyle = {};
    if (point.match(/EPC\s([A-G][+]?)/)) {
      const epcMatch = point.match(/EPC\s([A-G][+]?)/);
      colorStyle.background = getEpcColor(epcMatch[1]);
      colorStyle.color = '#fff';
      colorStyle.padding = '0.3em 0.7em';
      colorStyle.borderRadius = '0.5em';
      colorStyle.display = 'inline-block';
      colorStyle.marginBottom = '0.5em';
    } else if (point.match(/investment.*?(\d{1,3}(?:,\d{3})*(?:\.\d+)?)/i)) {
      const investMatch = point.match(/(\d{1,3}(?:,\d{3})*(?:\.\d+)?)/);
      if (investMatch) {
        const investmentColor = getInvestmentColor(parseFloat(investMatch[1].replace(/,/g, '')));
        colorStyle.background = investmentColor;
        colorStyle.color = getTextColorForBackground(investmentColor);
        colorStyle.padding = '0.3em 0.7em';
        colorStyle.borderRadius = '0.5em';
        colorStyle.display = 'inline-block';
        colorStyle.marginBottom = '0.5em';
      }
    } else if (point.toLowerCase().includes('investment recommendations')) {
      colorStyle.background = '#dc3545';
      colorStyle.color = getTextColorForBackground('#dc3545');
      colorStyle.padding = '1em';
      colorStyle.borderRadius = '1em';
      colorStyle.marginBottom = '0.7em';
      colorStyle.display = 'block';
    } else if (point.match(/ESG improvements.*?(\d{1,3})%/i)) {
      const esgMatch = point.match(/(\d{1,3})%/);
      if (esgMatch) {
        const esgColor = getEsgImprovementColor(parseInt(esgMatch[1]));
        colorStyle.background = esgColor;
        colorStyle.color = getTextColorForBackground(esgColor);
        colorStyle.padding = '0.3em 0.7em';
        colorStyle.borderRadius = '0.5em';
        colorStyle.display = 'inline-block';
        colorStyle.marginBottom = '0.5em';
      }
    }

    // Detect if the point starts with markdown-style formatting
    if (point.includes('**') && point.includes(':**')) {
      const parts = point.split(':**');
      if (parts.length >= 2) {
        const title = parts[0].replace(/\*\*/g, '').trim();
        const content = parts[1].trim();
        return (
          <div key={index} className="analysis-point">
            <h4 className="analysis-title">{title}</h4>
            <div className={`analysis-content ${isLoadingMessage ? 'loading-message' : ''}`} style={colorStyle}>
              {formatContentWithBullets(content)}
            </div>
          </div>
        );
      }
    }

    // For regular points with special loading message styling
    return (
      <div key={index} className="analysis-point">
        <div className={`analysis-content ${isLoadingMessage ? 'loading-message' : ''}`} style={colorStyle}>
          {formatContentWithBullets(point)}
          {isLoadingMessage && (
            <span className="esg-loading-dots">
              <span className="esg-loading-dot"></span>
              <span className="esg-loading-dot"></span>
              <span className="esg-loading-dot"></span>
            </span>
          )}
        </div>
      </div>
    );
  };

  const formatContentWithBullets = (content) => {
    if (!content) return null;
    
    // Remove blockquote formatting to avoid blue vertical bars
    let cleanContent = content.toString();
    
    // Remove blockquote markers (>) and clean up
    cleanContent = cleanContent.replace(/^>\s*/gm, '');
    cleanContent = cleanContent.replace(/^\s*>\s*/gm, '');
    
    // SUPPRIMER TOUTES LES BARRES HORIZONTALES POTENTIELLES
    cleanContent = cleanContent.replace(/^-{3,}.*$/gm, '');
    cleanContent = cleanContent.replace(/^={3,}.*$/gm, '');
    cleanContent = cleanContent.replace(/^_{3,}.*$/gm, '');
    cleanContent = cleanContent.replace(/^\*{3,}.*$/gm, '');
    cleanContent = cleanContent.replace(/^\s*-{3,}\s*$/gm, '');
    cleanContent = cleanContent.replace(/^\s*={3,}\s*$/gm, '');
    cleanContent = cleanContent.replace(/^\s*_{3,}\s*$/gm, '');
    cleanContent = cleanContent.replace(/^\s*\*{3,}\s*$/gm, '');
    cleanContent = cleanContent.replace(/\n\s*\n/g, '\n');
    
    // Split content into sentences and paragraphs
    const sentences = cleanContent.split(/(?<=[.!?])\s+/)
      .map(sentence => sentence.trim())
      .filter(sentence => sentence.length > 0);
    
    if (sentences.length <= 2) {
      // For short content, display as paragraph without any borders
      return <p style={{borderLeft: 'none', paddingLeft: '0'}}>{cleanContent}</p>;
    }
    
    // For longer content, create bullet points
    return (
      <div className="bullet-content">
        {sentences.map((sentence, idx) => (
          <div key={idx} className="bullet-point">
            <span className="bullet">•</span>
            <span className="bullet-text">{sentence}</span>
          </div>
        ))}
      </div>
    );
  };

  return (
    <>
      {/* Onglet visible pour rouvrir le panel */}
      <div
        className={`esg-panel-tab ${isOpen ? 'hidden' : ''}`}
        onClick={onToggle}
        title="Open ESG Analysis"
      >
        <span className="esg-panel-tab-text">ESG</span>
      </div>
      
      <div className={`esg-panel ${isOpen ? 'open' : ''}`}>
        <div className="esg-panel-header">
          <div className="esg-panel-title">
            <h3>ESG Analysis Report</h3>
            {propertyData && (
              <p className="property-summary">
                {propertyData.propertyType} in {propertyData.locality}, {propertyData.province}
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="esg-panel-close"
            aria-label="Close ESG Panel"
          >
            ✕
          </button>
        </div>

        <div className="esg-panel-content">
          {esgAnalysis && esgAnalysis.length > 0 ? (
            <div className="esg-analysis-container">
              <div className="analysis-header">
                <div className="analysis-badge">
                  <span className="badge-text">
                    {isLoadingState ? 'Analysis in progress...' : 'Detailed Analysis'}
                  </span>
                </div>
                <div className="analysis-meta">
                  {isLoadingState ? 
                    <div className="loading-meta">
                      <div className="spinner"></div>
                      <span>Azure OpenAI LLM Agent active</span>
                    </div> : 
                    `6 insights generated`
                  }
                </div>
              </div>

              <div className="analysis-sections">
                {isLoadingState ? (
                  <div className="analysis-loading">
                    <div className="loading-spinner-large"></div>
                    <div className="loading-message">
                      <h4>ESG Analysis in Progress...</h4>
                      <p>Our AI is analyzing the environmental, social, and governance aspects of your property.</p>
                      <div className="loading-steps">
                        <div className="step">• Energy assessment</div>
                        <div className="step">• Regulatory compliance</div>
                        <div className="step">• Improvement recommendations</div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <>
                    {esgAnalysis.map((point, index) => formatAnalysisPoint(point, index))}
                    {summarizeActionPoints(esgAnalysis)}
                  </>
                )}
              </div>

              {!isLoadingState && (
                <div className="analysis-footer">
                  <div className="disclaimer">
                    <p><strong>Disclaimer:</strong> Generated by AI for informational purposes. Please consult with ESG experts for detailed assessments.</p>
                  </div>
                </div>
              )}
            </div>
          ) : esgLoading ? (
            <div className="esg-loading-container">
              <div className="esg-loading-spinner"></div>
              <div className="esg-loading-text">
                <h4>ESG Analysis in Progress...</h4>
                <p>Our AI is analyzing the environmental, social, and governance aspects of your property. Please wait...</p>
              </div>
            </div>
          ) : (
            <div className="no-analysis">
              <p>No ESG analysis available. Click "Analyze Price & ESG" to generate a comprehensive report.</p>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default ESGPanel;
