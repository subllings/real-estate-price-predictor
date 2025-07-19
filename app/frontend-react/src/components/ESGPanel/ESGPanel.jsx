import React from 'react';
import './ESGAnalysisReport.css';

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
    if (!backgroundColor) return '#000000';

    // Convert hex to RGB
    const hex = backgroundColor.replace('#', '');
    const r = parseInt(hex.substring(0, 2), 16);
    const g = parseInt(hex.substring(2, 4), 16);
    const b = parseInt(hex.substring(4, 6), 16);

    // Calculate relative luminance
    const luminance = 0.299 * r + 0.587 * g + 0.114 * b;

    // Threshold: use black text if background is light, otherwise white
    return luminance > 160 ? '#000000' : '#ffffff';
};

  const getEsgImprovementColor = (improvement) => {
    if (improvement >= 15) return '#28a745'; // High improvement - green
    if (improvement >= 10) return '#8fd19e'; // Moderate improvement - light green
    if (improvement >= 5) return '#ffc107'; // Low improvement - yellow
    return '#f8d7da'; // Very low - light red
  };

  // Function to determine ESG risk level based on content
  const getESGRiskLevel = (content) => {
    if (!content || typeof content !== 'string') return 'low'; // Default to low risk
    
    const lowerContent = content.toLowerCase();
    
    // High risk indicators - Red border
    if (lowerContent.includes('high risk') || 
        lowerContent.includes('non-compliant') || 
        lowerContent.includes('poor performance') ||
        lowerContent.includes('major investment') ||
        lowerContent.includes('significant cost') ||
        lowerContent.includes('urgent') ||
        lowerContent.includes('critical') ||
        lowerContent.includes('expensive') ||
        lowerContent.includes('costly') ||
        lowerContent.includes('epc f') ||
        lowerContent.includes('epc g') ||
        lowerContent.includes('failing') ||
        lowerContent.includes('violation')) {
      return 'high'; // Red
    }
    
    // Medium risk indicators - Orange border
    if (lowerContent.includes('medium risk') || 
        lowerContent.includes('moderate investment') ||
        lowerContent.includes('consider upgrade') ||
        lowerContent.includes('improvement needed') ||
        lowerContent.includes('attention required') ||
        lowerContent.includes('upgrade') ||
        lowerContent.includes('investment') ||
        lowerContent.includes('cost') ||
        lowerContent.includes('epc d') ||
        lowerContent.includes('epc e') ||
        lowerContent.includes('below average')) {
      return 'medium'; // Orange
    }
    
    // Low risk/monitoring - Yellow border
    if (lowerContent.includes('low risk') || 
        lowerContent.includes('minor investment') ||
        lowerContent.includes('optional upgrade') ||
        lowerContent.includes('recommendation') ||
        lowerContent.includes('monitor') ||
        lowerContent.includes('maintain') ||
        lowerContent.includes('consider') ||
        lowerContent.includes('potential') ||
        lowerContent.includes('epc c') ||
        lowerContent.includes('average performance')) {
      return 'low'; // Yellow
    }
    
    // Excellent/positive indicators - Green border
    if (lowerContent.includes('excellent') || 
        lowerContent.includes('outstanding') ||
        lowerContent.includes('highly efficient') ||
        lowerContent.includes('compliant') ||
        lowerContent.includes('best in class') ||
        lowerContent.includes('future-proof') ||
        lowerContent.includes('a+') ||
        lowerContent.includes('epc a') ||
        lowerContent.includes('superior') ||
        lowerContent.includes('among the best') ||
        lowerContent.includes('exceeds') ||
        lowerContent.includes('optimal') ||
        lowerContent.includes('energy efficient') ||
        lowerContent.includes('low consumption') ||
        lowerContent.includes('savings')) {
      return 'excellent'; // Green
    }
    
    return 'neutral'; // Default neutral (no special border)
  };

  // Function to get border color based on ESG risk level
  const getESGBorderColor = (riskLevel) => {
    switch (riskLevel) {
      case 'high': return '#dc3545'; // Red - High risk
      case 'medium': return '#ff9800'; // Orange - Medium risk  
      case 'low': return '#ffc107'; // Yellow - Low risk/monitoring
      case 'excellent': return '#28a745'; // Green - Excellent performance
      default: return 'transparent'; // No border for neutral
    }
  };

  // Function to format text with markdown bold markers
  const formatTextWithMarkdown = (text) => {
    if (!text || typeof text !== 'string') return text;
    
    // Split text by ** markers and format as bold
    const parts = text.split(/\*\*(.*?)\*\*/g);
    return parts.map((part, index) => {
      // Every odd index is the text between ** markers
      if (index % 2 === 1) {
        return <strong key={index} style={{ color: '#2c5aa0', fontWeight: 'bold' }}>{part}</strong>;
      }
      return part;
    });
  };

  // Action points summary generator - creates concise actionable recommendations
  const summarizeActionPoints = (analysisArr) => {
    if (!analysisArr || analysisArr.length === 0) return null;
    
    // Extract key actionable insights instead of duplicating full content
    const actionableInsights = [];
    
    analysisArr.forEach(pt => {
      if (!pt || typeof pt !== 'string') return;
      
      const lower = pt.toLowerCase();
      
      // Extract specific actionable recommendations with detailed descriptions
      if (lower.includes('investment') && lower.includes('recommend')) {
        actionableInsights.push("Monitor investment opportunities: Keep track of emerging energy efficiency technologies and government incentives that could further improve the property's performance and value.");
      }
      
      if (lower.includes('a+') && lower.includes('maintain')) {
        actionableInsights.push("Maintain A+ EPC rating: Schedule annual maintenance of heating, ventilation, and insulation systems to preserve the exceptional energy performance and avoid rating degradation.");
      }
      
      if (lower.includes('heat pump') || lower.includes('heating system')) {
        actionableInsights.push("Optimize heating system: Install smart thermostats and zoning controls to maximize efficiency and reduce operational costs while maintaining comfort levels.");
      }
      
      if (lower.includes('solar') || lower.includes('renewable')) {
        actionableInsights.push("Consider renewable energy integration: Assess the feasibility of adding solar panels or battery storage systems to further reduce energy costs and carbon footprint.");
      }
      
      if (lower.includes('smart') && lower.includes('meter')) {
        actionableInsights.push("Implement smart monitoring: Install advanced energy monitoring systems to track consumption patterns, identify optimization opportunities, and ensure regulatory compliance.");
      }
      
      if (lower.includes('compliance') && lower.includes('regulation')) {
        actionableInsights.push("Stay regulatory compliant: Monitor upcoming changes in Belgian energy regulations and ensure the property continues to meet or exceed all requirements for rentals and sales.");
      }
      
      if (lower.includes('rental') && lower.includes('premium')) {
        actionableInsights.push("Maximize rental value: Highlight the A+ energy rating in marketing materials to justify premium rents and attract environmentally conscious tenants willing to pay more for low utility costs.");
      }
    });
    
    // Remove duplicates and limit to most relevant actions
    const uniqueActions = [...new Set(actionableInsights)].slice(0, 6);
    
    if (uniqueActions.length === 0) return null;
    
    return (
      <div className="action-summary" style={{ marginTop: '1.5em', padding: '1em', background: '#f6f8fa', borderRadius: '0.7em', boxShadow: '0 2px 8px #eee' }}>
        <h4 style={{ marginBottom: '0.7em', color: '#4a9eff', fontSize: '1.3rem', fontWeight: 'bold' }}>Key Action Items</h4>
        <div style={{ display: 'grid', gap: '0.8em' }}>
          {uniqueActions.map((action, idx) => {
            const [title, description] = action.split(': ');
            return (
              <div key={idx} style={{ 
                padding: '1em', 
                backgroundColor: '#fff',
                borderRadius: '0.5em',
                border: '1px solid #e0e0e0',
                fontSize: '0.95rem',
                lineHeight: '1.5'
              }}>
                <div style={{ fontWeight: 'bold', color: '#2c5aa0', marginBottom: '0.4em' }}>
                  {title}
                </div>
                <div style={{ color: '#555' }}>
                  {description}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };
  const isLoadingState = esgAnalysis && esgAnalysis.length > 0 && 
    esgAnalysis[0].includes('Generating ESG analysis in progress');

  // Simplified formatting with ESG risk-based border colors
  const formatAnalysisPoint = (point, index) => {
    // Skip empty or whitespace-only points
    if (!point || point.trim() === '') {
      return null;
    }

    // Check if this is a loading message
    const isLoadingMessage = point.includes('Generating') || point.includes('analysis in progress') || 
      point.includes('Azure OpenAI') || point.includes('Processing');

    // Determine ESG risk level and corresponding CSS class
    const riskLevel = getESGRiskLevel(point);
    const riskClass = `esg-risk-${riskLevel}`;

    // Base style for all content
    let colorStyle = {
      background: 'transparent',
      color: '#333333',
      padding: '1em',
      borderRadius: '0.5em',
      marginBottom: '0.7em',
      display: 'block'
    };

    // Detect if the point starts with markdown-style formatting
    if (point.includes('**') && point.includes(':**')) {
      const parts = point.split(':**');
      if (parts.length >= 2) {
        const title = parts[0].replace(/\*\*/g, '').trim();
        const content = parts[1].trim();
        return (
          <div key={index} className="esg-analysis-report-point">
            <h4 className="esg-analysis-report-section-title">{title}</h4>
            <div className={`esg-analysis-report-section-content ${riskClass} ${isLoadingMessage ? 'loading-message' : ''}`} style={colorStyle}>
              {formatContentWithBullets(content)}
            </div>
          </div>
        );
      }
    }

    // For regular points with black text, white background and colored border
    return (
      <div key={index} className="esg-analysis-report-point">
        <div className={`esg-analysis-report-section-content ${riskClass} ${isLoadingMessage ? 'loading-message' : ''}`} style={colorStyle}>
          {formatContentWithBullets(point)}
          {isLoadingMessage && (
            <span className="esg-analysis-report-loading-dots">
              <span className="esg-analysis-report-loading-dot"></span>
              <span className="esg-analysis-report-loading-dot"></span>
              <span className="esg-analysis-report-loading-dot"></span>
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
      {/* Tab for reopening the panel */}
      <div
        className={`esg-analysis-report-tab ${isOpen ? 'hidden' : ''}`}
        onClick={onToggle}
        title="Open ESG Analysis"
      >
        <span className="esg-analysis-report-tab-text">ESG</span>
      </div>
      
      <div className={`esg-analysis-report-panel ${isOpen ? 'open' : ''}`}>
        <div className="esg-analysis-report-header">
          <div className="esg-analysis-report-title">
            <h3>ESG Analysis Report</h3>
            {propertyData && (
              <p className="esg-analysis-report-subtitle">
                {propertyData.propertyType} in {propertyData.locality}, {propertyData.province}
              </p>
            )}
          </div>
          <button
            onClick={onClose}
            className="esg-analysis-report-close"
            aria-label="Close ESG Panel"
          >
            ✕
          </button>
        </div>

        <div className="esg-analysis-report-content">
          {esgAnalysis && esgAnalysis.length > 0 ? (
            <div className="esg-analysis-report-container">
              <div className="esg-analysis-report-analysis-header">
                <div className="esg-analysis-report-badge">
                  <span className="esg-analysis-report-badge-text">
                    {isLoadingState ? 'Analysis in progress...' : 'Detailed Analysis'}
                  </span>
                </div>
                <div className="esg-analysis-report-meta">
                  {isLoadingState ? 
                    <div className="loading-meta">
                      <div className="esg-analysis-report-loading-spinner"></div>
                      <span>Generating insights...</span>
                    </div> : 
                    <span>{esgAnalysis.length} insights generated</span>
                  }
                </div>
              </div>

              <div className="esg-analysis-report-sections">
                {isLoadingState ? (
                  <div className="esg-analysis-report-loading-state">
                    <div className="esg-analysis-report-loading-spinner"></div>
                    <div className="esg-analysis-report-loading-text">
                      <h4>ESG Analysis in Progress...</h4>
                      <p>Our AI is analyzing the environmental, social, and governance aspects of your property.</p>
                      <div className="esg-analysis-report-loading-progress">
                        <div className="esg-analysis-report-progress-dots">
                          <div className="esg-analysis-report-progress-dot"></div>
                          <div className="esg-analysis-report-progress-dot"></div>
                          <div className="esg-analysis-report-progress-dot"></div>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <>
                    {esgAnalysis.map((point, index) => formatAnalysisPoint(point, index)).filter(Boolean)}
                    {summarizeActionPoints(esgAnalysis)}
                  </>
                )}
              </div>

              {!isLoadingState && (
                <div className="esg-analysis-report-footer">
                  <div className="esg-analysis-report-disclaimer">
                    <p><strong>Disclaimer:</strong> Generated by AI for informational purposes. Please consult with ESG experts for detailed assessments.</p>
                  </div>
                </div>
              )}
            </div>
          ) : esgLoading ? (
            <div className="esg-analysis-report-loading-state">
              <div className="esg-analysis-report-loading-spinner"></div>
              <div className="esg-analysis-report-loading-text">
                <h4>ESG Analysis in Progress...</h4>
                <p>Our AI is analyzing the environmental, social, and governance aspects of your property. Please wait...</p>
              </div>
            </div>
          ) : (
            <div className="esg-analysis-report-no-analysis">
              <div className="esg-empty-state">
                {/* ESG Logo/Icon */}
                <div className="esg-logo-container">
                  <div className="esg-logo">
                    <div className="esg-letter esg-e">E</div>
                    <div className="esg-letter esg-s">S</div>
                    <div className="esg-letter esg-g">G</div>
                  </div>
                  <div className="esg-logo-subtitle">Environmental • Social • Governance</div>
                </div>

                {/* Main message */}
                <div className="esg-empty-content">
                  <h3 className="esg-empty-title">Ready for ESG Analysis</h3>
                  <p className="esg-empty-description">
                    Get comprehensive insights into your property's environmental impact, 
                    social compliance, and governance standards with our AI-powered analysis.
                  </p>
                </div>

                {/* Call to action */}
                <div className="esg-cta-container">
                  <p className="esg-cta-text">
                    Click <strong>"Analyze Price & ESG"</strong> to generate your comprehensive report
                  </p>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default ESGPanel;
