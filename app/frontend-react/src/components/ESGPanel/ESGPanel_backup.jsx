import React from 'react';
import './ESGPanel.css';

const ESGPanel = ({ isOpen, onClose, onToggle, esgAnalysis, propertyData,                 ) : (
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
                ) : (=> {
  const isLoadingState = esgAnalysis && esgAnalysis.length > 0 && 
    esgAnalysis[0].includes('Generating ESG analysis in progress');

  const formatAnalysisPoint = (point, index) => {
    // Check if this is a loading message - remove all emoji checks
    const isLoadingMessage = point.includes('Generating') || point.includes('analysis in progress') || 
      point.includes('Azure OpenAI') || point.includes('Processing');
    
    // Detect if the point starts with markdown-style formatting
    if (point.includes('**') && point.includes(':**')) {
      const parts = point.split(':**');
      if (parts.length >= 2) {
        const title = parts[0].replace(/\*\*/g, '').trim();
        const content = parts[1].trim();
        return (
          <div key={index} className="analysis-point">
            <h4 className="analysis-title">{title}</h4>
            <div className={`analysis-content ${isLoadingMessage ? 'loading-message' : ''}`}>
              {formatContentWithBullets(content)}
            </div>
          </div>
        );
      }
    }

    // For regular points with special loading message styling
    return (
      <div key={index} className="analysis-point">
        <div className={`analysis-content ${isLoadingMessage ? 'loading-message' : ''}`}>
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
            <h3>{isLoadingState ? 'Generating ESG Analysis...' : 'ESG Analysis Report'}</h3>
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
                      <h4>Analyse ESG en cours...</h4>
                      <p>Notre IA analyse les aspects environnementaux, sociaux et de gouvernance de votre propriété.</p>
                      <div className="loading-steps">
                        <div className="step">• Évaluation énergétique</div>
                        <div className="step">• Conformité réglementaire</div>
                        <div className="step">• Recommandations d'amélioration</div>
                      </div>
                    </div>
                  </div>
                ) : (
                  esgAnalysis.map((point, index) => formatAnalysisPoint(point, index))
                )}
              </div>

              {!isLoadingState && (
                <div className="analysis-footer">
                  <div className="disclaimer">
                    <p><strong>Disclaimer:</strong> This analysis is based on AI and property data.
                    For official certificates and renovation advice, consult certified professionals.</p>
                  </div>
                </div>
              )}
            </div>
          ) : esgLoading ? (
            <div className="esg-loading-state">
              <div className="esg-loading-spinner">
                <div className="spinner"></div>
              </div>
              <h4>ESG Analysis en cours</h4>
              <p>L'analyse ESG est en cours de traitement par notre IA. Veuillez patienter...</p>
              <div className="loading-progress">
                <div className="progress-dots">
                  <span className="dot"></span>
                  <span className="dot"></span>
                  <span className="dot"></span>
                </div>
              </div>
            </div>
          ) : (
            <div className="esg-empty-state">
              <span className="empty-text">ESG</span>
              <h4>No ESG Analysis Available</h4>
              <p>Click "Analyze Price & ESG" to generate your personalized ESG report.</p>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default ESGPanel;
