import React from 'react';
import './ESGPanel.css';

const ESGPanel = ({ isOpen, onClose, onToggle, esgAnalysis, propertyData }) => {
  const formatAnalysisPoint = (point, index) => {
    // Detect if the point starts with markdown-style formatting
    if (point.includes('**') && point.includes(':**')) {
      const parts = point.split(':**');
      if (parts.length >= 2) {
        const title = parts[0].replace(/\*\*/g, '').trim();
        const content = parts[1].trim();
        return (
          <div key={index} className="analysis-point">
            <h4 className="analysis-title">{title}</h4>
            <p className="analysis-content">{content}</p>
          </div>
        );
      }
    }

    // For regular points without special formatting
    return (
      <div key={index} className="analysis-point">
        <p className="analysis-content">{point}</p>
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
        <div className="esg-panel-tab-icon">ESG</div>
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
                  <span className="badge-text">Detailed Analysis</span>
                </div>
                <div className="analysis-meta">
                  {esgAnalysis.length} insights generated
                </div>
              </div>

              <div className="analysis-sections">
                {esgAnalysis.map((point, index) => formatAnalysisPoint(point, index))}
              </div>

              <div className="analysis-footer">
                <div className="disclaimer">
                  <p><strong>Disclaimer:</strong> This analysis is based on AI-generated insights and property data.
                  For official energy performance certificates and renovation advice, consult certified professionals.</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="esg-empty-state">
              <div className="empty-icon">📊</div>
              <h4>No ESG Analysis Available</h4>
              <p>Click "Detailed Analysis" after making a price prediction to generate your personalized ESG report.</p>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default ESGPanel;
