/**
 * ESG Integration Component
 * Connects Price Predictor with ESG analysis
 */

import React from 'react';

const ESGIntegrationPrompt = ({ propertyData, estimatedPrice, onDetailedAnalysis }) => {
  const hasPropertyData = propertyData && Object.keys(propertyData).length > 0;
  
  // Calculate potential ESG impact
  const estimateESGImpact = () => {
    if (!propertyData) return null;
    
    const constructionYear = propertyData.constructionYear || 1980;
    const surface = propertyData.surface || 150;
    
    // Estimate energy class based on construction year
    let energyClass = 'G';
    let impactPercentage = -25;
    
    if (constructionYear >= 2015) {
      energyClass = 'B';
      impactPercentage = +10;
    } else if (constructionYear >= 2006) {
      energyClass = 'D';
      impactPercentage = -5;
    } else if (constructionYear >= 1990) {
      energyClass = 'F';
      impactPercentage = -15;
    }
    
    const adjustedPrice = estimatedPrice * (1 + impactPercentage / 100);
    const renovationCost = surface * (energyClass === 'G' ? 400 : energyClass === 'F' ? 300 : 200);
    
    return {
      energyClass,
      impactPercentage,
      adjustedPrice,
      renovationCost,
      potentialValue: adjustedPrice + (energyClass === 'G' || energyClass === 'F' ? 50000 : 20000)
    };
  };

  const esgImpact = estimateESGImpact();

  if (!hasPropertyData) {
    return (
      <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mt-4">
        <div className="flex items-start space-x-3">
          <div className="text-yellow-600 text-xl"></div>
          <div>
            <h4 className="font-medium text-yellow-800">Incomplete Estimation</h4>
            <p className="text-sm text-yellow-700 mt-1">
              This estimation does not take into account energy performance and ESG compliance.
              Complete the property form to get a detailed ESG analysis.
            </p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-r from-green-50 to-blue-50 border border-green-200 rounded-lg p-6 mt-6">
      <div className="flex items-start justify-between mb-4">
        <div>
          <h4 className="font-semibold text-green-800 text-lg">Integrated ESG Analysis</h4>
          <p className="text-sm text-green-700">Energy performance impact on your estimation</p>
        </div>
      </div>

      {esgImpact && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Current Impact */}
          <div className="bg-white rounded-lg p-4 border">
            <h5 className="font-medium text-gray-700 mb-3">Current Impact</h5>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Estimated energy class:</span>
                <span className={`font-bold ${
                  esgImpact.energyClass === 'B' ? 'text-green-600' :
                  esgImpact.energyClass === 'D' ? 'text-yellow-600' :
                  'text-red-600'
                }`}>
                  {esgImpact.energyClass}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Price impact:</span>
                <span className={`font-bold ${esgImpact.impactPercentage >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {esgImpact.impactPercentage > 0 ? '+' : ''}{esgImpact.impactPercentage}%
                </span>
              </div>
              <div className="flex justify-between">
                <span>ESG adjusted price:</span>
                <span className="font-bold text-blue-600">
                  {esgImpact.adjustedPrice.toLocaleString('fr-BE', { style: 'currency', currency: 'EUR', maximumFractionDigits: 0 })}
                </span>
              </div>
            </div>
          </div>

          {/* Renovation Potential */}
          <div className="bg-white rounded-lg p-4 border">
            <h5 className="font-medium text-gray-700 mb-3">Renovation Potential</h5>
            <div className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span>Estimated renovation cost:</span>
                <span className="font-bold text-orange-600">
                  {esgImpact.renovationCost.toLocaleString('fr-BE', { style: 'currency', currency: 'EUR', maximumFractionDigits: 0 })}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Post-renovation value:</span>
                <span className="font-bold text-green-600">
                  {esgImpact.potentialValue.toLocaleString('fr-BE', { style: 'currency', currency: 'EUR', maximumFractionDigits: 0 })}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Potential ROI:</span>
                <span className="font-bold text-purple-600">
                  +{((esgImpact.potentialValue - esgImpact.adjustedPrice - esgImpact.renovationCost) / esgImpact.renovationCost * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Call to Action */}
      <div className="mt-4 p-4 bg-blue-50 rounded-lg border border-blue-200">
        <div className="flex items-center justify-between">
          <div>
            <h6 className="font-medium text-blue-800">Complete ESG Analysis</h6>
            <p className="text-sm text-blue-700">
              Get a detailed assessment including EPC, grants, 2030 compliance
            </p>
          </div>
          <button 
            onClick={onDetailedAnalysis}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 transition-colors"
          >
            Detailed Analysis
          </button>
        </div>
      </div>

      {/* Quick ESG Check */}
      <div className="mt-4 grid grid-cols-3 gap-2 text-xs">
        <div className="bg-white p-2 rounded text-center border">
          <div className="font-medium text-red-600">2030 Risk</div>
          <div className="text-gray-600">F-G classes banned</div>
        </div>
        <div className="bg-white p-2 rounded text-center border">
          <div className="font-medium text-yellow-600">Grants Available</div>
          <div className="text-gray-600">15-30k€ by region</div>
        </div>
        <div className="bg-white p-2 rounded text-center border">
          <div className="font-medium text-green-600">Value Added</div>
          <div className="text-gray-600">+25% class A-B</div>
        </div>
      </div>
    </div>
  );
};

export default ESGIntegrationPrompt;
