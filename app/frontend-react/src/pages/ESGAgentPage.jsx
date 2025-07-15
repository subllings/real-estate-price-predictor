import React from "react";
import BelgianESGAgent from "../components/BelgianESGAgent";

export default function ESGAgentPage() {
  return (
    <div className="min-h-screen bg-gray-50 py-6">
      <div className="max-w-6xl mx-auto px-4">
        <div className="text-center mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            🌱 Real Estate ESG Advisor
          </h1>
          <p className="text-gray-600">
            Artificial intelligence specialized in sustainability and Belgian real estate regulations
          </p>
        </div>
        
        <BelgianESGAgent />
        
        {/* Additional Info */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-white rounded-lg p-4 shadow">
            <h3 className="font-semibold text-green-600 mb-2">📊 EPC Performance</h3>
            <p className="text-sm text-gray-600">
              Analyze energy class impact on value and 2030 compliance
            </p>
          </div>
          <div className="bg-white rounded-lg p-4 shadow">
            <h3 className="font-semibold text-blue-600 mb-2">💰 Grants & ROI</h3>
            <p className="text-sm text-gray-600">
              Belgian subsidy optimization and renovation profitability calculations
            </p>
          </div>
          <div className="bg-white rounded-lg p-4 shadow">
            <h3 className="font-semibold text-purple-600 mb-2">⚖️ Compliance</h3>
            <p className="text-sm text-gray-600">
              Regulatory anticipation strategies and penalty avoidance
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
