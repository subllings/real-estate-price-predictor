import React from "react";
import AgentCard from "../components/AgentCard/AgentCard.jsx";

const agents = [
  {
    id: "predict",
    name: "Real Estate Price Predictor",
    description: "AI-powered property valuation with Belgian market data",
    image: "/images/realestate256x256.png",
    path: "/"
  },
  {
    id: "esg",
    name: "ESG Sustainability Agent",
    description: "Analyze ESG compliance, PEB ratings, and regulatory impact",
    image: "/images/esg256x256.png",
    path: "/esg-agent"
  },
  {
    id: "training",
    name: "Model Training Agent",
    description: "Azure ML training, hyperparameter optimization, quality gates",
    image: "/images/training256x256.png",
    path: "/training"
  },
  {
    id: "finance",
    name: "Real Estate Finance Agent",
    description: "Investment analysis, ROI calculations, market trends",
    image: "/images/investment256x256.png",
    path: "/agent/finance"
  }
];

export default function HomePage() {
  return (
    <div className="px-6 py-10">
      <h1 className="text-3xl font-bold mb-4 text-center">Real Estate AI Platform</h1>
      <p className="text-gray-500 text-lg mt-2 mb-8 text-center max-w-2xl mx-auto">
        Comprehensive AI-powered real estate analysis platform with price prediction, 
        ESG compliance, and advanced training capabilities.
      </p>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-2 gap-6 mt-10 max-w-4xl mx-auto">
        {agents.map((agent) => (
          <AgentCard
            key={agent.id}
            title={agent.name}
            imageSrc={agent.image}
            description={agent.description}
            path={agent.path}
          />
        ))}
      </div>
      
      {/* Platform Features */}
      <div className="mt-12 max-w-4xl mx-auto">
        <h2 className="text-xl font-semibold text-center mb-6 text-gray-700">Platform Capabilities</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="bg-blue-50 rounded-lg p-4 text-center">
            <h3 className="font-medium text-blue-800">Accurate Predictions</h3>
            <p className="text-sm text-blue-600">R² ≥ 0.85 quality gates</p>
          </div>
          <div className="bg-green-50 rounded-lg p-4 text-center">
            <h3 className="font-medium text-green-800">ESG Compliance</h3>
            <p className="text-sm text-green-600">Belgian PEB regulations</p>
          </div>
          <div className="bg-purple-50 rounded-lg p-4 text-center">
            <h3 className="font-medium text-purple-800">Cloud Training</h3>
            <p className="text-sm text-purple-600">Azure ML integration</p>
          </div>
        </div>
      </div>
    </div>
  );
}
