// src/pages/RealEstatePredictorPage.jsx
import React, { useState, useEffect } from "react";
import PropertyForm from "../components/PropertyForm";
import SidePanel from "../components/SidePanel/SidePanel";
import BelgianESGAgent from "../components/BelgianESGAgent";

const RealEstatePredictorPage = () => {
  const [isChatOpen, setIsChatOpen] = useState(true);
  const [isESGPanelOpen, setIsESGPanelOpen] = useState(true);
  const [propertyData, setPropertyData] = useState(null);
  const [estimatedPrice, setEstimatedPrice] = useState(null);
  const [esgAnalysis, setESGAnalysis] = useState(null);
  const [chatComments, setChatComments] = useState([]);

  const handlePropertyDataChange = (data) => {
    console.log("Property data updated:", data);
    setPropertyData(data);
  };

  const handlePriceEstimate = (price) => {
    console.log("Price estimated:", price);
    setEstimatedPrice(price);
  };

  const handleESGAnalysis = (analysis) => {
    setESGAnalysis(analysis);
  };

  return (
    <div className="min-h-screen bg-gray-50 relative overflow-hidden">
      {/* Navigation Bar */}
      <div className="bg-blue-600 text-white px-6 py-3 flex justify-between items-center relative z-50">
        <div className="flex items-center space-x-4">
          <span className="text-xl font-bold">🏠 RealEstate AI</span>
          <nav className="flex space-x-6 text-sm">
            <a href="#" className="hover:text-blue-200">Home</a>
            <a href="#" className="text-blue-200 font-medium">Price Predictor</a>
            <a href="#" className="hover:text-blue-200">ESG Agent</a>
            <a href="#" className="hover:text-blue-200">Model Training</a>
            <a href="#" className="hover:text-blue-200">Admin Panel</a>
          </nav>
        </div>
        <div className="flex items-center space-x-4">
          <span className="text-sm">🌟 RE Agents ▼</span>
          <div className="flex space-x-2">
            <button className="p-1 hover:bg-blue-700 rounded">⚙️</button>
            <button className="p-1 hover:bg-blue-700 rounded">👤</button>
          </div>
        </div>
      </div>

      {/* Chat Assistant Panel - Left Side */}
      <div className={`fixed left-0 top-16 bottom-0 w-80 bg-white shadow-lg z-40 border-r transition-transform duration-300 ${
        isChatOpen ? 'translate-x-0' : '-translate-x-full'
      }`}>
        <div className="bg-green-500 text-white p-3 flex justify-between items-center">
          <div>
            <div className="font-medium">Profile: Yves</div>
            <div className="text-xs text-green-100">AI Chat Assistant</div>
          </div>
          <button
            onClick={() => setIsChatOpen(false)}
            className="text-white hover:text-green-200"
          >
            ✕
          </button>
        </div>
        <div className="p-4 h-full overflow-hidden">
          <div className="bg-gray-50 p-3 mb-4 rounded text-sm">
            <strong>Commentary</strong><br/>
            No comment available.
          </div>
          <div className="mb-4">
            <div className="text-sm font-medium mb-2">Hello! How can I assist you today?</div>
          </div>
          <div className="absolute bottom-4 left-4 right-4">
            <div className="flex">
              <input
                type="text"
                placeholder="Ask your question..."
                className="flex-1 p-2 border rounded-l text-sm"
              />
              <button className="bg-blue-500 text-white px-4 py-2 rounded-r text-sm hover:bg-blue-600">
                Send
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ESG Analysis Panel - Right Side */}
      <div className={`fixed right-0 top-16 bottom-0 w-80 bg-white shadow-lg z-40 border-l transition-transform duration-300 ${
        isESGPanelOpen ? 'translate-x-0' : 'translate-x-full'
      }`}>
        <div className="bg-blue-500 text-white p-3 flex justify-between items-center">
          <div>
            <div className="font-medium">ESG Analysis Report</div>
            <div className="text-xs text-blue-100">HOUSE in Antwerpen, Antwerp</div>
          </div>
          <button
            onClick={() => setIsESGPanelOpen(false)}
            className="text-white hover:text-blue-200"
          >
            ✕
          </button>
        </div>
        <div className="p-4 h-full overflow-y-auto">
          <div className="bg-blue-500 text-white p-3 rounded mb-4">
            <div className="font-medium">🔄 Detailed Analysis</div>
            <div className="text-xs">17 insights generated</div>
          </div>
          
          <div className="text-sm mb-4">
            <strong>Certainly!</strong> Here's a comprehensive ESG analysis tailored to your Antwerp house:
          </div>

          <div className="space-y-4 text-sm">
            <div>
              <div className="font-medium text-blue-600 mb-2">🔷 EPC Rating Analysis</div>
              <div className="text-gray-700">
                With an EPC score of A+, this property is among the most energy-
                efficient homes in Belgium. This rating signals excellent insulation,
                modern windows, and efficient systems, placing it well above the
                legal threshold for both ownership and rental.
              </div>
            </div>

            <div>
              <div className="font-medium text-blue-600 mb-2">⚡ Energy Consumption Estimates</div>
              <div className="text-gray-700">
                For a 110 m² house with an A+ label, annual primary energy
                consumption is likely below 45 kWh/m²/year, translating to roughly
                11,000 kWh/year or less. This is significantly lower than the Belgian
                average, resulting in reduced utility bills and a smaller carbon
                footprint.
              </div>
            </div>

            <div>
              <div className="font-medium text-blue-600 mb-2">🔥 Heating System Efficiency</div>
              <div className="text-gray-700">
                The electric heating system's efficiency depends on the technology
                used—if it's a heat pump, efficiency is high; if it's traditional electric
                radiators, running costs and emissions could be higher. Confirming
                the type (ideally a heat pump) is key to maintaining low operating
                costs.
              </div>
            </div>

            <div>
              <div className="font-medium text-blue-600 mb-2">🏠 Belgian Energy Performance Requirements</div>
              <div className="text-gray-700">
                Flanders mandates that all residential properties achieve at least
                EPC label D by 2030, with stricter requirements possible. Your A+
                rating already surpasses these thresholds, ensuring full compliance.
              </div>
            </div>

            <div>
              <div className="font-medium text-blue-600 mb-2">🚫 Rental Restrictions for Low-Performing Properties</div>
              <div className="text-gray-700">
                From 2023, properties with EPC label E or F cannot be newly rented
                out. This house's A+ status means it faces no rental restrictions and
                offers great income potential if leased.
              </div>
            </div>

            <div className="bg-yellow-50 p-3 rounded">
              <div className="font-medium text-yellow-800">⚠️ Disclaimer:</div>
              <div className="text-yellow-700 text-xs">
                This analysis is based on AI-generated insights and
                property data. For official energy performance certificates and
                renovation advice, consult certified professionals.
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className={`transition-all duration-300 ${
        isChatOpen && isESGPanelOpen ? 'mx-80' : 
        isChatOpen ? 'ml-80' : 
        isESGPanelOpen ? 'mr-80' : ''
      }`}>
        <div className="py-6">
          <div className="max-w-4xl mx-auto px-4">
            <h1 className="text-3xl font-bold text-center mb-2">Real Estate Price Predictor</h1>
            <p className="text-center text-gray-600 mb-8">
              AI-powered property valuation using Belgian market data and machine learning algorithms
            </p>
            
            <PropertyForm 
              onDataChange={handlePropertyDataChange}
              onPriceEstimate={handlePriceEstimate}
            />
          </div>
        </div>
      </div>

      {/* Fixed Admin Panel Toggle */}
      <div className="fixed bottom-4 left-1/2 transform -translate-x-1/2 z-50">
        <div className="bg-gray-800 text-white px-4 py-2 rounded text-sm">
          Ctrl+A for Admin Panel
        </div>
      </div>

      {/* Toggle Buttons */}
      <div className="fixed bottom-4 left-4 flex flex-col gap-2 z-50">
        {!isChatOpen && (
          <button
            onClick={() => setIsChatOpen(true)}
            className="bg-green-500 text-white px-3 py-2 rounded-lg shadow hover:bg-green-600 text-sm"
          >
            💬 AI Chat Assistant
          </button>
        )}
      </div>

      <div className="fixed bottom-4 right-4 flex flex-col gap-2 z-50">
        {!isESGPanelOpen && (
          <button
            onClick={() => setIsESGPanelOpen(true)}
            className="bg-blue-500 text-white px-3 py-2 rounded-lg shadow hover:bg-blue-600 text-sm"
          >
            📊 ESG Analysis
          </button>
        )}
      </div>
    </div>
  );
};

export default RealEstatePredictorPage;
