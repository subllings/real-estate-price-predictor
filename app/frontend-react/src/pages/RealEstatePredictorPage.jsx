// src/pages/RealEstatePredictorPage.jsx
import React from "react";
import PropertyForm from "../components/PropertyForm";

const RealEstatePredictorPage = () => {
  return (
    <div className="min-h-screen bg-gray-50 py-6">
      <div className="max-w-6xl mx-auto px-4">
        <div className="text-center mb-6">
          <h1 className="text-3xl font-bold text-gray-800 mb-2">
            Real Estate Price Predictor
          </h1>
          <p className="text-gray-600">
            AI-powered property valuation using Belgian market data and machine learning algorithms
          </p>
        </div>
        
        <PropertyForm />
      </div>
    </div>
  );
};

export default RealEstatePredictorPage;
