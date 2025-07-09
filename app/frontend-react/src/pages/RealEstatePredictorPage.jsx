// src/pages/RealEstatePredictorPage.jsx
import React from "react";
import PropertyForm from "../components/PropertyForm";

const RealEstatePredictorPage = () => {
  return (
    <div className="min-h-screen bg-gray-50 py-6">
      <h1 className="text-3xl font-bold text-center mb-6">Real Estate Price Predictor</h1>
      <PropertyForm />
    </div>
  );
};

export default RealEstatePredictorPage;
