import React, { useState } from "react";
import PropertyForm from "../components/PropertyForm/PropertyForm.js";
import GlobalMegaMenu from "../components/GlobalMegaMenu";

const RealEstatePredictorPage = () => {
  const [showAdmin, setShowAdmin] = useState(false);

  const handleAdminToggle = () => {
    setShowAdmin(!showAdmin);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <GlobalMegaMenu onAdminToggle={handleAdminToggle} />
      
      <div className="pt-6 pb-6">
        <div className="max-w-4xl mx-auto px-4">
          <div className="text-center mb-8">
            <h1 className="text-4xl font-bold text-gray-800 mb-2">
              Real Estate Price Predictor
            </h1>
            <p className="text-gray-600">
              AI-powered property valuation with ESG analysis
            </p>
          </div>
          
          <PropertyForm />
          
          {showAdmin && (
            <div className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center">
              <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
                <div className="flex justify-between items-center mb-4">
                  <h3 className="text-lg font-semibold">Admin Panel</h3>
                  <button
                    onClick={() => setShowAdmin(false)}
                    className="text-gray-500 hover:text-gray-700"
                  >
                    ✕
                  </button>
                </div>
                <div className="space-y-3">
                  <button className="w-full p-3 bg-blue-50 hover:bg-blue-100 rounded-lg text-left">
                    📊 System Monitoring
                  </button>
                  <button className="w-full p-3 bg-green-50 hover:bg-green-100 rounded-lg text-left">
                    🔧 Model Training
                  </button>
                  <button className="w-full p-3 bg-purple-50 hover:bg-purple-100 rounded-lg text-left">
                    📈 Analytics Dashboard
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default RealEstatePredictorPage;
