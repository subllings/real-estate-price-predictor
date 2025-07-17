import React, { useState } from "react";
import PropertyForm from "../components/PropertyForm/PropertyForm.js";
import ESGPanel from "../components/ESGPanel/ESGPanel.jsx";
import SidePanel from "../components/SidePanel/SidePanel.jsx";

const RealEstatePredictorPage = () => {
  const [showAdmin, setShowAdmin] = useState(false);
  const [esgPanelOpen, setEsgPanelOpen] = useState(false);
  const [esgAnalysis, setEsgAnalysis] = useState([]);
  const [propertyData, setPropertyData] = useState(null);
  const [predictionData, setPredictionData] = useState(null);
  const [esgData, setEsgData] = useState(null);
  const [esgLoading, setEsgLoading] = useState(false);
  
  // SidePanel states
  const [sidePanelExpanded, setSidePanelExpanded] = useState(false);
  const [comments, setComments] = useState([]);
  
  // Reference to SidePanel's sendMessageToChat function
  const [sendMessageToChat, setSendMessageToChat] = useState(null);

  const handleAdminToggle = () => {
    setShowAdmin(!showAdmin);
  };

  const handleEsgPanelToggle = () => {
    setEsgPanelOpen(!esgPanelOpen);
  };

  const handleEsgPanelOpen = () => {
    setEsgPanelOpen(true);
  };

  const handleEsgPanelClose = () => {
    setEsgPanelOpen(false);
  };

  const handleSidePanelToggle = () => {
    setSidePanelExpanded(!sidePanelExpanded);
  };

  const handleSidePanelOpen = () => {
    setSidePanelExpanded(true);
  };

  const handleSidePanelClose = () => {
    setSidePanelExpanded(false);
  };

  const clearComments = () => {
    setComments([]);
  };

  const handleSendChatMessage = (message) => {
    if (sendMessageToChat) {
      sendMessageToChat(message);
    }
  };

  const handleSetSendMessageToChat = (sendMessageFunction) => {
    setSendMessageToChat(() => sendMessageFunction);
  };

  const handlePredictionComment = (newComments) => {
    // Filtrer les doublons avant d'ajouter
    setComments(prev => {
      const existingTexts = prev.map(comment => comment.trim());
      const uniqueNewComments = newComments.filter(comment => 
        comment.trim() !== '' && !existingTexts.includes(comment.trim())
      );
      
      if (uniqueNewComments.length > 0) {
        return [...prev, ...uniqueNewComments];
      }
      return prev;
    });
  };

  const user = {
    profile: "Yves"
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Side Panel (Left) */}
      <SidePanel 
        user={user}
        isExpanded={sidePanelExpanded}
        onToggle={handleSidePanelToggle}
        onClose={handleSidePanelClose}
        comments={comments}
        clearComments={clearComments}
        propertyData={propertyData}
        predictionData={predictionData}
        esgData={esgData}
        onSendChatMessage={handleSetSendMessageToChat}
      />
      
      {/* ESG Panel (Right) */}
      <ESGPanel 
        isOpen={esgPanelOpen}
        onClose={handleEsgPanelClose}
        onToggle={handleEsgPanelToggle}
        esgAnalysis={esgAnalysis}
        propertyData={propertyData}
        esgLoading={esgLoading}
      />
      
      <div className="pt-6 pb-6">
        <div className="max-w-4xl mx-auto px-4">
          
          <PropertyForm 
            onPredictionComment={handlePredictionComment}
            onToggleSidePanel={handleSidePanelToggle}
            onOpenSidePanel={handleSidePanelOpen}
            onOpenEsgPanel={handleEsgPanelOpen}
            onSetEsgAnalysis={setEsgAnalysis}
            onSetPropertyData={setPropertyData}
            onSetPredictionData={setPredictionData}
            onSetEsgData={setEsgData}
            onSetEsgLoading={setEsgLoading}
            onClearComments={clearComments}
            onSendChatMessage={handleSendChatMessage}
          />
          
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
