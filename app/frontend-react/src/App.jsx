import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import RealEstatePredictorPage from './pages/RealEstatePredictorPage';
import ESGAgentPage from './pages/ESGAgentPage';
import ModelTrainingPage from './pages/ModelTrainingPage';
import AdminPage from './pages/AdminPage';
import SimpleAdminPanel from './components/SimpleAdminPanel';
import GlobalMegaMenu from './components/GlobalMegaMenu';

function App() {
  const [isAdminVisible, setIsAdminVisible] = useState(false);

  const toggleAdmin = () => {
    setIsAdminVisible(!isAdminVisible);
  };

  return (
    <Router>
      <div className="relative min-h-screen">
        {/* Global Mega Menu - Always visible */}
        <GlobalMegaMenu onAdminToggle={toggleAdmin} />
        
        {/* Main Content */}
        <main>
          <Routes>
            <Route path="/home" element={<HomePage />} />
            <Route path="/" element={<RealEstatePredictorPage />} />
            <Route path="/esg-agent" element={<ESGAgentPage />} />
            <Route path="/training" element={<ModelTrainingPage />} />
            <Route path="/admin" element={<AdminPage />} />
            
            {/* Remaining relevant agent routes */}
            <Route path="/agent/finance" element={<div className="p-8 text-center">🚧 Real Estate Finance Agent - Coming Soon</div>} />
            <Route path="/agent/passive" element={<div className="p-8 text-center">🚧 Investment Analysis Agent - Coming Soon</div>} />
          </Routes>
        </main>
        
        {/* Global Admin Panel - Available on all pages for demo */}
        {isAdminVisible && <SimpleAdminPanel onClose={() => setIsAdminVisible(false)} />}
      </div>
    </Router>
  );
}

export default App;
