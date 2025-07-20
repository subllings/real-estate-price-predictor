import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { GoogleOAuthProvider } from '@react-oauth/google';
import { UserProvider, useUser } from './contexts/UserContext';
import HomePage from './pages/HomePage';
import RealEstatePredictorPage from './pages/RealEstatePredictorPage';
import ESGAgentPage from './pages/ESGAgentPage';
import ESGAgentPageNew from './pages/ESGAgentPageNew';
import ModelTrainingPage from './pages/ModelTrainingPage';
import AdminPage from './pages/AdminPage';
import LoginPage from './components/Auth/LoginPage';
import AdminPanel from './components/AdminPanel/AdminPanel';
import SimpleAdminPanelTest from './components/SimpleAdminPanelTest';
import TestAdminPanel from './TestAdminPanel';
import GlobalMegaMenu from './components/GlobalMegaMenu';

// Login Route Component
const LoginRoute = () => {
  const { login } = useUser();
  return <LoginPage onLogin={login} />;
};

// Protected Route Component
const ProtectedRoute = ({ children }) => {
  const { user, loading, login } = useUser();
  
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-green-500 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading authentication...</p>
        </div>
      </div>
    );
  }
  
  if (!user) {
    return <LoginPage onLogin={login} />;
  }
  
  return children;
};

function App() {
  const [isAdminVisible, setIsAdminVisible] = useState(false);

  const toggleAdmin = () => {
    console.log('toggleAdmin called, current isAdminVisible:', isAdminVisible);
    setIsAdminVisible(!isAdminVisible);
    console.log('toggleAdmin setting isAdminVisible to:', !isAdminVisible);
  };

  return (
    <GoogleOAuthProvider clientId={process.env.REACT_APP_GOOGLE_CLIENT_ID || "demo-client-id"}>
      <UserProvider>
        <Router>
          <div className="relative min-h-screen">
            <Routes>
              <Route path="/login" element={<LoginRoute />} />
              <Route path="/*" element={
                <ProtectedRoute>
                  {/* Global Mega Menu - Only visible when authenticated */}
                  <GlobalMegaMenu onAdminToggle={toggleAdmin} />
                  
                  {/* Main Content */}
                  <main>
                    <Routes>
                      <Route path="/home" element={<HomePage />} />
                      <Route path="/" element={<RealEstatePredictorPage />} />
                      <Route path="/esg-agent" element={<ESGAgentPageNew />} />
                      <Route path="/esg-agent-old" element={<ESGAgentPage />} />
                      <Route path="/training" element={<ModelTrainingPage />} />
                      <Route path="/admin" element={<AdminPage />} />
                      <Route path="/test-admin" element={<TestAdminPanel />} />
                      
                      {/* Remaining relevant agent routes */}
                      <Route path="/agent/finance" element={<div className="p-8 text-center">Real Estate Finance Agent - Coming Soon</div>} />
                      <Route path="/agent/passive" element={<div className="p-8 text-center">Investment Analysis Agent - Coming Soon</div>} />
                    </Routes>
                  </main>
                  
                  {/* Global Admin Panel - Available on all pages when authenticated */}
                  {isAdminVisible && (
                    <AdminPanel 
                      isExpanded={isAdminVisible}
                      onToggle={toggleAdmin}
                      onClose={() => setIsAdminVisible(false)} 
                    />
                  )}
                </ProtectedRoute>
              } />
            </Routes>
          </div>
        </Router>
      </UserProvider>
    </GoogleOAuthProvider>
  );
}

export default App;
