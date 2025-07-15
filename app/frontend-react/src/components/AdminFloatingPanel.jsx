/**
 * Global Admin Floating Panel Component
 * Accessible from any page for seamless demo experience
 */

import React, { useState, useEffect } from 'react';
import { X, Settings, BarChart, FileText, Activity } from 'lucide-react';
import ModelManagementPanel from './ModelManagementPanel';
import TrainingStatusPanel from './TrainingStatusPanel'; 
import DocumentUploadPanel from './DocumentUploadPanel';
import RealTimeMetricsPanel from './RealTimeMetricsPanel';

const AdminFloatingPanel = () => {
  const [isVisible, setIsVisible] = useState(false);
  const [activeTab, setActiveTab] = useState('models');

  // Keyboard shortcut for demo (Ctrl+Admin)
  useEffect(() => {
    const handleKeyPress = (e) => {
      if (e.ctrlKey && e.key === 'a') {
        e.preventDefault();
        setIsVisible(!isVisible);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [isVisible]);

  const tabs = [
    { id: 'models', label: 'Models', icon: BarChart },
    { id: 'training', label: 'Training', icon: Activity },
    { id: 'documents', label: 'Documents', icon: FileText },
    { id: 'monitoring', label: 'Monitor', icon: Settings }
  ];

  return (
    <>
      {/* Admin Toggle Button - Always Visible */}
      <div className="fixed top-4 right-4 z-50">
        <button 
          onClick={() => setIsVisible(!isVisible)}
          className={`
            p-3 rounded-full shadow-lg transition-all duration-300 transform hover:scale-110
            ${isVisible 
              ? 'bg-red-600 text-white hover:bg-red-700' 
              : 'bg-blue-600 text-white hover:bg-blue-700'
            }
          `}
          title={`${isVisible ? 'Close' : 'Open'} Admin Panel (Ctrl+A)`}
        >
          {isVisible ? <X size={20} /> : <Settings size={20} />}
        </button>
      </div>

      {/* Sliding Admin Panel */}
      {isVisible && (
        <>
          {/* Backdrop */}
          <div 
            className="fixed inset-0 bg-black bg-opacity-50 z-40"
            onClick={() => setIsVisible(false)}
          />
          
          {/* Panel */}
          <div className="fixed right-0 top-0 h-full w-96 bg-white shadow-2xl z-50 transform transition-transform duration-300 flex flex-col">
            
            {/* Admin Header */}
            <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4">
              <div className="flex justify-between items-center mb-3">
                <h2 className="text-xl font-bold">🔧 Admin Dashboard</h2>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                  <span className="text-xs">LIVE</span>
                </div>
              </div>
              
              {/* Tab Navigation */}
              <div className="flex space-x-1">
                {tabs.map(tab => {
                  const IconComponent = tab.icon;
                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      className={`
                        flex items-center space-x-1 px-3 py-2 rounded text-xs transition-all
                        ${activeTab === tab.id 
                          ? 'bg-white text-blue-600 shadow-md' 
                          : 'bg-blue-500 hover:bg-blue-400 text-white'
                        }
                      `}
                    >
                      <IconComponent size={14} />
                      <span>{tab.label}</span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Dynamic Content */}
            <div className="flex-1 overflow-y-auto">
              <div className="p-4">
                {activeTab === 'models' && <ModelManagementPanel />}
                {activeTab === 'training' && <TrainingStatusPanel />}
                {activeTab === 'documents' && <DocumentUploadPanel />}
                {activeTab === 'monitoring' && <RealTimeMetricsPanel />}
              </div>
            </div>

            {/* Quick Actions Footer */}
            <div className="border-t bg-gray-50 p-3">
              <div className="flex space-x-2">
                <button className="flex-1 bg-green-600 text-white py-2 px-3 rounded text-sm hover:bg-green-700 transition-colors">
                  🚀 Deploy Model
                </button>
                <button className="flex-1 bg-blue-600 text-white py-2 px-3 rounded text-sm hover:bg-blue-700 transition-colors">
                  📊 View Analytics
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </>
  );
};

export default AdminFloatingPanel;
