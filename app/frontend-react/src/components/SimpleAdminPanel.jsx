/**
 * Simplified Admin Panel for testing - no external dependencies
 */

import React, { useState, useEffect } from 'react';

const SimpleAdminPanel = ({ onClose }) => {
  const [activeTab, setActiveTab] = useState('models');

  // Global keyboard shortcut
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.ctrlKey && event.key === 'a') {
        event.preventDefault();
        if (onClose) onClose();
      }
      if (event.key === 'Escape') {
        if (onClose) onClose();
      }
    };

    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  return (
    <>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black bg-opacity-30 backdrop-blur-sm z-40"
        onClick={onClose}
      />
      
      {/* Panel */}
      <div className="fixed top-0 right-0 w-96 h-full bg-white shadow-2xl z-50 transform transition-transform duration-300">
        {/* Header */}
        <div className="bg-gradient-to-r from-blue-600 to-purple-600 text-white p-4 flex justify-between items-center">
          <h2 className="text-lg font-semibold">🔧 Admin Panel</h2>
          <button
            onClick={onClose}
            className="text-white hover:bg-white hover:bg-opacity-20 rounded p-1"
          >
            ✕
          </button>
        </div>

        {/* Tabs */}
        <div className="flex border-b">
          {[
            { id: 'models', label: '📊 Models' },
            { id: 'training', label: '🚀 Training' },
            { id: 'documents', label: '📄 Documents' },
            { id: 'monitoring', label: '📈 Monitoring' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex-1 py-3 px-2 text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'border-b-2 border-blue-600 text-blue-600 bg-blue-50'
                  : 'text-gray-600 hover:text-blue-600 hover:bg-gray-50'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="p-4 h-full overflow-y-auto">
          {activeTab === 'models' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">📊 Model Management</h3>
              <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-medium">Production Model</span>
                  <span className="bg-green-600 text-white px-2 py-1 rounded text-xs">LIVE</span>
                </div>
                <div className="text-sm text-gray-600">
                  <div>R² Score: <span className="font-medium text-green-600">0.89</span></div>
                  <div>MAE: <span className="font-medium">€8,450</span></div>
                  <div>Inference: <span className="font-medium">145ms</span></div>
                </div>
              </div>
              
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-medium">Candidate Model</span>
                  <span className="bg-blue-600 text-white px-2 py-1 rounded text-xs">TESTING</span>
                </div>
                <div className="text-sm text-gray-600">
                  <div>R² Score: <span className="font-medium text-blue-600">0.91</span></div>
                  <div>MAE: <span className="font-medium">€7,200</span></div>
                  <div>A/B Split: <span className="font-medium">20%</span></div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'training' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">🚀 Training Status</h3>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
                <div className="flex justify-between items-center mb-2">
                  <span className="font-medium">Azure ML Job</span>
                  <span className="bg-blue-600 text-white px-2 py-1 rounded text-xs">RUNNING</span>
                </div>
                <div className="text-sm text-gray-600 mb-2">
                  <div>Progress: <span className="font-medium">73%</span></div>
                  <div>ETA: <span className="font-medium">8 minutes</span></div>
                  <div>Compute: <span className="font-medium">Tesla V100</span></div>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-blue-600 h-2 rounded-full" style={{width: '73%'}}></div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'documents' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">📄 Document Management</h3>
              <div className="space-y-2">
                <div className="bg-gray-50 border rounded p-2 text-sm">
                  <div className="font-medium">Belgian Housing Q3 2025.pdf</div>
                  <div className="text-gray-600">2.4 MB • 45 vectors • Processed</div>
                </div>
                <div className="bg-gray-50 border rounded p-2 text-sm">
                  <div className="font-medium">Antwerp Zoning Regulations.docx</div>
                  <div className="text-gray-600">1.8 MB • 32 vectors • Processing...</div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'monitoring' && (
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">📈 System Monitoring</h3>
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-green-50 border border-green-200 rounded p-2 text-center">
                  <div className="font-medium text-green-600">99.9%</div>
                  <div className="text-gray-600">Uptime</div>
                </div>
                <div className="bg-blue-50 border border-blue-200 rounded p-2 text-center">
                  <div className="font-medium text-blue-600">87ms</div>
                  <div className="text-gray-600">Response</div>
                </div>
                <div className="bg-purple-50 border border-purple-200 rounded p-2 text-center">
                  <div className="font-medium text-purple-600">1,247</div>
                  <div className="text-gray-600">Requests</div>
                </div>
                <div className="bg-orange-50 border border-orange-200 rounded p-2 text-center">
                  <div className="font-medium text-orange-600">€62</div>
                  <div className="text-gray-600">Monthly Cost</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </>
  );
};

export default SimpleAdminPanel;
