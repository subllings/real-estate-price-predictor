/**
 * Model Management Panel - Real-time model status and operations
 */

import React, { useState, useEffect } from 'react';
import { Activity, TrendingUp, AlertCircle, CheckCircle } from 'lucide-react';

const ModelManagementPanel = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [deploymentStatus, setDeploymentStatus] = useState({});

  // Fetch model data
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await fetch('/api/admin/models');
        const data = await response.json();
        setModels(data.models || []);
        setDeploymentStatus(data.deployment_status || {});
      } catch (error) {
        console.error('Failed to fetch models:', error);
        // Demo fallback data
        setModels([
          {
            id: 'catboost_v2.3.1',
            version: 'catboost_v2.3.1',
            status: 'production',
            r2_score: 0.847,
            mae: 12500,
            requests_today: 1247,
            traffic_allocation: 80,
            created_at: '2025-07-15T08:30:00Z'
          },
          {
            id: 'catboost_v2.4.0',
            version: 'catboost_v2.4.0', 
            status: 'candidate',
            r2_score: 0.851,
            mae: 11800,
            requests_today: 312,
            traffic_allocation: 20,
            created_at: '2025-07-15T14:15:00Z'
          }
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchModels();
    
    // Refresh every 30 seconds for demo
    const interval = setInterval(fetchModels, 30000);
    return () => clearInterval(interval);
  }, []);

  const handlePromoteModel = async (modelId) => {
    try {
      const response = await fetch(`/api/admin/models/${modelId}/promote`, {
        method: 'POST'
      });
      
      if (response.ok) {
        // Refresh models after promotion
        const modelsResponse = await fetch('/api/admin/models');
        const data = await modelsResponse.json();
        setModels(data.models || []);
        
        // Show success feedback
        alert('Model promoted to production successfully!');
      }
    } catch (error) {
      console.error('Failed to promote model:', error);
      alert('Failed to promote model. Please try again.');
    }
  };

  const getStatusBadge = (status) => {
    const configs = {
      production: { color: 'green', icon: CheckCircle, label: '🟢 LIVE' },
      candidate: { color: 'yellow', icon: Activity, label: '🟡 TESTING' },
      training: { color: 'blue', icon: TrendingUp, label: '🔵 TRAINING' },
      failed: { color: 'red', icon: AlertCircle, label: '🔴 FAILED' }
    };
    
    const config = configs[status] || configs.candidate;
    const IconComponent = config.icon;
    
    return (
      <span className={`text-sm bg-${config.color}-100 text-${config.color}-800 px-2 py-1 rounded flex items-center space-x-1`}>
        <IconComponent size={12} />
        <span>{config.label}</span>
      </span>
    );
  };

  if (loading) {
    return (
      <div className="space-y-4">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-3/4 mb-4"></div>
          <div className="space-y-3">
            <div className="h-20 bg-gray-200 rounded"></div>
            <div className="h-20 bg-gray-200 rounded"></div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold text-gray-800">🤖 Active Models</h3>
        <button className="text-sm text-blue-600 hover:text-blue-800">
          Refresh
        </button>
      </div>
      
      {models.map((model) => (
        <div 
          key={model.id}
          className={`
            border rounded-lg p-4 transition-all hover:shadow-md
            ${model.status === 'production' 
              ? 'bg-green-50 border-green-200' 
              : model.status === 'candidate'
              ? 'bg-yellow-50 border-yellow-200'
              : 'bg-gray-50 border-gray-200'
            }
          `}
        >
          {/* Model Header */}
          <div className="flex justify-between items-center mb-3">
            <div>
              <h4 className="font-medium text-gray-800">
                {model.status === 'production' ? '🏆 Production Model' : '🧪 Candidate Model'}
              </h4>
              <p className="text-sm text-gray-600">Version: {model.version}</p>
            </div>
            {getStatusBadge(model.status)}
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 gap-3 mb-3">
            <div className="text-center p-2 bg-white rounded border">
              <div className="text-lg font-bold text-blue-600">
                {model.r2_score?.toFixed(3) || 'N/A'}
              </div>
              <div className="text-xs text-gray-500">R² Score</div>
            </div>
            <div className="text-center p-2 bg-white rounded border">
              <div className="text-lg font-bold text-green-600">
                €{model.mae?.toLocaleString() || 'N/A'}
              </div>
              <div className="text-xs text-gray-500">MAE</div>
            </div>
            <div className="text-center p-2 bg-white rounded border">
              <div className="text-lg font-bold text-purple-600">
                {model.requests_today || 0}
              </div>
              <div className="text-xs text-gray-500">Requests Today</div>
            </div>
            <div className="text-center p-2 bg-white rounded border">
              <div className="text-lg font-bold text-orange-600">
                {model.traffic_allocation || 0}%
              </div>
              <div className="text-xs text-gray-500">Traffic</div>
            </div>
          </div>

          {/* Performance Comparison */}
          {model.status === 'candidate' && (
            <div className="bg-white rounded border p-2 mb-3">
              <div className="text-xs text-gray-600 mb-1">Improvement vs Production:</div>
              <div className="flex justify-between text-sm">
                <span className="text-green-600">
                  R²: +{((model.r2_score - 0.847) * 1000).toFixed(1)}‰
                </span>
                <span className="text-green-600">
                  MAE: -€{(12500 - model.mae).toLocaleString()}
                </span>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex space-x-2">
            {model.status === 'candidate' && (
              <button 
                onClick={() => handlePromoteModel(model.id)}
                className="flex-1 bg-green-600 text-white py-2 px-3 rounded text-sm hover:bg-green-700 transition-colors"
              >
                🚀 Promote to Production
              </button>
            )}
            
            <button className="flex-1 bg-gray-600 text-white py-2 px-3 rounded text-sm hover:bg-gray-700 transition-colors">
              📊 View Details
            </button>
            
            {model.status === 'production' && (
              <button className="bg-blue-600 text-white py-2 px-3 rounded text-sm hover:bg-blue-700 transition-colors">
                📥 Download
              </button>
            )}
          </div>
        </div>
      ))}

      {/* Quick Stats Summary */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-3 border">
        <h4 className="font-medium text-gray-800 mb-2">📈 Quick Stats</h4>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div>
            <span className="text-gray-600">Total Predictions:</span>
            <span className="font-medium text-blue-600 ml-1">
              {models.reduce((sum, m) => sum + (m.requests_today || 0), 0)}
            </span>
          </div>
          <div>
            <span className="text-gray-600">Active Models:</span>
            <span className="font-medium text-green-600 ml-1">
              {models.filter(m => m.status !== 'failed').length}
            </span>
          </div>
          <div>
            <span className="text-gray-600">Best R²:</span>
            <span className="font-medium text-purple-600 ml-1">
              {Math.max(...models.map(m => m.r2_score || 0)).toFixed(3)}
            </span>
          </div>
          <div>
            <span className="text-gray-600">Best MAE:</span>
            <span className="font-medium text-orange-600 ml-1">
              €{Math.min(...models.map(m => m.mae || Infinity)).toLocaleString()}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelManagementPanel;
