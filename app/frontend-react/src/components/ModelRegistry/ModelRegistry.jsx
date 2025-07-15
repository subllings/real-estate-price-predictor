// ModelRegistry.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { PREDICTION_API_URL } from '../../config/api';
import './ModelRegistry.css';

const ModelRegistry = () => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [promotionStatus, setPromotionStatus] = useState('');

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`${PREDICTION_API_URL}/models`);
      setModels(response.data.models);
      setError(null);
    } catch (err) {
      setError(`Failed to fetch models: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const refreshRegistry = async () => {
    try {
      setLoading(true);
      const response = await axios.post(`${PREDICTION_API_URL}/models/refresh`);
      setModels(response.data.models);
      setPromotionStatus('Model registry refreshed successfully!');
      setTimeout(() => setPromotionStatus(''), 3000);
    } catch (err) {
      setError(`Failed to refresh registry: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const promoteModel = async (modelId, variant) => {
    try {
      const response = await axios.post(`${PREDICTION_API_URL}/models/${modelId}/promote`, null, {
        params: { variant }
      });
      setPromotionStatus(`✅ ${response.data.message}`);
      setTimeout(() => setPromotionStatus(''), 5000);
      
      // Refresh the model list to show updated statuses
      await fetchModels();
    } catch (err) {
      setError(`Failed to promote model: ${err.response?.data?.detail || err.message}`);
    }
  };

  const getBestModel = async (metric) => {
    try {
      const response = await axios.get(`${PREDICTION_API_URL}/models/best/${metric}`);
      setSelectedModel(response.data.model_info);
      setPromotionStatus(`🏆 Best model by ${metric}: ${response.data.best_model_id}`);
      setTimeout(() => setPromotionStatus(''), 5000);
    } catch (err) {
      setError(`Failed to get best model: ${err.response?.data?.detail || err.message}`);
    }
  };

  const formatMetric = (value) => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'number') {
      return value < 1 ? value.toFixed(4) : value.toLocaleString();
    }
    return value;
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case 'production':
        return <span className="status-badge production">🟢 Production</span>;
      case 'available':
        return <span className="status-badge available">🔵 Available</span>;
      default:
        return <span className="status-badge unknown">⚪ Unknown</span>;
    }
  };

  const getVariantBadge = (variant) => {
    switch (variant) {
      case 'all_features':
        return <span className="variant-badge all">📊 All Features</span>;
      case 'top_features':
        return <span className="variant-badge top">⭐ Top Features</span>;
      default:
        return <span className="variant-badge unknown">❓ Unknown</span>;
    }
  };

  if (loading) {
    return (
      <div className="model-registry">
        <div className="loading">
          <div className="spinner"></div>
          <p>Loading models...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="model-registry">
      <div className="registry-header">
        <h2>🤖 Model Registry</h2>
        <div className="header-actions">
          <button onClick={refreshRegistry} className="refresh-btn">
            🔄 Refresh Registry
          </button>
          <div className="best-model-actions">
            <button onClick={() => getBestModel('r2')} className="best-btn r2">
              🏆 Best R²
            </button>
            <button onClick={() => getBestModel('mae')} className="best-btn mae">
              🎯 Best MAE
            </button>
          </div>
        </div>
      </div>

      {promotionStatus && (
        <div className="promotion-status">
          {promotionStatus}
        </div>
      )}

      {error && (
        <div className="error-message">
          ❌ {error}
        </div>
      )}

      <div className="models-summary">
        <div className="summary-card">
          <h3>📈 Registry Statistics</h3>
          <p><strong>Total Models:</strong> {models.length}</p>
          <p><strong>Production Models:</strong> {models.filter(m => m.status === 'production').length}</p>
          <p><strong>CatBoost Models:</strong> {models.filter(m => m.type === 'CatBoost').length}</p>
          <p><strong>XGBoost Models:</strong> {models.filter(m => m.type === 'XGBoost').length}</p>
        </div>
      </div>

      <div className="models-grid">
        {models.map((model) => (
          <div key={model.model_id} className={`model-card ${model.status}`}>
            <div className="model-header">
              <h3>{model.name}</h3>
              <div className="model-badges">
                {getStatusBadge(model.status)}
                {getVariantBadge(model.variant)}
              </div>
            </div>

            <div className="model-info">
              <div className="model-type">
                <strong>Type:</strong> {model.type}
              </div>
              <div className="model-created">
                <strong>Created:</strong> {new Date(model.created_at).toLocaleDateString()}
              </div>
              {model.feature_count && (
                <div className="feature-count">
                  <strong>Features:</strong> {model.feature_count}
                </div>
              )}
            </div>

            {model.metrics && (
              <div className="model-metrics">
                <h4>📊 Performance Metrics</h4>
                <div className="metrics-grid">
                  {model.r2 && (
                    <div className="metric">
                      <span className="metric-label">R²:</span>
                      <span className="metric-value r2">{formatMetric(model.r2)}</span>
                    </div>
                  )}
                  {model.mae && (
                    <div className="metric">
                      <span className="metric-label">MAE:</span>
                      <span className="metric-value mae">€{formatMetric(model.mae)}</span>
                    </div>
                  )}
                  {model.rmse && (
                    <div className="metric">
                      <span className="metric-label">RMSE:</span>
                      <span className="metric-value rmse">€{formatMetric(model.rmse)}</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            <div className="model-actions">
              {model.status !== 'production' && (
                <button
                  onClick={() => promoteModel(model.model_id, model.variant)}
                  className="promote-btn"
                >
                  🚀 Promote to Production
                </button>
              )}
              <button
                onClick={() => setSelectedModel(model)}
                className="details-btn"
              >
                📋 View Details
              </button>
            </div>
          </div>
        ))}
      </div>

      {selectedModel && (
        <div className="model-details-modal" onClick={() => setSelectedModel(null)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <h3>📋 Model Details: {selectedModel.name}</h3>
              <button onClick={() => setSelectedModel(null)} className="close-btn">
                ✕
              </button>
            </div>
            <div className="modal-body">
              <pre>{JSON.stringify(selectedModel, null, 2)}</pre>
            </div>
          </div>
        </div>
      )}

      {models.length === 0 && !loading && (
        <div className="no-models">
          <h3>📭 No Models Found</h3>
          <p>No models are currently available in the registry.</p>
          <button onClick={refreshRegistry} className="refresh-btn">
            🔄 Refresh Registry
          </button>
        </div>
      )}
    </div>
  );
};

export default ModelRegistry;
