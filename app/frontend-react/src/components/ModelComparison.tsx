import React, { useState } from 'react';

const ModelComparison = ({ models, onClose }) => {
  const [selectedMetric, setSelectedMetric] = useState('r2_test');
  const [viewMode, setViewMode] = useState('table'); // 'table' | 'chart'

  if (!models || models.length < 2) {
    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
        <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
          <h3 className="text-lg font-semibold mb-4">Comparaison impossible</h3>
          <p className="text-gray-600 mb-4">
            Sélectionnez au moins 2 modèles pour effectuer une comparaison.
          </p>
          <button
            onClick={onClose}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
          >
            Fermer
          </button>
        </div>
      </div>
    );
  }

  const metrics = {
    r2_test: { label: 'R² Test', format: (v) => `${(v * 100).toFixed(1)}%`, higherBetter: true },
    rmse_test: { label: 'RMSE Test', format: (v) => `${Math.round(v).toLocaleString()}€`, higherBetter: false },
    mae_test: { label: 'MAE Test', format: (v) => `${Math.round(v).toLocaleString()}€`, higherBetter: false },
    r2_gap: { label: 'Gap R²', format: (v) => v.toFixed(3), higherBetter: false }
  };

  const getBestValue = (metric) => {
    const values = models.map(m => m[metric]);
    return metrics[metric].higherBetter ? Math.max(...values) : Math.min(...values);
  };

  const getWorstValue = (metric) => {
    const values = models.map(m => m[metric]);
    return metrics[metric].higherBetter ? Math.min(...values) : Math.max(...values);
  };

  const getRank = (model, metric) => {
    const sorted = [...models].sort((a, b) => {
      return metrics[metric].higherBetter ? b[metric] - a[metric] : a[metric] - b[metric];
    });
    return sorted.findIndex(m => m.model === model.model) + 1;
  };

  const getPerformanceColor = (value, metric) => {
    const best = getBestValue(metric);
    const worst = getWorstValue(metric);
    const range = Math.abs(best - worst);
    
    if (range === 0) return 'bg-blue-100 text-blue-800';
    
    const normalized = Math.abs(value - worst) / range;
    
    if (normalized > 0.8) return 'bg-green-100 text-green-800';
    if (normalized > 0.6) return 'bg-yellow-100 text-yellow-800';
    if (normalized > 0.4) return 'bg-orange-100 text-orange-800';
    return 'bg-red-100 text-red-800';
  };

  const ComparisonTable = () => (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
              Modèle
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
              Catégorie
            </th>
            {Object.entries(metrics).map(([key, metric]) => (
              <th key={key} className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
                {metric.label}
              </th>
            ))}
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
              Features
            </th>
            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">
              Date
            </th>
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {models.map((model, index) => (
            <tr key={model.model} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
              <td className="px-6 py-4">
                <div className="flex items-center">
                  <div className={`w-3 h-3 rounded-full mr-3`} style={{ backgroundColor: model.color }}></div>
                  <div>
                    <div className="text-sm font-medium text-gray-900 truncate max-w-xs">
                      {model.model.split('/').pop() || model.model}
                    </div>
                    <div className="text-xs text-gray-500">
                      {model.experiment_name || 'Standard'}
                    </div>
                  </div>
                </div>
              </td>
              <td className="px-6 py-4">
                <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full text-white`}
                      style={{ backgroundColor: model.color }}>
                  {model.category}
                </span>
              </td>
              {Object.entries(metrics).map(([key, metric]) => (
                <td key={key} className="px-6 py-4">
                  <div className="flex items-center gap-2">
                    <span className={`px-2 py-1 rounded text-sm font-medium ${getPerformanceColor(model[key], key)}`}>
                      {metric.format(model[key])}
                    </span>
                    <span className="text-xs text-gray-500">
                      #{getRank(model, key)}
                    </span>
                  </div>
                </td>
              ))}
              <td className="px-6 py-4 text-sm text-gray-900">
                {model.n_features}
              </td>
              <td className="px-6 py-4 text-sm text-gray-500">
                {new Date(model.timestamp).toLocaleDateString('fr-FR')}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );

  const ComparisonChart = () => (
    <div className="space-y-6">
      {/* Metric selector */}
      <div className="flex gap-2 flex-wrap">
        {Object.entries(metrics).map(([key, metric]) => (
          <button
            key={key}
            onClick={() => setSelectedMetric(key)}
            className={`px-4 py-2 rounded-lg text-sm font-medium ${
              selectedMetric === key
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            {metric.label}
          </button>
        ))}
      </div>

      {/* Bar chart visualization */}
      <div className="bg-gray-50 p-6 rounded-lg">
        <h4 className="text-lg font-semibold mb-4">
          Comparaison: {metrics[selectedMetric].label}
        </h4>
        
        <div className="space-y-3">
          {models
            .sort((a, b) => {
              return metrics[selectedMetric].higherBetter 
                ? b[selectedMetric] - a[selectedMetric]
                : a[selectedMetric] - b[selectedMetric];
            })
            .map((model, index) => {
              const value = model[selectedMetric];
              const bestValue = getBestValue(selectedMetric);
              const worstValue = getWorstValue(selectedMetric);
              const range = Math.abs(bestValue - worstValue);
              const normalizedValue = range === 0 ? 100 : (Math.abs(value - worstValue) / range) * 100;
              
              return (
                <div key={model.model} className="flex items-center gap-4">
                  <div className="w-8 text-center">
                    <span className="text-sm font-bold text-gray-600">#{index + 1}</span>
                  </div>
                  
                  <div className="flex-1">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-sm font-medium text-gray-900 truncate max-w-xs">
                        {model.model.split('/').pop() || model.model}
                      </span>
                      <span className="text-sm font-semibold">
                        {metrics[selectedMetric].format(value)}
                      </span>
                    </div>
                    
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="h-2 rounded-full"
                        style={{
                          width: `${normalizedValue}%`,
                          backgroundColor: model.color
                        }}
                      />
                    </div>
                  </div>
                </div>
              );
            })}
        </div>
      </div>

      {/* Performance insights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-green-50 p-4 rounded-lg border border-green-200">
          <h5 className="font-semibold text-green-800 mb-2">🏆 Meilleur</h5>
          <div className="text-sm">
            {(() => {
              const bestModel = models.find(m => m[selectedMetric] === getBestValue(selectedMetric));
              return (
                <div>
                  <p className="font-medium truncate">{bestModel?.model.split('/').pop()}</p>
                  <p className="text-green-700">
                    {metrics[selectedMetric].format(bestModel?.[selectedMetric])}
                  </p>
                </div>
              );
            })()}
          </div>
        </div>

        <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
          <h5 className="font-semibold text-blue-800 mb-2">📊 Moyenne</h5>
          <div className="text-sm">
            <p className="text-blue-700 font-medium">
              {metrics[selectedMetric].format(
                models.reduce((sum, m) => sum + m[selectedMetric], 0) / models.length
              )}
            </p>
            <p className="text-xs text-blue-600">Sur {models.length} modèles</p>
          </div>
        </div>

        <div className="bg-red-50 p-4 rounded-lg border border-red-200">
          <h5 className="font-semibold text-red-800 mb-2">📉 Plus faible</h5>
          <div className="text-sm">
            {(() => {
              const worstModel = models.find(m => m[selectedMetric] === getWorstValue(selectedMetric));
              return (
                <div>
                  <p className="font-medium truncate">{worstModel?.model.split('/').pop()}</p>
                  <p className="text-red-700">
                    {metrics[selectedMetric].format(worstModel?.[selectedMetric])}
                  </p>
                </div>
              );
            })()}
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-2xl max-w-7xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="p-6 border-b bg-gradient-to-r from-purple-600 to-pink-600 text-white rounded-t-lg">
          <div className="flex justify-between items-center">
            <div>
              <h2 className="text-2xl font-bold">⚖️ Comparaison de Modèles</h2>
              <p className="text-purple-100 mt-1">
                Analyse comparative de {models.length} modèles sélectionnés
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-white hover:text-gray-200 text-2xl font-bold"
            >
              ✕
            </button>
          </div>
        </div>

        {/* View mode selector */}
        <div className="border-b">
          <nav className="flex space-x-8 px-6">
            <button
              onClick={() => setViewMode('table')}
              className={`py-4 px-2 border-b-2 font-medium text-sm ${
                viewMode === 'table'
                  ? 'border-purple-500 text-purple-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              📊 Vue Tableau
            </button>
            <button
              onClick={() => setViewMode('chart')}
              className={`py-4 px-2 border-b-2 font-medium text-sm ${
                viewMode === 'chart'
                  ? 'border-purple-500 text-purple-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              📈 Vue Graphique
            </button>
          </nav>
        </div>

        {/* Content */}
        <div className="p-6">
          {viewMode === 'table' ? <ComparisonTable /> : <ComparisonChart />}
        </div>

        {/* Summary footer */}
        <div className="border-t px-6 py-4 bg-gray-50 rounded-b-lg">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="font-medium text-gray-600">Modèles comparés:</span>
              <p className="text-lg font-bold text-gray-900">{models.length}</p>
            </div>
            <div>
              <span className="font-medium text-gray-600">Meilleur R²:</span>
              <p className="text-lg font-bold text-green-600">
                {(getBestValue('r2_test') * 100).toFixed(1)}%
              </p>
            </div>
            <div>
              <span className="font-medium text-gray-600">Meilleur RMSE:</span>
              <p className="text-lg font-bold text-green-600">
                {Math.round(getBestValue('rmse_test')).toLocaleString()}€
              </p>
            </div>
            <div>
              <span className="font-medium text-gray-600">Plus petit gap:</span>
              <p className="text-lg font-bold text-green-600">
                {getBestValue('r2_gap').toFixed(3)}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelComparison;
