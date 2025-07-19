import React, { useState, useEffect } from 'react';
import { useExperiments } from '../hooks/useExperiments';

const ModelTrainingPage = () => {
  const [activeTab, setActiveTab] = useState('experiments');
  const [isTraining, setIsTraining] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [showHyperparametersPanel, setShowHyperparametersPanel] = useState(false);
  
  const { 
    experiments, 
    summary, 
    loading, 
    error, 
    refresh 
  } = useExperiments();

  const tabs = [
    { id: 'pipeline', label: 'Training Pipeline' },
    { id: 'experiments', label: 'Experiments & Optimization' },
    { id: 'deployment', label: 'Deployment' }
  ];

  const startTraining = () => {
    setIsTraining(true);
    setTimeout(() => setIsTraining(false), 5000);
  };

  const handleModelClick = (experiment) => {
    setSelectedModel(experiment);
    setShowHyperparametersPanel(true);
  };

  const getTopHyperparametersForModel = (modelName) => {
    // Filtrer les expériences pour ce modèle spécifique
    const modelExperiments = experiments.filter(exp => 
      exp.model_name && exp.model_name.includes(modelName.split(' ')[0]) // Match base model name
    );
    
    // Trier par R² et prendre les 10 meilleurs
    const topExperiments = modelExperiments
      .filter(exp => exp.hyperparameters) // Seulement ceux avec hyperparamètres
      .sort((a, b) => (b.r2_test || 0) - (a.r2_test || 0))
      .slice(0, 10);
    
    return topExperiments;
  };

  const formatDate = (timestamp) => {
    if (!timestamp) return 'N/A';
    try {
      const date = new Date(timestamp);
      return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit'
      }) + ' ' + date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      });
    } catch (e) {
      return timestamp;
    }
  };

  const formatR2Score = (score) => {
    if (score === null || score === undefined) return '0.000000';
    return score.toFixed(6);
  };

  const formatMAE = (mae) => {
    if (mae === null || mae === undefined || mae === 0) return 'N/A';
    if (mae >= 1000) {
      return `${(mae / 1000).toFixed(1)}k€`;
    }
    return `${mae.toFixed(0)}€`;
  };

  const formatDuration = (duration) => {
    if (!duration || duration === 0) return 'N/A';
    
    // Conversion en nombre si c'est une chaîne
    const time = typeof duration === 'string' ? parseFloat(duration) : duration;
    if (isNaN(time) || time === 0) return 'N/A';
    
    // Formatage selon la durée
    if (time < 60) {
      return `${time.toFixed(1)}s`;
    } else if (time < 3600) {
      const minutes = Math.floor(time / 60);
      const seconds = Math.floor(time % 60);
      return seconds > 0 ? `${minutes}m ${seconds}s` : `${minutes}m`;
    } else {
      const hours = Math.floor(time / 3600);
      const minutes = Math.floor((time % 3600) / 60);
      return minutes > 0 ? `${hours}h ${minutes}m` : `${hours}h`;
    }
  };

  const formatTrainingTime = (time) => {
    if (!time || time === 0) return 'N/A';
    
    // Si c'est déjà une chaîne formatée (comme "2h 30m"), la retourner
    if (typeof time === 'string' && (time.includes('h') || time.includes('m') || time.includes('s'))) {
      return time;
    }
    
    // Sinon, formater comme une durée
    return formatDuration(time);
  };

  const getScoreColor = (score) => {
    if (!score) return 'text-gray-500';
    if (score >= 0.85) return 'text-green-600';
    if (score >= 0.75) return 'text-blue-600';
    if (score >= 0.65) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getR2GapColor = (gap) => {
    const gapValue = parseFloat(gap) || 0;
    
    // Logique alignée avec train_test_metrics_logger.py
    if (gapValue < 0) {
      return 'text-purple-600 font-medium'; // Possible underfitting
    } else if (gapValue < 0.05) {
      return 'text-green-600 font-medium'; // Excellent generalization
    } else if (gapValue < 0.08) {
      return 'text-blue-600 font-medium'; // Good generalization
    } else if (gapValue < 0.12) {
      return 'text-yellow-600 font-medium'; // Moderate overfitting
    } else {
      return 'text-red-600 font-medium'; // Strong overfitting
    }
  };

  const getDiagnosticColor = (diagnostic) => {
    switch (diagnostic) {
      // Nouveaux diagnostics alignés avec train_test_metrics_logger.py
      case 'Excellent generalization': return 'bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs font-medium';
      case 'Good generalization': return 'bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs font-medium';
      case 'Moderate overfitting': return 'bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full text-xs font-medium';
      case 'Strong overfitting': return 'bg-red-100 text-red-800 px-2 py-1 rounded-full text-xs font-medium';
      case 'Possible underfitting': return 'bg-purple-100 text-purple-800 px-2 py-1 rounded-full text-xs font-medium';
      
      // Support anciens diagnostics pour compatibilité
      case 'Excellent': return 'bg-green-100 text-green-800 px-2 py-1 rounded-full text-xs font-medium';
      case 'Good': return 'bg-blue-100 text-blue-800 px-2 py-1 rounded-full text-xs font-medium';
      case 'Fair': return 'bg-yellow-100 text-yellow-800 px-2 py-1 rounded-full text-xs font-medium';
      case 'Poor': return 'bg-red-100 text-red-800 px-2 py-1 rounded-full text-xs font-medium';
      
      default: return 'bg-gray-100 text-gray-800 px-2 py-1 rounded-full text-xs font-medium';
    }
  };

  const getGeneralizationDiagnostic = (r2_train, r2_test) => {
    if (!r2_train || !r2_test) return 'Unknown';
    const r2_gap = r2_train - r2_test;
    
    // Logique exacte depuis train_test_metrics_logger.py
    if (r2_gap < 0) {
      return 'Possible underfitting';
    } else if (r2_gap < 0.05) {
      return 'Excellent generalization';
    } else if (r2_gap < 0.08) {
      return 'Good generalization';
    } else if (r2_gap < 0.12) {
      return 'Moderate overfitting';
    } else {
      return 'Strong overfitting';
    }
  };

  const getStatusBadge = (status) => {
    switch (status) {
      case 'completed':
        return <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">Complete</span>;
      case 'failed':
        return <span className="bg-red-100 text-red-800 px-2 py-1 rounded text-xs">Failed</span>;
      case 'running':
        return <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">Running</span>;
      default:
        return <span className="bg-gray-100 text-gray-800 px-2 py-1 rounded text-xs">Complete</span>;
    }
  };

  const renderExperimentsTab = () => {
    // Traitement des données pour le format du tableau avec métriques structurées
    const processedExperiments = experiments.map((exp, index) => {
      // Trier par R² test décroissant
      const sortedExperiments = [...experiments].sort((a, b) => (b.r2_test || 0) - (a.r2_test || 0));
      const rank = sortedExperiments.findIndex(e => e.id === exp.id) + 1;
      
      return {
        ...exp,
        rank,
        best: rank === 1 ? '✓' : '',
        model: exp.model_name || 'CatBoost CV (All Features)',
        r2_gap: exp.r2_gap ? exp.r2_gap.toFixed(6) : ((exp.r2_train || 0) - (exp.r2_test || 0)).toFixed(6),
        r2_gap_diagnostic: exp.generalization_status || getGeneralizationDiagnostic(exp.r2_train, exp.r2_test),
        n_features: exp.feature_count || 2885
      };
    }).sort((a, b) => a.rank - b.rank);

    return (
      <div className="space-y-6">
        
        {/* Information Panel */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <div className="text-blue-500">
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div>
              <h4 className="font-medium text-blue-800">Interactive Experiments & Optimization</h4>
              <p className="text-sm text-blue-600 mt-1">
                Click on any experiment row to view detailed hyperparameters and optimization history. 
                Launch new optimizations via the LLM agent.
              </p>
            </div>
            <button
              onClick={() => window.open('http://localhost:7860', '_blank')}
              className="ml-auto bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 transition-colors"
            >
              🚀 Launch Optimization
            </button>
          </div>
        </div>

        {/* Header avec statistiques enrichies */}
        <div className="bg-white rounded-lg border p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-800">Experiment History</h3>
            <button 
              onClick={refresh}
              className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-colors"
            >
              Refresh
            </button>
          </div>
          
          {/* Statistiques de résumé enrichies */}
          {summary && (
            <div className="grid grid-cols-5 gap-4 mb-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{summary.total_experiments}</div>
                <div className="text-sm text-gray-600">Total Experiments</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">{summary.best_r2_score?.toFixed(3)}</div>
                <div className="text-sm text-gray-600">Best R² Score</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">{summary.average_r2_score?.toFixed(3)}</div>
                <div className="text-sm text-gray-600">Average R² Score</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">{summary.average_r2_gap?.toFixed(4)}</div>
                <div className="text-sm text-gray-600">Avg R² Gap</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-600">
                  {summary.latest_experiment?.timestamp ? 
                    new Date(summary.latest_experiment.timestamp).toLocaleDateString('en-US', {
                      year: 'numeric',
                      month: '2-digit',
                      day: '2-digit'
                    }) : 
                    'N/A'
                  }
                </div>
                <div className="text-sm text-gray-600">Latest Date</div>
              </div>
            </div>
          )}
          
          {/* Indicateur de meilleure généralisation */}
          {summary?.best_generalization && (
            <div className="bg-indigo-50 rounded-lg p-4 mb-6">
              <h4 className="font-medium text-indigo-800 mb-2">Best Generalization</h4>
              <div className="flex items-center justify-between">
                <span className="text-sm text-indigo-600">
                  R² Gap: {summary.best_generalization.r2_gap?.toFixed(6)}
                </span>
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  summary.best_generalization.generalization_status === 'Excellent' ? 'bg-green-100 text-green-800' :
                  summary.best_generalization.generalization_status === 'Good' ? 'bg-blue-100 text-blue-800' :
                  summary.best_generalization.generalization_status === 'Fair' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-red-100 text-red-800'
                }`}>
                  {summary.best_generalization.generalization_status}
                </span>
              </div>
            </div>
          )}
          
          {loading && (
            <div className="bg-blue-50 rounded-lg p-4 text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
              <p className="mt-2 text-blue-600">Loading experiments...</p>
            </div>
          )}
          
          {error && (
            <div className="bg-red-50 rounded-lg p-4">
              <div className="flex items-center">
                <svg className="w-5 h-5 text-red-400 mr-2" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <div>
                  <p className="text-red-800 font-medium">Error loading experiments</p>
                  <p className="text-red-600 text-sm">Failed to fetch</p>
                </div>
              </div>
            </div>
          )}
          
          {!loading && !error && experiments.length === 0 && (
            <div className="bg-gray-50 rounded-lg p-8 text-center">
              <p className="text-gray-500">No experiments found</p>
            </div>
          )}
        </div>

        {/* Tableau des expériences - Format exact selon les métriques structurées */}
        {!loading && !error && experiments.length > 0 && (
          <div className="bg-white rounded-lg border overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rank</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Best</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Timestamp</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Model</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Training Time</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">MAE Train</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">RMSE Train</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">R² Train</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">MAE Test</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">RMSE Test</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">R² Test</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">R² Gap</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Generalization Status</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">N Features</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {processedExperiments.map((exp, index) => (
                    <tr 
                      key={exp.id} 
                      className={`cursor-pointer transition-colors ${exp.rank === 1 ? 'bg-green-50 hover:bg-green-100' : 'hover:bg-blue-50'}`}
                      onClick={() => handleModelClick(exp)}
                      title="Click to view hyperparameters"
                    >
                      <td className={`px-4 py-3 text-sm ${exp.rank === 1 ? 'bg-green-600 text-white font-bold' : 'text-gray-900'}`}>
                        {exp.rank}
                      </td>
                      <td className={`px-4 py-3 text-sm ${exp.best ? 'bg-green-600 text-white font-bold' : 'text-gray-900'}`}>
                        {exp.best}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-900">
                        {formatDate(exp.timestamp)}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-900">
                        {exp.model}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-900">
                        {formatTrainingTime(exp.training_time)}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-900">
                        {formatMAE(exp.mae_train)}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-900">
                        {formatMAE(exp.rmse_train)}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-900">
                        {formatR2Score(exp.r2_train)}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-900">
                        {formatMAE(exp.mae_test)}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-900">
                        {formatMAE(exp.rmse_test)}
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-900">
                        <span className={`font-medium ${getScoreColor(exp.r2_test)}`}>
                          {formatR2Score(exp.r2_test)}
                        </span>
                      </td>
                      <td className={`px-4 py-3 text-sm ${getR2GapColor(exp.r2_gap)}`}>
                        {exp.r2_gap}
                      </td>
                      <td className="px-4 py-3 text-sm">
                        <span className={getDiagnosticColor(exp.r2_gap_diagnostic)}>
                          {exp.r2_gap_diagnostic}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-sm text-gray-900">
                        {exp.n_features}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderPipelineTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg border p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Training Pipeline</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-gray-700 mb-2">Current Model</h4>
            <p className="text-sm text-gray-600">CatBoost Regressor</p>
          </div>
          <div>
            <h4 className="font-medium text-gray-700 mb-2">Features</h4>
            <p className="text-sm text-gray-600">2,885 features</p>
          </div>
        </div>
        <div className="mt-6">
          <button 
            onClick={startTraining}
            disabled={isTraining}
            className="bg-green-500 text-white px-6 py-2 rounded hover:bg-green-600 disabled:bg-gray-400"
          >
            {isTraining ? 'Training...' : 'Start Training'}
          </button>
        </div>
      </div>
    </div>
  );

  const renderHyperparametersPanel = () => {
    if (!selectedModel) return null;

    const topHyperparameters = getTopHyperparametersForModel(selectedModel.model_name);

    return (
      <div className={`fixed left-0 top-0 h-full bg-white shadow-2xl transform transition-transform duration-300 ease-in-out z-50 ${
        showHyperparametersPanel ? 'translate-x-0' : '-translate-x-full'
      }`} style={{ width: '500px' }}>
        
        {/* Header */}
        <div className="bg-blue-500 text-white p-4 flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold">Hyperparameters</h3>
            <p className="text-sm opacity-90">{selectedModel.model_name}</p>
          </div>
          <button
            onClick={() => setShowHyperparametersPanel(false)}
            className="text-white hover:bg-blue-600 p-2 rounded-md transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Content */}
        <div className="p-4 h-full overflow-y-auto">
          
          {/* Model Summary */}
          <div className="bg-gray-50 p-4 rounded-lg mb-4">
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium text-gray-600">Best R² Score:</span>
                <div className={`text-lg font-bold ${getScoreColor(selectedModel.r2_test)}`}>
                  {formatR2Score(selectedModel.r2_test)}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-600">Trial Number:</span>
                <div className="text-lg font-bold text-gray-900">
                  {selectedModel.trial_number || 'Baseline'}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-600">Training Time:</span>
                <div className="text-lg font-bold text-gray-900">
                  {formatTrainingTime(selectedModel.training_time)}
                </div>
              </div>
              <div>
                <span className="font-medium text-gray-600">Status:</span>
                <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                  selectedModel.generalization_status === 'Excellent generalization' ? 'bg-green-100 text-green-800' :
                  selectedModel.generalization_status === 'Good generalization' ? 'bg-blue-100 text-blue-800' :
                  selectedModel.generalization_status === 'Moderate overfitting' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-gray-100 text-gray-800'
                }`}>
                  {selectedModel.generalization_status || 'Unknown'}
                </span>
              </div>
            </div>
          </div>

          {/* Top Hyperparameters */}
          <div className="mb-4">
            <h4 className="text-md font-semibold text-gray-900 mb-3">
              Top 10 Hyperparameter Configurations for {selectedModel.model_name.split(' ')[0]}
            </h4>
            
            {topHyperparameters.length === 0 ? (
              <div className="text-gray-500 text-center py-8">
                No hyperparameter data available for this model type
              </div>
            ) : (
              <div className="space-y-3">
                {topHyperparameters.map((exp, index) => (
                  <div key={exp.id || index} className="border rounded-lg p-3 hover:bg-gray-50">
                    
                    {/* Rank & Performance */}
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <div className={`w-6 h-6 rounded-full flex items-center justify-center text-white text-xs font-bold ${
                          index === 0 ? 'bg-yellow-500' : 
                          index === 1 ? 'bg-gray-400' : 
                          index === 2 ? 'bg-amber-600' : 'bg-gray-300'
                        }`}>
                          {index + 1}
                        </div>
                        <span className="text-sm font-medium">
                          Trial {exp.trial_number || 'N/A'}
                        </span>
                      </div>
                      <div className={`text-sm font-medium ${getScoreColor(exp.r2_test)}`}>
                        R² {formatR2Score(exp.r2_test)}
                      </div>
                    </div>

                    {/* Hyperparameters */}
                    {exp.hyperparameters && (
                      <div className="grid grid-cols-2 gap-2 text-xs">
                        {Object.entries(exp.hyperparameters).slice(0, 8).map(([key, value]) => (
                          <div key={key} className="bg-gray-100 p-2 rounded">
                            <div className="font-medium text-gray-600 truncate">{key}</div>
                            <div className="text-gray-900 truncate">
                              {typeof value === 'number' ? 
                                (value < 0.01 ? value.toExponential(2) : value.toFixed(4)) : 
                                String(value)
                              }
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Current Model's Hyperparameters (if available) */}
          {selectedModel.hyperparameters && (
            <div className="mb-4">
              <h4 className="text-md font-semibold text-gray-900 mb-3">Current Model Hyperparameters</h4>
              <div className="grid grid-cols-1 gap-2">
                {Object.entries(selectedModel.hyperparameters).map(([key, value]) => (
                  <div key={key} className="bg-blue-50 p-3 rounded-lg border border-blue-200">
                    <div className="flex justify-between items-center">
                      <span className="font-medium text-blue-800">{key}</span>
                      <span className="text-blue-900 font-mono">
                        {typeof value === 'number' ? 
                          (value < 0.01 ? value.toExponential(3) : value.toFixed(4)) : 
                          String(value)
                        }
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Launch Optimization Button */}
          <div className="mt-6 pt-4 border-t">
            <button
              onClick={() => {
                window.open('http://localhost:7860', '_blank');
              }}
              className="w-full bg-blue-500 text-white py-3 px-4 rounded-lg hover:bg-blue-600 transition-colors font-medium"
            >
              🚀 Launch New Optimization Study
            </button>
          </div>
        </div>
      </div>
    );
  };
    // Filtrer les expériences pour ne garder que les optimizations (celles avec trial_number)
    const optimizations = experiments.filter(exp => 
      exp.trial_number !== undefined && exp.trial_number !== null
    );
    
    // Trier par performance (R² test)
    const sortedOptimizations = optimizations.sort((a, b) => 
      (b.r2_test || 0) - (a.r2_test || 0)
    );
    
    // Statistiques des optimizations
    const totalOptimizations = optimizations.length;
    const bestR2 = sortedOptimizations.length > 0 ? sortedOptimizations[0].r2_test : 0;
    const avgR2 = totalOptimizations > 0 ? 
      optimizations.reduce((sum, exp) => sum + (exp.r2_test || 0), 0) / totalOptimizations : 0;

    return (
      <div className="space-y-6">
        {/* Header avec statistiques */}
        <div className="bg-white p-6 rounded-lg shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900">CatBoost Hyperparameter Optimization</h2>
            <div className="flex space-x-2">
              <button
                onClick={() => {
                  // Déclencher une nouvelle optimization via l'agent LLM
                  window.open('http://localhost:7860', '_blank');
                }}
                className="bg-blue-500 text-white px-4 py-2 rounded-md hover:bg-blue-600 transition-colors"
              >
                🚀 Launch New Optimization
              </button>
              <button
                onClick={refresh}
                className="bg-gray-500 text-white px-4 py-2 rounded-md hover:bg-gray-600 transition-colors"
              >
                🔄 Refresh
              </button>
            </div>
          </div>

          {/* Statistiques */}
          <div className="grid grid-cols-4 gap-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-gray-900">{totalOptimizations}</div>
              <div className="text-sm text-gray-500">Total Optimizations</div>
            </div>
            <div className="bg-green-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-green-600">{formatR2Score(bestR2)}</div>
              <div className="text-sm text-gray-500">Best R² Score</div>
            </div>
            <div className="bg-blue-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-blue-600">{formatR2Score(avgR2)}</div>
              <div className="text-sm text-gray-500">Average R² Score</div>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg">
              <div className="text-2xl font-bold text-purple-600">
                {sortedOptimizations.length > 0 ? `Trial ${sortedOptimizations[0].trial_number}` : 'N/A'}
              </div>
              <div className="text-sm text-gray-500">Best Trial</div>
            </div>
          </div>
        </div>

        {/* Tableau des meilleures optimizations */}
        <div className="bg-white rounded-lg shadow-sm">
          <div className="px-6 py-4 border-b border-gray-200">
            <h3 className="text-lg font-medium text-gray-900">Best Optimization Results</h3>
            <p className="text-sm text-gray-500 mt-1">
              Top {Math.min(10, sortedOptimizations.length)} optimization trials sorted by R² score
            </p>
          </div>

          {loading ? (
            <div className="p-8 text-center">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto"></div>
              <p className="mt-2 text-gray-600">Loading optimizations...</p>
            </div>
          ) : error ? (
            <div className="p-8 text-center text-red-600">
              <p>Error loading optimizations: {error}</p>
            </div>
          ) : sortedOptimizations.length === 0 ? (
            <div className="p-8 text-center text-gray-500">
              <p>No optimization trials found. Launch your first optimization!</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Rank</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Trial #</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">R² Score</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">R² Gap</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">MAE Test</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Training Time</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Status</th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {sortedOptimizations.slice(0, 10).map((exp, index) => (
                    <tr key={exp.id || index} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="flex items-center">
                          <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${
                            index === 0 ? 'bg-yellow-500' : 
                            index === 1 ? 'bg-gray-400' : 
                            index === 2 ? 'bg-amber-600' : 'bg-gray-300'
                          }`}>
                            {index + 1}
                          </div>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm font-medium text-gray-900">
                          {exp.trial_number || 'N/A'}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className={`text-sm font-medium ${getScoreColor(exp.r2_test)}`}>
                          {formatR2Score(exp.r2_test)}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className={`text-sm ${getR2GapColor(exp.r2_gap)}`}>
                          {exp.r2_gap ? exp.r2_gap.toFixed(4) : 'N/A'}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">
                          {formatMAE(exp.mae_test)}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <div className="text-sm text-gray-900">
                          {formatTrainingTime(exp.training_time)}
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${
                          exp.generalization_status === 'Excellent generalization' ? 'bg-green-100 text-green-800' :
                          exp.generalization_status === 'Good generalization' ? 'bg-blue-100 text-blue-800' :
                          exp.generalization_status === 'Moderate overfitting' ? 'bg-yellow-100 text-yellow-800' :
                          exp.generalization_status === 'Strong overfitting' ? 'bg-red-100 text-red-800' :
                          'bg-gray-100 text-gray-800'
                        }`}>
                          {exp.generalization_status || 'Unknown'}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatDate(exp.timestamp)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Détails des hyperparamètres du meilleur modèle */}
        {sortedOptimizations.length > 0 && (
          <div className="bg-white rounded-lg shadow-sm">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-medium text-gray-900">
                Best Trial Hyperparameters (Trial #{sortedOptimizations[0].trial_number})
              </h3>
              <p className="text-sm text-gray-500 mt-1">
                R² Score: {formatR2Score(sortedOptimizations[0].r2_test)}
              </p>
            </div>
            <div className="p-6">
              {sortedOptimizations[0].hyperparameters ? (
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                  {Object.entries(sortedOptimizations[0].hyperparameters).map(([key, value]) => (
                    <div key={key} className="bg-gray-50 p-3 rounded-lg">
                      <div className="text-xs font-medium text-gray-500 uppercase">{key}</div>
                      <div className="text-sm font-medium text-gray-900 mt-1">
                        {typeof value === 'number' ? value.toFixed(4) : String(value)}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-gray-500">No hyperparameters data available</p>
              )}
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderTabContent = () => {
    switch (activeTab) {
      case 'pipeline': return renderPipelineTab();
      case 'experiments': return renderExperimentsTab();
      case 'deployment': return <div className="text-center py-8">Deployment options coming soon...</div>;
      default: return renderExperimentsTab();
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      
      {/* Hyperparameters Panel */}
      {renderHyperparametersPanel()}
      
      {/* Overlay */}
      {showHyperparametersPanel && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-40"
          onClick={() => setShowHyperparametersPanel(false)}
        />
      )}
      
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Model Training</h1>
          <p className="text-gray-600">Train and evaluate machine learning models with hyperparameter optimization</p>
        </div>

        {/* Onglets */}
        <div className="mb-6">
          <nav className="flex space-x-1 bg-white rounded-lg p-1 shadow-sm">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex-1 py-2 px-4 text-sm font-medium rounded-md transition-colors ${
                  activeTab === tab.id
                    ? 'bg-blue-500 text-white shadow-sm'
                    : 'text-gray-500 hover:text-gray-700 hover:bg-gray-100'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Contenu des onglets */}
        {renderTabContent()}
      </div>
    </div>
  );
};

export default ModelTrainingPage;
