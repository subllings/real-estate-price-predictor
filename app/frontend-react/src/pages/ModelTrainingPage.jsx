import React, { useState } from 'react';
import { useExperiments } from '../hooks/useExperiments';

const ModelTrainingPage = () => {
  const [activeTab, setActiveTab] = useState('experiments');
  const [isTraining, setIsTraining] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [showHyperparametersPanel, setShowHyperparametersPanel] = useState(false);
  const [showGeneralizationModal, setShowGeneralizationModal] = useState(false);
  const [includeTestModels, setIncludeTestModels] = useState(false);
  
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
      if (isNaN(date.getTime())) return 'N/A';
      
      // Format belge avec heure (DD/MM/YYYY HH:MM)
      return date.toLocaleString('fr-BE', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        hour12: false
      });
    } catch (e) {
      return 'N/A';
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
      case 'Excellent generalization': return 'text-green-600 bg-green-50 px-2 py-1 rounded-full text-xs font-medium';
      case 'Good generalization': return 'text-blue-600 bg-blue-50 px-2 py-1 rounded-full text-xs font-medium';
      case 'Moderate overfitting': return 'text-yellow-600 bg-yellow-50 px-2 py-1 rounded-full text-xs font-medium';
      case 'Strong overfitting': return 'text-red-600 bg-red-50 px-2 py-1 rounded-full text-xs font-medium';
      case 'Possible underfitting': return 'text-purple-600 bg-purple-50 px-2 py-1 rounded-full text-xs font-medium';
      
      // Support anciens diagnostics pour compatibilité
      case 'Excellent': return 'text-green-600 bg-green-50 px-2 py-1 rounded-full text-xs font-medium';
      case 'Good': return 'text-blue-600 bg-blue-50 px-2 py-1 rounded-full text-xs font-medium';
      case 'Fair': return 'text-yellow-600 bg-yellow-50 px-2 py-1 rounded-full text-xs font-medium';
      case 'Poor': return 'text-red-600 bg-red-50 px-2 py-1 rounded-full text-xs font-medium';
      
      default: return 'text-gray-600 bg-gray-50 px-2 py-1 rounded-full text-xs font-medium';
    }
  };

  // Nouvelle fonction pour les couleurs des labels de généralisation
  const getGeneralizationLabelColor = (label) => {
    switch (label) {
      case 'Excellent': return 'text-green-800 bg-green-100 px-3 py-1 rounded-full text-xs font-medium inline-block min-w-[70px] text-center';
      case 'Good': return 'text-green-700 bg-green-50 px-3 py-1 rounded-full text-xs font-medium inline-block min-w-[70px] text-center';
      case 'Fair': return 'text-yellow-700 bg-yellow-100 px-3 py-1 rounded-full text-xs font-medium inline-block min-w-[70px] text-center';
      case 'Poor': return 'text-red-700 bg-red-100 px-3 py-1 rounded-full text-xs font-medium inline-block min-w-[70px] text-center';
      default: return 'text-gray-600 bg-gray-50 px-3 py-1 rounded-full text-xs font-medium inline-block min-w-[70px] text-center';
    }
  };

  // Nouvelle fonction pour les couleurs du risque de surapprentissage
  const getOverfittingRiskColor = (risk) => {
    switch (risk) {
      case 'Low': return 'text-green-700 bg-green-100 px-3 py-1 rounded-full text-xs font-medium inline-block min-w-[80px] text-center';
      case 'Moderate': return 'text-blue-700 bg-blue-100 px-3 py-1 rounded-full text-xs font-medium inline-block min-w-[80px] text-center';
      case 'High': return 'text-yellow-700 bg-yellow-100 px-3 py-1 rounded-full text-xs font-medium inline-block min-w-[80px] text-center';
      case 'Very High': return 'text-red-700 bg-red-100 px-3 py-1 rounded-full text-xs font-medium inline-block min-w-[80px] text-center';
      default: return 'text-gray-600 bg-gray-50 px-3 py-1 rounded-full text-xs font-medium inline-block min-w-[80px] text-center';
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

  // Nouvelle fonction pour calculer l'index de généralisation (0-100)
  const computeGeneralizationIndex = (r2_train, r2_test) => {
    if (!r2_train || !r2_test) return 0;
    const r2_gap = r2_train - r2_test;
    const rawScore = 100 - (r2_gap * 1000);
    return Math.max(0, Math.min(100, Math.round(rawScore * 10) / 10)); // Clamp entre 0 et 100
  };

  // Nouvelle fonction pour le label de généralisation basé sur l'index
  const getGeneralizationLabel = (index) => {
    if (index >= 95) return 'Excellent';
    if (index >= 90) return 'Good';
    if (index >= 80) return 'Fair';
    return 'Poor';
  };

  // Nouvelle fonction pour le risque de surapprentissage
  const getOverfittingRisk = (r2_gap) => {
    if (r2_gap < 0.03) return 'Low';
    if (r2_gap < 0.07) return 'Moderate';
    if (r2_gap < 0.12) return 'High';
    return 'Very High';
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
        </div>
      </div>
    );
  };

  const renderGeneralizationModal = () => {
    if (!showGeneralizationModal) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
        <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
          {/* Header */}
          <div className="bg-blue-500 text-white p-6 flex items-center justify-between rounded-t-lg">
            <div>
              <h3 className="text-xl font-semibold">Understanding Generalization Metrics</h3>
              <p className="text-sm opacity-90 mt-1">Learn how to interpret model generalization indicators</p>
            </div>
            <button
              onClick={() => setShowGeneralizationModal(false)}
              className="text-white hover:bg-blue-600 p-2 rounded-md transition-colors"
            >
              ✕
            </button>
          </div>

          {/* Content */}
          <div className="p-6 space-y-6">
            
            {/* Introduction */}
            <div className="bg-blue-50 p-4 rounded-lg border-l-4 border-blue-400">
              <p className="text-gray-700">
                The performance of a machine learning model is not only measured by how well it fits the training data, 
                but more importantly by <strong>how well it generalizes to unseen data</strong>. The following indicators help assess this:
              </p>
            </div>

            {/* R² Gap Section */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-800 flex items-center">
                R² Gap
              </h4>
              <div className="bg-gray-50 p-4 rounded-lg">
                <p className="text-gray-700 mb-2">
                  Measures the difference between R² on the training and test sets.
                </p>
                <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
                  <li>A small gap indicates good generalization (e.g. 0.02 is excellent)</li>
                  <li>A large gap (e.g. 0.14) suggests overfitting</li>
                </ul>
              </div>
            </div>

            {/* Generalization Index Section */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-800 flex items-center">
                Generalization Index (0–100)
              </h4>
              <div className="bg-gray-50 p-4 rounded-lg">
                <p className="text-gray-700 mb-3">
                  A custom score based on the R² Gap. Calculated as:
                </p>
                <div className="bg-white p-3 rounded border font-mono text-sm text-center mb-3">
                  Generalization Index = 100 - (R² Gap × 1000), capped between 0 and 100
                </div>
                <p className="text-sm text-gray-600 mb-4">
                  Closer to 100 = better generalization • Index below 80 = poor generalization
                </p>
                
                {/* Table */}
                <div className="overflow-x-auto">
                  <table className="w-full text-sm border border-gray-200">
                    <thead className="bg-gray-100">
                      <tr>
                        <th className="px-3 py-2 text-left border-r">R² Gap</th>
                        <th className="px-3 py-2 text-left border-r">Gen. Index</th>
                        <th className="px-3 py-2 text-left">Interpretation</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-200">
                      <tr>
                        <td className="px-3 py-2 border-r text-green-600 font-medium">≤ 0.02</td>
                        <td className="px-3 py-2 border-r text-green-600 font-medium">≥ 98</td>
                        <td className="px-3 py-2">Excellent generalization</td>
                      </tr>
                      <tr className="bg-gray-50">
                        <td className="px-3 py-2 border-r text-blue-600 font-medium">~0.05</td>
                        <td className="px-3 py-2 border-r text-blue-600 font-medium">~95</td>
                        <td className="px-3 py-2">Good generalization</td>
                      </tr>
                      <tr>
                        <td className="px-3 py-2 border-r text-yellow-600 font-medium">~0.08</td>
                        <td className="px-3 py-2 border-r text-yellow-600 font-medium">~92</td>
                        <td className="px-3 py-2">Moderate overfitting risk</td>
                      </tr>
                      <tr className="bg-gray-50">
                        <td className="px-3 py-2 border-r text-red-600 font-medium">&gt; 0.12</td>
                        <td className="px-3 py-2 border-r text-red-600 font-medium">&lt; 90</td>
                        <td className="px-3 py-2">Poor generalization / overfitting</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Generalization Label Section */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-800 flex items-center">
                Generalization Label
              </h4>
              <div className="bg-gray-50 p-4 rounded-lg">
                <p className="text-gray-700 mb-3">
                  A categorical label based on the Generalization Index:
                </p>
                <div className="grid grid-cols-2 gap-3">
                  <div className="flex items-center justify-between bg-white p-2 rounded border">
                    <span className="text-sm font-medium">≥ 95</span>
                    <span className="text-green-800 bg-green-100 px-2 py-1 rounded-full text-xs font-medium">Excellent</span>
                  </div>
                  <div className="flex items-center justify-between bg-white p-2 rounded border">
                    <span className="text-sm font-medium">90–94</span>
                    <span className="text-green-700 bg-green-50 px-2 py-1 rounded-full text-xs font-medium">Good</span>
                  </div>
                  <div className="flex items-center justify-between bg-white p-2 rounded border">
                    <span className="text-sm font-medium">80–89</span>
                    <span className="text-yellow-700 bg-yellow-100 px-2 py-1 rounded-full text-xs font-medium">Fair</span>
                  </div>
                  <div className="flex items-center justify-between bg-white p-2 rounded border">
                    <span className="text-sm font-medium">&lt; 80</span>
                    <span className="text-red-700 bg-red-100 px-2 py-1 rounded-full text-xs font-medium">Poor</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Overfitting Risk Section */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-800 flex items-center">
                Overfitting Risk
              </h4>
              <div className="bg-gray-50 p-4 rounded-lg">
                <p className="text-gray-700 mb-3">
                  Describes how likely the model is to have overfitted the training data. 
                  It is based on both the R² gap and error spread (MAE/RMSE train vs test).
                </p>
                <div className="space-y-2">
                  <div className="flex items-center justify-between bg-white p-3 rounded border">
                    <span className="text-green-700 bg-green-100 px-2 py-1 rounded-full text-xs font-medium">Low</span>
                    <span className="text-sm text-gray-600">Minor R² gap, consistent errors</span>
                  </div>
                  <div className="flex items-center justify-between bg-white p-3 rounded border">
                    <span className="text-blue-700 bg-blue-100 px-2 py-1 rounded-full text-xs font-medium">Moderate</span>
                    <span className="text-sm text-gray-600">Acceptable gap, but needs monitoring</span>
                  </div>
                  <div className="flex items-center justify-between bg-white p-3 rounded border">
                    <span className="text-yellow-700 bg-yellow-100 px-2 py-1 rounded-full text-xs font-medium">High</span>
                    <span className="text-sm text-gray-600">Model may not generalize well</span>
                  </div>
                  <div className="flex items-center justify-between bg-white p-3 rounded border">
                    <span className="text-red-700 bg-red-100 px-2 py-1 rounded-full text-xs font-medium">Very High</span>
                    <span className="text-sm text-gray-600">Strong signs of overfitting</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Why It Matters */}
            <div className="bg-amber-50 p-4 rounded-lg border-l-4 border-amber-400">
              <h4 className="text-lg font-semibold text-amber-800 mb-2">Why It Matters</h4>
              <ul className="list-disc list-inside text-amber-700 space-y-1">
                <li>A model that performs well on training data but poorly on unseen data is likely overfitting and will fail in production.</li>
                <li>These metrics help you identify robust, stable, and deployable models.</li>
              </ul>
            </div>

            {/* Close button */}
            <div className="text-center pt-4">
              <button
                onClick={() => setShowGeneralizationModal(false)}
                className="bg-blue-500 text-white px-6 py-2 rounded hover:bg-blue-600 transition-colors"
              >
                Got it!
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  };

  const renderExperimentsTab = () => {
    // Filtrer les expériences selon la checkbox
    const filteredExperiments = includeTestModels 
      ? experiments 
      : experiments.filter(exp => !exp.model_name?.includes('[TEST]'));

    // Recalculer les statistiques basées sur les données filtrées
    const filteredSummary = filteredExperiments.length > 0 ? {
      total_experiments: filteredExperiments.length,
      best_r2_score: Math.max(...filteredExperiments.map(exp => exp.r2_test || 0)),
      average_r2_score: filteredExperiments.reduce((sum, exp) => sum + (exp.r2_test || 0), 0) / filteredExperiments.length,
      average_r2_gap: filteredExperiments.reduce((sum, exp) => sum + ((exp.r2_train || 0) - (exp.r2_test || 0)), 0) / filteredExperiments.length,
      latest_experiment: filteredExperiments.reduce((latest, exp) => 
        (!latest.timestamp || new Date(exp.timestamp || 0) > new Date(latest.timestamp || 0)) ? exp : latest, {}),
      best_generalization: filteredExperiments.reduce((best, exp) => {
        const gap = (exp.r2_train || 0) - (exp.r2_test || 0);
        return (!best || gap < ((best.r2_train || 0) - (best.r2_test || 0))) ? exp : best;
      }, null)
    } : summary;

    // Traitement des données pour le format du tableau avec métriques structurées
    const processedExperiments = filteredExperiments.map((exp, index) => {
      // Trier par R² test décroissant
      const sortedExperiments = [...filteredExperiments].sort((a, b) => (b.r2_test || 0) - (a.r2_test || 0));
      const rank = sortedExperiments.findIndex(e => e.id === exp.id) + 1;
      
      return {
        ...exp,
        rank,
        best: rank === 1 ? '✓' : '',
        model: exp.model_name || 'CatBoost CV (All Features)',
        r2_gap: exp.r2_gap ? exp.r2_gap.toFixed(6) : ((exp.r2_train || 0) - (exp.r2_test || 0)).toFixed(6),
        r2_gap_diagnostic: getGeneralizationDiagnostic(exp.r2_train, exp.r2_test),
        generalization_index: computeGeneralizationIndex(exp.r2_train, exp.r2_test),
        generalization_label: getGeneralizationLabel(computeGeneralizationIndex(exp.r2_train, exp.r2_test)),
        overfitting_risk: getOverfittingRisk((exp.r2_train || 0) - (exp.r2_test || 0)),
        generalization_status: exp.generalization_status || getGeneralizationDiagnostic(exp.r2_train, exp.r2_test),
        n_features: exp.feature_count || 2885
      };
    }).sort((a, b) => a.rank - b.rank);

    return (
      <div className="space-y-6">
        
        {/* Header avec statistiques enrichies */}
        <div className="bg-white rounded-lg border p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-800">Experiment History</h3>
            <div className="flex space-x-2">
              <label className="flex items-center space-x-2 text-sm text-gray-700 cursor-pointer">
                <input
                  type="checkbox"
                  checked={includeTestModels}
                  onChange={(e) => setIncludeTestModels(e.target.checked)}
                  className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                />
                <span>Include test models</span>
              </label>
              <button 
                onClick={() => setShowGeneralizationModal(true)}
                className="bg-gray-100 text-gray-700 px-3 py-2 rounded hover:bg-gray-200 transition-colors flex items-center text-sm"
                title="Understand Generalization Metrics"
              >
                Help
              </button>
              <button 
                onClick={refresh}
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-colors"
              >
                Refresh
              </button>
            </div>
          </div>
          
          {/* Statistiques de résumé enrichies */}
          {filteredSummary && (
            <div className="grid grid-cols-5 gap-4 mb-6">
              <div className="text-center">
                <div className="text-2xl font-bold text-blue-600">{filteredSummary.total_experiments}</div>
                <div className="text-sm text-gray-600">Total Experiments</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-green-600">{filteredSummary.best_r2_score?.toFixed(3)}</div>
                <div className="text-sm text-gray-600">Best R² Score</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">{filteredSummary.average_r2_score?.toFixed(3)}</div>
                <div className="text-sm text-gray-600">Average R² Score</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-purple-600">{filteredSummary.average_r2_gap?.toFixed(4)}</div>
                <div className="text-sm text-gray-600">Avg R² Gap</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-gray-600">
                  {filteredSummary.latest_experiment?.timestamp ? 
                    new Date(filteredSummary.latest_experiment.timestamp).toLocaleString('fr-BE', {
                      year: 'numeric',
                      month: '2-digit',
                      day: '2-digit',
                      hour: '2-digit',
                      minute: '2-digit',
                      hour12: false
                    }) : 
                    'N/A'
                  }
                </div>
                <div className="text-sm text-gray-600">Latest</div>
              </div>
            </div>
          )}
          
          {/* Indicateur de meilleure généralisation */}
          {filteredSummary?.best_generalization && (
            <div className="bg-indigo-50 rounded-lg p-4 mb-6">
              <h4 className="font-medium text-indigo-800 mb-2">Best Generalization</h4>
              <div className="flex items-center justify-between">
                <span className="text-sm text-indigo-600">
                  R² Gap: {((filteredSummary.best_generalization.r2_train || 0) - (filteredSummary.best_generalization.r2_test || 0)).toFixed(6)}
                </span>
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  getGeneralizationDiagnostic(filteredSummary.best_generalization.r2_train, filteredSummary.best_generalization.r2_test) === 'Excellent generalization' ? 'bg-green-100 text-green-800' :
                  getGeneralizationDiagnostic(filteredSummary.best_generalization.r2_train, filteredSummary.best_generalization.r2_test) === 'Good generalization' ? 'bg-blue-100 text-blue-800' :
                  getGeneralizationDiagnostic(filteredSummary.best_generalization.r2_train, filteredSummary.best_generalization.r2_test) === 'Moderate overfitting' ? 'bg-yellow-100 text-yellow-800' :
                  'bg-red-100 text-red-800'
                }`}>
                  {getGeneralizationDiagnostic(filteredSummary.best_generalization.r2_train, filteredSummary.best_generalization.r2_test)}
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
          
          {!loading && !error && filteredExperiments.length === 0 && (
            <div className="bg-gray-50 rounded-lg p-8 text-center">
              <p className="text-gray-500">
                {experiments.length === 0 ? 'No experiments found' : 'No experiments found with current filter'}
              </p>
            </div>
          )}
        </div>

        {/* Tableau des expériences - Format exact selon les métriques structurées */}
        {!loading && !error && filteredExperiments.length > 0 && (
          <div className="bg-white rounded-lg border overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[1400px] text-xs">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-1 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style={{width: '40px'}}>Rank</th>
                    <th className="px-1 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style={{width: '50px'}}>Best</th>
                    <th className="px-2 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style={{width: '160px'}}>Timestamp</th>
                    <th className="px-2 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style={{width: '200px'}}>Model</th>
                    <th className="px-1 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style={{width: '100px'}}>Training Time</th>
                    <th className="px-1 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style={{width: '90px'}}>MAE Train</th>
                    <th className="px-1 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style={{width: '90px'}}>RMSE Train</th>
                    <th className="px-1 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style={{width: '90px'}}>R² Train</th>
                    <th className="px-1 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style={{width: '90px'}}>MAE Test</th>
                    <th className="px-1 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style={{width: '90px'}}>RMSE Test</th>
                    <th className="px-1 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style={{width: '90px'}}>R² Test</th>
                    <th className="px-1 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style={{width: '70px'}}>R² Gap</th>
                    <th className="px-2 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style={{width: '120px'}}>R² Gap Diagnostic</th>
                    <th className="px-1 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style={{width: '90px'}}>Gen. Index</th>
                    <th className="px-2 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style={{width: '100px'}}>Generalization</th>
                    <th className="px-2 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style={{width: '110px'}}>Overfitting Risk</th>
                    <th className="px-1 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style={{width: '80px'}}>N Features</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {processedExperiments.map((exp, index) => (
                    <tr 
                      key={exp.id} 
                      className={`cursor-pointer transition-colors ${exp.rank === 1 ? 'bg-green-50 hover:bg-green-100' : 'hover:bg-blue-50'}`}
                      onClick={() => handleModelClick(exp)}
                      title="Click to view hyperparameters - Panel stays open for multiple selections"
                    >
                      <td className={`px-1 py-3 text-xs text-center ${exp.rank === 1 ? 'bg-green-600 text-white font-bold' : 'text-gray-900'}`} style={{width: '40px'}}>
                        {exp.rank}
                      </td>
                      <td className={`px-1 py-3 text-xs text-center ${exp.best ? 'bg-green-600 text-white font-bold' : 'text-gray-900'}`} style={{width: '50px'}}>
                        {exp.best}
                      </td>
                      <td className="px-2 py-3 text-xs text-center text-gray-900" style={{width: '160px', whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden'}}>
                        {formatDate(exp.timestamp)}
                      </td>
                      <td className="px-2 py-3 text-xs text-center text-gray-900" style={{width: '200px', whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden'}} title={exp.model}>
                        {exp.model}
                      </td>
                      <td className="px-1 py-3 text-xs text-center text-gray-900" style={{width: '100px'}}>
                        {formatTrainingTime(exp.training_time)}
                      </td>
                      <td className="px-1 py-3 text-xs text-center text-gray-900" style={{width: '90px'}}>
                        {formatMAE(exp.mae_train)}
                      </td>
                      <td className="px-1 py-3 text-xs text-center text-gray-900" style={{width: '90px'}}>
                        {formatMAE(exp.rmse_train)}
                      </td>
                      <td className="px-1 py-3 text-xs text-center text-gray-900" style={{width: '90px'}}>
                        {formatR2Score(exp.r2_train)}
                      </td>
                      <td className="px-1 py-3 text-xs text-center text-gray-900" style={{width: '90px'}}>
                        {formatMAE(exp.mae_test)}
                      </td>
                      <td className="px-1 py-3 text-xs text-center text-gray-900" style={{width: '90px'}}>
                        {formatMAE(exp.rmse_test)}
                      </td>
                      <td className="px-1 py-3 text-xs text-center text-gray-900" style={{width: '90px'}}>
                        <span className={`font-medium ${getScoreColor(exp.r2_test)}`}>
                          {formatR2Score(exp.r2_test)}
                        </span>
                      </td>
                      <td className={`px-1 py-3 text-xs text-center ${getR2GapColor(exp.r2_gap)}`} style={{width: '70px'}}>
                        {exp.r2_gap}
                      </td>
                      <td className="px-2 py-3 text-xs text-center" style={{width: '120px', whiteSpace: 'nowrap', textOverflow: 'ellipsis', overflow: 'hidden'}}>
                        <span className={getDiagnosticColor(exp.r2_gap_diagnostic)}>
                          {exp.r2_gap_diagnostic}
                        </span>
                      </td>
                      <td className="px-1 py-3 text-xs text-center" style={{width: '90px'}}>
                        <div className="flex items-center justify-center">
                          <div className="bg-gray-200 rounded-full h-2 w-8 mr-1">
                            <div 
                              className={`h-2 rounded-full ${
                                exp.generalization_index >= 95 ? 'bg-green-600' :
                                exp.generalization_index >= 90 ? 'bg-green-400' :
                                exp.generalization_index >= 80 ? 'bg-yellow-400' :
                                'bg-red-400'
                              }`}
                              style={{ width: `${exp.generalization_index}%` }}
                            ></div>
                          </div>
                          <span className="text-xs font-medium">{exp.generalization_index}</span>
                        </div>
                      </td>
                      <td className="px-2 py-3 text-xs text-center" style={{width: '100px'}}>
                        <span className={getGeneralizationLabelColor(exp.generalization_label)}>
                          {exp.generalization_label}
                        </span>
                      </td>
                      <td className="px-2 py-3 text-xs text-center" style={{width: '110px'}}>
                        <span className={getOverfittingRiskColor(exp.overfitting_risk)}>
                          {exp.overfitting_risk}
                        </span>
                      </td>
                      <td className="px-1 py-3 text-xs text-center text-gray-900" style={{width: '80px'}}>
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
      
      {/* Hyperparameters Panel - SANS overlay sombre */}
      {renderHyperparametersPanel()}
      
      {/* Generalization Modal */}
      {renderGeneralizationModal()}
      
      <div className={`max-w-none mx-auto px-4 py-8 transition-all duration-300 ${
        showHyperparametersPanel ? 'ml-[500px]' : ''
      }`}>
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Model Training</h1>
          <p className="text-gray-600">Train and evaluate machine learning models</p>
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
