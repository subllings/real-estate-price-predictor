import React, { useState } from 'react';
import { useExperiments } from '../hooks/useExperiments';
import useTrainingJobs from '../hooks/useTrainingJobs';
import TrainingJobCard from '../components/TrainingJobCard';
import { tunerApi } from '../services/api';

const ModelTrainingPage = () => {
  const [activeTab, setActiveTab] = useState('experiments');
  const [isTraining, setIsTraining] = useState(false);
  const [selectedModel, setSelectedModel] = useState(null);
  const [showHyperparametersPanel, setShowHyperparametersPanel] = useState(false);
  const [showGeneralizationModal, setShowGeneralizationModal] = useState(false);
  const [showNewTrainingModal, setShowNewTrainingModal] = useState(false);
  const [includeTestModels, setIncludeTestModels] = useState(false);
  const [showHelpModal, setShowHelpModal] = useState(false);
  const [newTrainingFormData, setNewTrainingFormData] = useState({
    model_type: 'catboost',
    target_r2: 0.85,
    max_trials: 50,
    compute_target: 'local',
    machine_preference: 'auto',
    termination_type: 'max_trials',
    max_duration_hours: 2,
    end_time: '07:00'
  });
  
  const { 
    experiments, 
    summary, 
    loading, 
    error, 
    refresh 
  } = useExperiments();

  const {
    trainingJobs,
    loading: trainingLoading,
    error: trainingError,
    refresh: refreshTraining,
    startNewTraining,
    stopTrainingJob
  } = useTrainingJobs();

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
    
    // Appliquer la même logique de ranking que dans le tableau principal
    const penalty_map = {
      "Low": 0.0,
      "Moderate": 0.02,
      "High": 0.04,
      "Very High": 0.07
    };

    const processedExperiments = modelExperiments
      .filter(exp => exp.hyperparameters) // Seulement ceux avec hyperparamètres
      .map((exp) => {
        const r2_test = exp.r2_test || 0;
        const r2_train = exp.r2_train || 0;
        const r2_gap = exp.r2_gap !== undefined ? exp.r2_gap : (r2_train - r2_test);
        const generalization_index = exp.generalization_index !== undefined ? exp.generalization_index : computeGeneralizationIndex(r2_train, r2_test);
        const overfitting_risk = exp.overfitting_risk || getOverfittingRisk(r2_gap);
        const overfitting_penalty = penalty_map[overfitting_risk] ?? 0.07;
        
        // Bonus production-readiness
        let production_bonus = 0;
        if (r2_test >= 0.85) production_bonus += 0.1;
        if (generalization_index >= 90) production_bonus += 0.05;
        if (["Low", "Moderate"].includes(overfitting_risk)) production_bonus += 0.03;
        
        const ranking_score = r2_test - (r2_gap * 2) + (generalization_index / 100 * 0.2) - overfitting_penalty + production_bonus;
        
        return {
          ...exp,
          ranking_score: ranking_score,
          generalization_index: generalization_index,
          overfitting_risk: overfitting_risk
        };
      })
      .sort((a, b) => b.ranking_score - a.ranking_score); // Trier par ranking score au lieu de R² seulement
    
    return processedExperiments; // Retourner TOUS les trials, pas seulement les 10 premiers
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
      // Nouveaux diagnostics simplifiés
      case 'Excellent': return 'text-green-600 bg-green-50 px-2 py-1 rounded-full text-xs font-medium';
      case 'Moderate': return 'text-yellow-600 bg-yellow-50 px-2 py-1 rounded-full text-xs font-medium';
      case 'Strong overfitting': return 'text-red-600 bg-red-50 px-2 py-1 rounded-full text-xs font-medium';
      case 'Possible underfitting': return 'text-purple-600 bg-purple-50 px-2 py-1 rounded-full text-xs font-medium';
      
      // Support anciens diagnostics pour compatibilité
      case 'Excellent generalization': return 'text-green-600 bg-green-50 px-2 py-1 rounded-full text-xs font-medium';
      case 'Good generalization': return 'text-blue-600 bg-blue-50 px-2 py-1 rounded-full text-xs font-medium';
      case 'Moderate overfitting': return 'text-yellow-600 bg-yellow-50 px-2 py-1 rounded-full text-xs font-medium';
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
      case 'Moderate': return 'text-yellow-700 bg-yellow-100 px-3 py-1 rounded-full text-xs font-medium inline-block min-w-[70px] text-center';
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
    
    // Logique selon les instructions : ≤ 0.05 → Excellent, ≤ 0.10 → Moderate, > 0.10 → Strong overfitting
    if (r2_gap < 0) {
      return 'Possible underfitting';
    } else if (r2_gap <= 0.05) {
      return 'Excellent';
    } else if (r2_gap <= 0.10) {
      return 'Moderate';
    } else {
      return 'Strong overfitting';
    }
  };

  // Nouvelle fonction pour calculer l'index de généralisation (0-100)
  const computeGeneralizationIndex = (r2_train, r2_test) => {
    if (!r2_train || !r2_test) return 0;
    const r2_gap = r2_train - r2_test;
    const rawScore = 100 - (r2_gap * 1000);
    return Math.max(0, Math.min(100, Math.round(rawScore))); // Retourner entier entre 0 et 100
  };

  // Nouvelle fonction pour le label de généralisation basé sur l'index
  const getGeneralizationLabel = (index) => {
    if (index >= 95) return 'Excellent';
    if (index >= 90) return 'Good';
    if (index >= 80) return 'Moderate';
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
                <span className="font-medium text-gray-600">Ranking Score:</span>
                <div className="text-lg font-bold text-blue-600">
                  {selectedModel.ranking_score ? selectedModel.ranking_score.toFixed(3) : 'N/A'}
                </div>
              </div>
            </div>
          </div>

          {/* All Trials */}
          <div className="mb-4">
            <h4 className="text-md font-semibold text-gray-900 mb-3">
              All Trials for {selectedModel.model_name.split(' ')[0]} (Ranked by Production-Readiness)
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
                      <div className="flex flex-col items-end text-xs">
                        <div className={`font-medium ${getScoreColor(exp.r2_test)}`}>
                          R² {formatR2Score(exp.r2_test)}
                        </div>
                        <div className="text-blue-600 font-medium">
                          Rank: {exp.ranking_score ? exp.ranking_score.toFixed(3) : 'N/A'}
                        </div>
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
                        <td className="px-3 py-2 border-r text-red-600 font-medium">{'>'} 0.12</td>
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

  const renderHelpModal = () => {
    if (!showHelpModal) return null;

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
        <div className="bg-white rounded-lg max-w-5xl w-full max-h-[90vh] overflow-y-auto">
          {/* Header */}
          <div className="bg-blue-500 text-white p-6 flex items-center justify-between rounded-t-lg">
            <div>
              <h3 className="text-xl font-semibold">ML Dashboard - Metrics Interpretation</h3>
              <p className="text-sm opacity-90 mt-1">Understanding metrics, ranking logic, and production-readiness indicators</p>
            </div>
            <button
              onClick={() => setShowHelpModal(false)}
              className="text-white hover:bg-blue-600 p-2 rounded-md transition-colors"
            >
              ✕
            </button>
          </div>

          {/* Content */}
          <div className="p-6 space-y-6">
            
            {/* Overview */}
            <div className="bg-blue-50 p-4 rounded-lg border-l-4 border-blue-400">
              <h4 className="text-lg font-semibold text-blue-800 mb-2">Smart Ranking System</h4>
              <p className="text-blue-700">
                Models are automatically ranked based on <strong>production-readiness</strong>, not just performance. 
                The ranking score prioritizes models that generalize well and avoid overfitting.
              </p>
            </div>

            {/* Core Metrics */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              
              {/* R² Scores */}
              <div className="space-y-4">
                <h4 className="text-lg font-semibold text-gray-800">R² Scores (Coefficient of Determination)</h4>
                <div className="bg-gray-50 p-4 rounded-lg space-y-3">
                  <div>
                    <p className="font-medium text-gray-800">R² Train</p>
                    <p className="text-sm text-gray-600">How well the model fits the training data (0-1, higher = better)</p>
                  </div>
                  <div>
                    <p className="font-medium text-gray-800">R² Test <span className="text-green-600">(≥ 0.85 = Production Ready)</span></p>
                    <p className="text-sm text-gray-600">How well the model performs on unseen test data</p>
                  </div>
                </div>
              </div>

              {/* Error Metrics */}
              <div className="space-y-4">
                <h4 className="text-lg font-semibold text-gray-800">Error Metrics</h4>
                <div className="bg-gray-50 p-4 rounded-lg space-y-3">
                  <div>
                    <p className="font-medium text-gray-800">MAE (Mean Absolute Error)</p>
                    <p className="text-sm text-gray-600">Average prediction error in euros</p>
                  </div>
                  <div>
                    <p className="font-medium text-gray-800">RMSE (Root Mean Square Error)</p>
                    <p className="text-sm text-gray-600">Penalizes larger errors more heavily than MAE</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Generalization Metrics */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-800">Generalization & Overfitting Assessment</h4>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="font-medium text-gray-800 mb-2">R² Gap <span className="text-red-600">({'>'}0.12 = Overfitting)</span></p>
                  <p className="text-sm text-gray-600 mb-3">Difference between R² Train and R² Test</p>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span>≤ 0.05</span>
                      <span className="text-green-600 font-medium">Excellent</span>
                    </div>
                    <div className="flex justify-between">
                      <span>≤ 0.10</span>
                      <span className="text-yellow-600 font-medium">Moderate</span>
                    </div>
                    <div className="flex justify-between">
                      <span>{'>'} 0.10</span>
                      <span className="text-red-600 font-medium">Strong overfitting</span>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 p-4 rounded-lg">
                  <p className="font-medium text-gray-800 mb-2">Generalization Index (0-100)</p>
                  <p className="text-sm text-gray-600 mb-2">Formula: 100 - (R² Gap × 1000)</p>
                  <div className="space-y-1 text-xs">
                    <div className="flex justify-between">
                      <span>≥ 95</span>
                      <span className="text-green-600 font-medium">Excellent</span>
                    </div>
                    <div className="flex justify-between">
                      <span>≥ 90</span>
                      <span className="text-green-600 font-medium">Good</span>
                    </div>
                    <div className="flex justify-between">
                      <span>≥ 80</span>
                      <span className="text-yellow-600 font-medium">Moderate</span>
                    </div>
                    <div className="flex justify-between">
                      <span>&lt; 80</span>
                      <span className="text-red-600 font-medium">Poor</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Overfitting Risk */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-800">Overfitting Risk Assessment</h4>
              <div className="bg-gray-50 p-4 rounded-lg">
                <p className="text-sm text-gray-600 mb-3">
                  Categorical assessment based on R² Gap and error consistency between train/test sets
                </p>
                <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
                  <div className="text-center p-2 bg-green-100 rounded">
                    <div className="text-green-700 font-medium text-sm">Low</div>
                    <div className="text-xs text-green-600">Production Ready</div>
                  </div>
                  <div className="text-center p-2 bg-blue-100 rounded">
                    <div className="text-blue-700 font-medium text-sm">Moderate</div>
                    <div className="text-xs text-blue-600">Acceptable</div>
                  </div>
                  <div className="text-center p-2 bg-yellow-100 rounded">
                    <div className="text-yellow-700 font-medium text-sm">High</div>
                    <div className="text-xs text-yellow-600">Risky</div>
                  </div>
                  <div className="text-center p-2 bg-red-100 rounded">
                    <div className="text-red-700 font-medium text-sm">Very High</div>
                    <div className="text-xs text-red-600">Avoid</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Ranking Formula */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-800">Ranking Score Formula</h4>
              <div className="bg-amber-50 p-4 rounded-lg border-l-4 border-amber-400">
                <div className="font-mono text-sm text-center mb-4 bg-white p-3 rounded border">
                  Ranking Score = R² Test - (R² Gap × 2) + (Gen. Index / 100 × 0.2) - Overfitting Penalty + Production Bonus
                </div>
                
                {/* Detailed explanation of each component */}
                <div className="space-y-4">
                  <div className="bg-white p-4 rounded border">
                    <h5 className="font-semibold text-amber-800 mb-2">Formula Components:</h5>
                    
                    <div className="space-y-3 text-sm">
                      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                        
                        {/* R² Test */}
                        <div className="bg-green-50 p-3 rounded">
                          <div className="font-medium text-green-800">R² Test</div>
                          <div className="text-green-700 text-xs mt-1">
                            <strong>Weight:</strong> +1.0 (score base)<br/>
                            <strong>Role:</strong> Performance on unseen data<br/>
                            <strong>Example:</strong> R² = 0.856 → +0.856 points
                          </div>
                        </div>

                        {/* R² Gap Penalty */}
                        <div className="bg-red-50 p-3 rounded">
                          <div className="font-medium text-red-800">R² Gap × 2</div>
                          <div className="text-red-700 text-xs mt-1">
                            <strong>Weight:</strong> -2.0 (strong penalty)<br/>
                            <strong>Role:</strong> Penalizes overfitting<br/>
                            <strong>Example:</strong> Gap = 0.045 → -0.090 points
                          </div>
                        </div>

                        {/* Generalization Index */}
                        <div className="bg-blue-50 p-3 rounded">
                          <div className="font-medium text-blue-800">Gen. Index / 100 × 0.2</div>
                          <div className="text-blue-700 text-xs mt-1">
                            <strong>Weight:</strong> +0.2 max (generalization bonus)<br/>
                            <strong>Role:</strong> Rewards good generalization<br/>
                            <strong>Example:</strong> Index = 95 → +0.190 points
                          </div>
                        </div>

                        {/* Overfitting Penalty */}
                        <div className="bg-orange-50 p-3 rounded">
                          <div className="font-medium text-orange-800">Overfitting Penalty</div>
                          <div className="text-orange-700 text-xs mt-1">
                            <strong>Weight:</strong> -0.0 to -0.07<br/>
                            <strong>Role:</strong> Penalizes by risk level<br/>
                            <strong>Example:</strong> Risk "Low" → -0.0 points
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Complete calculation example */}
                  <div className="bg-white p-4 rounded border">
                    <h5 className="font-semibold text-amber-800 mb-2">Complete Calculation Example:</h5>
                    <div className="bg-gray-50 p-3 rounded font-mono text-xs">
                      <div className="mb-2"><strong>Input data:</strong></div>
                      <div>• R² Test = 0.856</div>
                      <div>• R² Gap = 0.045</div>
                      <div>• Gen. Index = 95</div>
                      <div>• Overfitting Risk = "Low"</div>
                      <div className="mt-3 mb-2"><strong>Calculation:</strong></div>
                      <div>0.856 - (0.045 × 2) + (95/100 × 0.2) - 0.0 + 0.18</div>
                      <div>= 0.856 - 0.090 + 0.190 - 0.0 + 0.18</div>
                      <div className="text-green-600 font-bold mt-2">= 1.136 (Ranking Score)</div>
                    </div>
                  </div>

                  {/* Detailed Production Bonuses */}
                  <div className="bg-white p-4 rounded border">
                    <h5 className="font-semibold text-amber-800 mb-2">Production Bonuses Detailed:</h5>
                    <div className="space-y-2 text-sm">
                      <div className="flex items-center justify-between bg-green-50 p-2 rounded">
                        <span className="font-medium">R² Excellence (≥ 0.85)</span>
                        <span className="text-green-600 font-bold">+0.1 points</span>
                      </div>
                      <div className="flex items-center justify-between bg-blue-50 p-2 rounded">
                        <span className="font-medium">Good Generalization (≥ 90)</span>
                        <span className="text-blue-600 font-bold">+0.05 points</span>
                      </div>
                      <div className="flex items-center justify-between bg-purple-50 p-2 rounded">
                        <span className="font-medium">Low Overfitting (Low/Moderate)</span>
                        <span className="text-purple-600 font-bold">+0.03 points</span>
                      </div>
                      <div className="text-xs text-gray-600 mt-2 p-2 bg-gray-50 rounded">
                        <strong>Note:</strong> Bonuses are cumulative! A "perfect" model can get +0.18 bonus points.
                      </div>
                    </div>
                  </div>

                  {/* Score interpretation */}
                  <div className="bg-white p-4 rounded border">
                    <h5 className="font-semibold text-amber-800 mb-2">Score Interpretation:</h5>
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 bg-green-100 p-2 rounded">
                          <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                          <span className="text-sm"><strong>&gt; 1.0:</strong> Excellence</span>
                        </div>
                        <div className="flex items-center gap-2 bg-blue-100 p-2 rounded">
                          <div className="w-3 h-3 bg-blue-500 rounded-full"></div>
                          <span className="text-sm"><strong>0.8 - 1.0:</strong> Very Good</span>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <div className="flex items-center gap-2 bg-yellow-100 p-2 rounded">
                          <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                          <span className="text-sm"><strong>0.6 - 0.8:</strong> Acceptable</span>
                        </div>
                        <div className="flex items-center gap-2 bg-red-100 p-2 rounded">
                          <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                          <span className="text-sm"><strong>&lt; 0.6:</strong> Avoid</span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Color Coding */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-800">Color Coding in Table</h4>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                <div className="bg-gray-50 p-3 rounded-lg">
                  <p className="font-medium text-gray-800 mb-2">R² Test Colors</p>
                  <div className="space-y-1 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 bg-green-500 rounded"></div>
                      <span>≥ 0.85 (Production Ready)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 bg-gray-500 rounded"></div>
                      <span>&lt; 0.85 (Not Ready)</span>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 p-3 rounded-lg">
                  <p className="font-medium text-gray-800 mb-2">R² Gap Colors</p>
                  <div className="space-y-1 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 bg-red-500 rounded"></div>
                      <span>{'>'} 0.12 (Overfitting)</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 bg-gray-500 rounded"></div>
                      <span>≤ 0.12 (Acceptable)</span>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 p-3 rounded-lg">
                  <p className="font-medium text-gray-800 mb-2">Overfitting Risk</p>
                  <div className="space-y-1 text-sm">
                    <div className="flex items-center gap-2">
                      <div className="w-4 h-4 bg-red-500 rounded"></div>
                      <span>Very High (Red badge)</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Close Button */}
            <div className="text-center pt-4">
              <button
                onClick={() => setShowHelpModal(false)}
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

  const renderNewTrainingModal = () => {
    if (!showNewTrainingModal) return null;

    const availableComputeTargets = [
      { value: 'local', label: 'Local Machine', description: 'Run on current machine' },
      { value: 'azure-ml', label: 'Azure Machine Learning', description: 'Use Azure ML compute cluster' },
      { value: 'desktop-cluster', label: 'Desktop Cluster', description: 'Distributed across desktop machines' },
      { value: 'laptop-cluster', label: 'Laptop Cluster', description: 'Distributed across laptop machines' }
    ];

    const availableMachines = [
      { value: 'auto', label: 'Auto-select', description: 'Let the system choose the best available machine' },
      { value: 'LAPTOP-DEV-01', label: 'LAPTOP-DEV-01', description: 'Development laptop' },
      { value: 'DESKTOP-MAIN', label: 'DESKTOP-MAIN', description: 'Main desktop workstation' },
      { value: 'AZURE-COMPUTE', label: 'AZURE-COMPUTE', description: 'Azure compute instance' }
    ];

    const handleSubmit = async (e) => {
      e.preventDefault();
      
      try {
        // Préparer la configuration pour le tuner agent
        const tunerConfig = {
          model_type: newTrainingFormData.model_type.toLowerCase().replace('+optuna', ''),
          termination_type: newTrainingFormData.termination_type
        };
        
        // Ajouter les paramètres selon le type de terminaison
        switch (newTrainingFormData.termination_type) {
          case 'duration':
            tunerConfig.duration_hours = parseFloat(newTrainingFormData.max_duration_hours);
            break;
          case 'end_time':
            tunerConfig.end_time = newTrainingFormData.end_time;
            // Extraire hour et minute depuis end_time si format HH:MM
            const [hour, minute] = newTrainingFormData.end_time.split(':');
            if (hour) tunerConfig.stop_hour = parseInt(hour);
            if (minute) tunerConfig.stop_minute = parseInt(minute);
            break;
          case 'max_trials':
            tunerConfig.max_trials = parseInt(newTrainingFormData.max_trials);
            break;
          case 'endless':
            // Pas de paramètres supplémentaires pour endless
            break;
        }
        
        console.log('Lancement du tuner agent avec config:', tunerConfig);
        
        // Lancer le tuner agent via l'API
        const result = await tunerApi.startTuner(tunerConfig);
        
        if (result.status === 'success') {
          console.log('Tuner agent lancé avec succès:', result);
          setShowNewTrainingModal(false);
          
          // Ajouter le job à la liste des training jobs localement (optionnel)
          const newJob = {
            id: result.job_id,
            name: `Tuner ${tunerConfig.model_type.charAt(0).toUpperCase() + tunerConfig.model_type.slice(1)}`,
            status: 'starting',
            created_at: new Date().toISOString(),
            type: 'tuner_agent',
            model_type: tunerConfig.model_type,
            termination_type: tunerConfig.termination_type,
            command: result.command
          };
          
          // Refresh des training jobs pour voir le nouveau job
          refreshTraining();
          
          alert(`✅ Tuner agent lancé avec succès!\n🆔 Job ID: ${result.job_id}\n🤖 Model: ${tunerConfig.model_type}\n⏱️ Type: ${tunerConfig.termination_type}`);
          
          // Reset form
          setNewTrainingFormData({
            model_type: 'catboost',
            target_r2: 0.85,
            max_trials: 50,
            compute_target: 'local',
            machine_preference: 'auto',
            termination_type: 'max_trials',
            max_duration_hours: 2,
            end_time: '07:00'
          });
        } else {
          console.error('Erreur lors du lancement du tuner:', result);
          alert(`❌ Erreur lors du lancement du tuner agent:\n${result.detail || result.message || 'Erreur inconnue'}`);
        }
        
      } catch (error) {
        console.error('Erreur lors de la soumission:', error);
        alert(`❌ Erreur de connexion:\n${error.message}`);
      }
    };

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
        <div className="bg-white rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
          {/* Header */}
          <div className="bg-green-500 text-white p-6 flex items-center justify-between rounded-t-lg">
            <div>
              <h3 className="text-xl font-semibold">Start New Training</h3>
              <p className="text-sm opacity-90 mt-1">Configure your machine learning training job</p>
            </div>
            <button
              onClick={() => setShowNewTrainingModal(false)}
              className="text-white hover:bg-green-600 p-2 rounded-md transition-colors"
            >
              ✕
            </button>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="p-6 space-y-6">
            
            {/* Model Configuration */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-800">Model Configuration</h4>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Model Type</label>
                <select
                  value={newTrainingFormData.model_type}
                  onChange={(e) => setNewTrainingFormData({...newTrainingFormData, model_type: e.target.value})}
                  className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-green-500 focus:border-green-500"
                >
                  <option value="catboost">CatBoost + Optuna</option>
                  <option value="xgboost">XGBoost + Optuna</option>
                  <option value="lightgbm">LightGBM + Optuna</option>
                  <option value="random_forest">Random Forest + Optuna</option>
                  <option value="stack_ensemble">Stacked Ensemble + Optuna</option>
                </select>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Target R² Score</label>
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    value={newTrainingFormData.target_r2}
                    onChange={(e) => setNewTrainingFormData({...newTrainingFormData, target_r2: e.target.value})}
                    className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-green-500 focus:border-green-500"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Max Trials</label>
                  <input
                    type="number"
                    min="1"
                    max="1000"
                    value={newTrainingFormData.max_trials}
                    onChange={(e) => setNewTrainingFormData({...newTrainingFormData, max_trials: e.target.value})}
                    className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-green-500 focus:border-green-500"
                    disabled={newTrainingFormData.termination_type === 'endless'}
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Training Termination</label>
                <div className="space-y-2">
                  <label className="flex items-center p-3 border rounded-md hover:bg-gray-50 cursor-pointer">
                    <input
                      type="radio"
                      name="termination_type"
                      value="max_trials"
                      checked={newTrainingFormData.termination_type === 'max_trials'}
                      onChange={(e) => setNewTrainingFormData({...newTrainingFormData, termination_type: e.target.value})}
                      className="mr-3 text-green-600 focus:ring-green-500"
                    />
                    <div>
                      <div className="font-medium text-gray-900">Max Trials</div>
                      <div className="text-sm text-gray-600">Stop after reaching maximum number of trials</div>
                    </div>
                  </label>
                  <label className="flex items-center p-3 border rounded-md hover:bg-gray-50 cursor-pointer">
                    <input
                      type="radio"
                      name="termination_type"
                      value="time_duration"
                      checked={newTrainingFormData.termination_type === 'time_duration'}
                      onChange={(e) => setNewTrainingFormData({...newTrainingFormData, termination_type: e.target.value})}
                      className="mr-3 text-green-600 focus:ring-green-500"
                    />
                    <div className="flex-1">
                      <div className="font-medium text-gray-900">Time Duration</div>
                      <div className="text-sm text-gray-600">Stop after specified hours</div>
                      {newTrainingFormData.termination_type === 'time_duration' && (
                        <div className="mt-2">
                          <input
                            type="number"
                            min="0.5"
                            max="24"
                            step="0.5"
                            value={newTrainingFormData.max_duration_hours}
                            onChange={(e) => setNewTrainingFormData({...newTrainingFormData, max_duration_hours: e.target.value})}
                            className="w-20 p-2 border border-gray-300 rounded-md text-sm"
                            placeholder="Hours"
                          />
                          <span className="ml-2 text-sm text-gray-600">hours</span>
                        </div>
                      )}
                    </div>
                  </label>
                  <label className="flex items-center p-3 border rounded-md hover:bg-gray-50 cursor-pointer">
                    <input
                      type="radio"
                      name="termination_type"
                      value="end_time"
                      checked={newTrainingFormData.termination_type === 'end_time'}
                      onChange={(e) => setNewTrainingFormData({...newTrainingFormData, termination_type: e.target.value})}
                      className="mr-3 text-green-600 focus:ring-green-500"
                    />
                    <div className="flex-1">
                      <div className="font-medium text-gray-900">End Time</div>
                      <div className="text-sm text-gray-600">Stop at specific time (e.g., 7:00 AM)</div>
                      {newTrainingFormData.termination_type === 'end_time' && (
                        <div className="mt-2">
                          <input
                            type="time"
                            value={newTrainingFormData.end_time}
                            onChange={(e) => setNewTrainingFormData({...newTrainingFormData, end_time: e.target.value})}
                            className="w-32 p-2 border border-gray-300 rounded-md text-sm"
                          />
                        </div>
                      )}
                    </div>
                  </label>
                  <label className="flex items-center p-3 border rounded-md hover:bg-gray-50 cursor-pointer">
                    <input
                      type="radio"
                      name="termination_type"
                      value="endless"
                      checked={newTrainingFormData.termination_type === 'endless'}
                      onChange={(e) => setNewTrainingFormData({...newTrainingFormData, termination_type: e.target.value})}
                      className="mr-3 text-green-600 focus:ring-green-500"
                    />
                    <div>
                      <div className="font-medium text-gray-900">Endless Loop</div>
                      <div className="text-sm text-gray-600">Continue training until manually stopped</div>
                    </div>
                  </label>
                </div>
              </div>
            </div>

            {/* Compute Configuration */}
            <div className="space-y-4">
              <h4 className="text-lg font-semibold text-gray-800">Compute Configuration</h4>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Compute Target</label>
                <div className="space-y-2">
                  {availableComputeTargets.map((target) => (
                    <label key={target.value} className="flex items-center p-3 border rounded-md hover:bg-gray-50 cursor-pointer">
                      <input
                        type="radio"
                        name="compute_target"
                        value={target.value}
                        checked={newTrainingFormData.compute_target === target.value}
                        onChange={(e) => setNewTrainingFormData({...newTrainingFormData, compute_target: e.target.value})}
                        className="mr-3 text-green-600 focus:ring-green-500"
                      />
                      <div>
                        <div className="font-medium text-gray-900">{target.label}</div>
                        <div className="text-sm text-gray-600">{target.description}</div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">Machine Preference</label>
                <div className="space-y-2">
                  {availableMachines.map((machine) => (
                    <label key={machine.value} className="flex items-center p-3 border rounded-md hover:bg-gray-50 cursor-pointer">
                      <input
                        type="radio"
                        name="machine_preference"
                        value={machine.value}
                        checked={newTrainingFormData.machine_preference === machine.value}
                        onChange={(e) => setNewTrainingFormData({...newTrainingFormData, machine_preference: e.target.value})}
                        className="mr-3 text-green-600 focus:ring-green-500"
                      />
                      <div>
                        <div className="font-medium text-gray-900">{machine.label}</div>
                        <div className="text-sm text-gray-600">{machine.description}</div>
                      </div>
                    </label>
                  ))}
                </div>
              </div>
            </div>

            {/* Info Box */}
            <div className="bg-blue-50 p-4 rounded-lg border border-blue-200">
              <div className="flex items-start">
                <div className="ml-3">
                  <p className="text-sm text-blue-800">
                    <strong>Training will start immediately</strong> on the selected compute target. 
                    You can monitor progress in the Training Pipeline tab and stop the job at any time.
                  </p>
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-3 pt-4">
              <button
                type="button"
                onClick={() => setShowNewTrainingModal(false)}
                className="flex-1 px-4 py-3 text-gray-700 bg-gray-100 rounded-md hover:bg-gray-200 transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                className="flex-1 px-4 py-3 bg-green-500 text-white rounded-md hover:bg-green-600 transition-colors font-medium"
              >
                Start Training
              </button>
            </div>
          </form>
        </div>
      </div>
    );
  };

  const renderExperimentsTab = () => {
    // Calcul du ranking_score et enrichissement des expériences avec logique de production-readiness
    const penalty_map = {
      "Low": 0.0,
      "Moderate": 0.02,
      "High": 0.04,
      "Very High": 0.07
    };

    let filteredExperiments = includeTestModels 
      ? experiments 
      : experiments.filter(exp => !exp.model_name?.includes('[TEST]'));

    let processedExperiments = filteredExperiments.map((exp) => {
      const r2_test = exp.r2_test || 0;
      const r2_train = exp.r2_train || 0;
      const r2_gap = exp.r2_gap !== undefined ? exp.r2_gap : (r2_train - r2_test);
      const generalization_index = exp.generalization_index !== undefined ? exp.generalization_index : computeGeneralizationIndex(r2_train, r2_test);
      const generalization_label = getGeneralizationLabel(generalization_index);
      const overfitting_risk = exp.overfitting_risk || getOverfittingRisk(r2_gap);
      const overfitting_penalty = penalty_map[overfitting_risk] ?? 0.07;
      
      // Nouveau scoring axé sur la production-readiness
      let production_bonus = 0;
      if (r2_test >= 0.85) production_bonus += 0.1;
      if (generalization_index >= 90) production_bonus += 0.05;
      if (["Low", "Moderate"].includes(overfitting_risk)) production_bonus += 0.03;
      
      const ranking_score = r2_test - (r2_gap * 2) + (generalization_index / 100 * 0.2) - overfitting_penalty + production_bonus;
      
      return {
        ...exp,
        model: exp.model_name || 'CatBoost CV (All Features)',
        r2_gap: r2_gap.toFixed(6),
        r2_gap_diagnostic: getGeneralizationDiagnostic(r2_train, r2_test),
        generalization_index: generalization_index,
        generalization_label: generalization_label,
        overfitting_risk,
        ranking_score: ranking_score,
        n_features: exp.feature_count || 2885
      };
    });

    // Tri automatique par ranking_score décroissant (production-readiness first)
    processedExperiments = processedExperiments.sort((a, b) => b.ranking_score - a.ranking_score);

    // Attribution du nouveau rang
    processedExperiments = processedExperiments.map((exp, idx) => ({
      ...exp,
      rank: idx + 1,
      best: idx === 0 ? '✓' : ''
    }));

    // Nouveau résumé basé sur le filtrage
    const filteredSummary = processedExperiments.length > 0 ? {
      total_experiments: processedExperiments.length,
      best_r2_score: Math.max(...processedExperiments.map(exp => exp.r2_test || 0)),
      average_r2_score: processedExperiments.reduce((sum, exp) => sum + (exp.r2_test || 0), 0) / processedExperiments.length,
      average_r2_gap: processedExperiments.reduce((sum, exp) => sum + (parseFloat(exp.r2_gap) || 0), 0) / processedExperiments.length,
      latest_experiment: processedExperiments.reduce((latest, exp) => 
        (!latest.timestamp || new Date(exp.timestamp || 0) > new Date(latest.timestamp || 0)) ? exp : latest, {}),
      best_generalization: processedExperiments.reduce((best, exp) => {
        const gap = parseFloat(exp.r2_gap);
        return (!best || gap < parseFloat(best.r2_gap)) ? exp : best;
      }, null)
    } : summary;

    return (
      <div className="space-y-6">
        
        {/* Header avec statistiques enrichies */}
        <div className="bg-white rounded-lg border p-6">
          <div className="flex flex-col md:flex-row md:justify-between md:items-center mb-4 gap-2">
            <h3 className="text-lg font-semibold text-gray-800">Experiment History</h3>
            <div className="flex flex-col md:flex-row md:space-x-4 gap-2">
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
                onClick={() => setShowHelpModal(true)}
                className="bg-gray-500 text-white px-4 py-2 rounded hover:bg-gray-600 transition-colors"
              >
                Metrics Interpretation
              </button>
              <button 
                onClick={refresh}
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 transition-colors"
              >
                Refresh
              </button>
            </div>
          </div>

          {/* Bloc d'explication amélioré */}
          <div className="mb-4 p-4 bg-blue-50 border-l-4 border-blue-400 rounded text-gray-700 text-sm">
            <div className="font-semibold text-blue-800 mb-2">Smart Model Ranking System</div>
            <div>Models are automatically ranked based on <strong>production-readiness</strong>, not just performance.<br/>
            The ranking score prioritizes R² Test ≥ 0.85, low R² Gap, high Generalization Index, and low Overfitting Risk.<br/>
            <span className="text-blue-600 font-medium">→ Top models generalize well and avoid overfitting.</span></div>
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
                <span className={getDiagnosticColor(getGeneralizationDiagnostic(filteredSummary.best_generalization.r2_train, filteredSummary.best_generalization.r2_test))}>
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
        {!loading && !error && processedExperiments.length > 0 && (
          <div className="bg-white rounded-lg border overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[1400px] text-xs">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-1 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style={{width: '40px'}}>Rank</th>
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
                    <th className="px-1 py-3 text-center text-xs font-medium text-gray-500 uppercase tracking-wider" style={{width: '110px'}}>Ranking Score</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {processedExperiments.map((exp, index) => (
                    <tr 
                      key={exp.id} 
                      className="hover:bg-blue-50 cursor-pointer"
                      onClick={() => handleModelClick(exp)}
                    >
                      <td className="px-1 py-3 text-xs text-center text-gray-900" style={{width: '40px'}}>{exp.rank}</td>
                      <td className="px-2 py-3 text-xs text-center text-gray-900" style={{width: '160px'}}>{formatDate(exp.timestamp)}</td>
                      <td className="px-2 py-3 text-xs text-center text-gray-900" style={{width: '200px'}} title={exp.model}>{exp.model}</td>
                      <td className="px-1 py-3 text-xs text-center text-gray-900" style={{width: '100px'}}>{formatTrainingTime(exp.training_time)}</td>
                      <td className="px-1 py-3 text-xs text-center text-gray-900" style={{width: '90px'}}>{formatMAE(exp.mae_train)}</td>
                      <td className="px-1 py-3 text-xs text-center text-gray-900" style={{width: '90px'}}>{formatMAE(exp.rmse_train)}</td>
                      <td className="px-1 py-3 text-xs text-center text-gray-900" style={{width: '90px'}}>{formatR2Score(exp.r2_train)}</td>
                      <td className="px-1 py-3 text-xs text-center text-gray-900" style={{width: '90px'}}>{formatMAE(exp.mae_test)}</td>
                      <td className="px-1 py-3 text-xs text-center text-gray-900" style={{width: '90px'}}>{formatMAE(exp.rmse_test)}</td>
                      <td className={`px-1 py-3 text-xs text-center font-medium ${exp.r2_test >= 0.85 ? 'text-green-600' : 'text-gray-900'}`} style={{width: '90px'}}>{formatR2Score(exp.r2_test)}</td>
                      <td className={`px-1 py-3 text-xs text-center font-medium ${parseFloat(exp.r2_gap) > 0.12 ? 'text-red-600' : 'text-gray-900'}`} style={{width: '70px'}}>{exp.r2_gap}</td>
                      <td className="px-2 py-3 text-xs text-center" style={{width: '120px'}}>
                        <span className={getDiagnosticColor(exp.r2_gap_diagnostic)}>{exp.r2_gap_diagnostic}</span>
                      </td>
                      <td className="px-1 py-3 text-xs text-center text-gray-900" style={{width: '90px'}}>{Number(exp.generalization_index)}</td>
                      <td className="px-2 py-3 text-xs text-center" style={{width: '100px'}}>
                        <span className={getGeneralizationLabelColor(exp.generalization_label)}>{exp.generalization_label}</span>
                      </td>
                      <td className="px-2 py-3 text-xs text-center" style={{width: '110px'}}>
                        <span className={getOverfittingRiskColor(exp.overfitting_risk)}>{exp.overfitting_risk}</span>
                      </td>
                      <td className="px-1 py-3 text-xs text-center text-gray-900" style={{width: '80px'}}>{exp.n_features}</td>
                      <td className="px-1 py-3 text-xs text-center font-medium text-blue-600" style={{width: '110px'}}>{exp.ranking_score.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {!loading && !error && processedExperiments.length === 0 && (
          <div className="bg-gray-50 rounded-lg p-8 text-center">
            <p className="text-gray-500">
              {experiments.length === 0 ? 'No experiments found' : 'No experiments found with current filter'}
            </p>
          </div>
        )}
      </div>
    );
  };

  const renderPipelineTab = () => {
    const activeJobs = trainingJobs.filter(job => job.status === 'running' || job.status === 'queued');
    const recentJobs = trainingJobs.filter(job => job.status === 'completed' || job.status === 'stopped' || job.status === 'failed');

    // Groupe les jobs par machine pour affichage
    const groupJobsByMachine = (jobs) => {
      return jobs.reduce((acc, job) => {
        const machine = job.machine_name || 'Unknown';
        if (!acc[machine]) acc[machine] = [];
        acc[machine].push(job);
        return acc;
      }, {});
    };

    const activeJobsByMachine = groupJobsByMachine(activeJobs);
    const uniqueMachines = [...new Set(trainingJobs.map(j => j.machine_name || 'Unknown'))];

    const handleStartNewTraining = async () => {
      setShowNewTrainingModal(true);
    };

    const handleStopTraining = async (jobId) => {
      const result = await stopTrainingJob(jobId);
      if (result.success) {
        console.log('Training stopped successfully');
      } else {
        console.error('Error stopping training:', result.error);
      }
    };

    return (
      <div className="space-y-6">
        {/* Controls Section */}
        <div className="bg-white rounded-lg border p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-gray-800">Training Pipeline</h3>
            <div className="flex gap-3">
              <button 
                onClick={refreshTraining}
                disabled={trainingLoading}
                className="bg-blue-500 text-white px-4 py-2 rounded hover:bg-blue-600 disabled:bg-gray-400 text-sm"
              >
                Refresh
              </button>
              <button 
                onClick={handleStartNewTraining}
                className="bg-green-500 text-white px-4 py-2 rounded hover:bg-green-600 text-sm"
              >
                New Training
              </button>
            </div>
          </div>

          {/* Quick Statistics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-green-50 border border-green-200 rounded p-3 text-center">
              <div className="text-2xl font-bold text-green-600">{activeJobs.length}</div>
              <div className="text-sm text-green-700">Running</div>
            </div>
            <div className="bg-blue-50 border border-blue-200 rounded p-3 text-center">
              <div className="text-2xl font-bold text-blue-600">{recentJobs.length}</div>
              <div className="text-sm text-blue-700">Completed</div>
            </div>
            <div className="bg-purple-50 border border-purple-200 rounded p-3 text-center">
              <div className="text-2xl font-bold text-purple-600">{uniqueMachines.length}</div>
              <div className="text-sm text-purple-700">Machines</div>
            </div>
            <div className="bg-orange-50 border border-orange-200 rounded p-3 text-center">
              <div className="text-2xl font-bold text-orange-600">
                {trainingJobs.filter(j => j.compute_target && j.compute_target.includes('Azure')).length}
              </div>
              <div className="text-sm text-orange-700">On Azure</div>
            </div>
          </div>

          {trainingError && (
            <div className="bg-red-50 border border-red-200 rounded p-3 mb-4">
              <p className="text-red-700 text-sm">Error: {trainingError}</p>
            </div>
          )}
        </div>

        {/* Active Training Jobs */}
        {activeJobs.length > 0 && (
          <div className="bg-white rounded-lg border p-6">
            <h4 className="text-lg font-semibold text-gray-800 mb-4">
              Active Training Jobs ({activeJobs.length})
            </h4>
            
            {/* Group by machine */}
            {Object.entries(activeJobsByMachine).map(([machine, jobs]) => (
              <div key={machine} className="mb-6 last:mb-0">
                <div className="flex items-center mb-3 p-2 bg-gray-50 rounded">
                  <div className="flex items-center">
                    <div className="w-3 h-3 bg-green-500 rounded-full mr-2"></div>
                    <span className="font-medium text-gray-800">Machine: {machine}</span>
                    <span className="ml-3 text-sm text-gray-600">({jobs.length} job{jobs.length > 1 ? 's' : ''})</span>
                  </div>
                </div>
                <div className="space-y-4">
                  {jobs.map(job => (
                    <TrainingJobCard 
                      key={job.id} 
                      job={job} 
                      onStop={handleStopTraining}
                      showMachine={false} // Don't show machine in card since it's in group header
                    />
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Recent Training Jobs */}
        {recentJobs.length > 0 && (
          <div className="bg-white rounded-lg border p-6">
            <h4 className="text-lg font-semibold text-gray-800 mb-4">
              Recent Training Jobs ({recentJobs.length})
            </h4>
            <div className="space-y-4">
              {recentJobs.slice(0, 5).map(job => (
                <TrainingJobCard 
                  key={job.id} 
                  job={job} 
                  onStop={handleStopTraining}
                  showMachine={true} // Show machine in card for completed jobs
                />
              ))}
            </div>
            {recentJobs.length > 5 && (
              <div className="text-center mt-4">
                <button className="text-blue-500 hover:text-blue-700 text-sm">
                  View all recent training jobs ({recentJobs.length - 5} more)
                </button>
              </div>
            )}
          </div>
        )}

        {/* Empty State */}
        {trainingJobs.length === 0 && !trainingLoading && (
          <div className="bg-white rounded-lg border p-12 text-center">
            <div className="text-gray-400 text-6xl mb-4">No Training Jobs</div>
            <h4 className="text-lg font-medium text-gray-600 mb-2">No training jobs found</h4>
            <p className="text-gray-500 mb-6">
              Start your first training job to see them here
            </p>
            <button 
              onClick={() => setShowNewTrainingModal(true)}
              className="bg-green-500 text-white px-6 py-3 rounded hover:bg-green-600"
            >
              Start Training
            </button>
          </div>
        )}

        {/* Loading state */}
        {trainingLoading && (
          <div className="bg-white rounded-lg border p-12 text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading training jobs...</p>
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
      
      {/* Hyperparameters Panel - SANS overlay sombre */}
      {renderHyperparametersPanel()}
      
      {/* Generalization Modal */}
      {renderGeneralizationModal()}
      
      {/* Help Modal */}
      {renderHelpModal()}
      
      {/* New Training Modal */}
      {renderNewTrainingModal()}
      
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
