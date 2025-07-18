import React, { useState, useEffect } from 'react';
import { useExperiments } from '../hooks/useExperiments';

const ModelTrainingPage = () => {
  const [activeTab, setActiveTab] = useState('experiments');
  const [isTraining, setIsTraining] = useState(false);
  
  const { 
    experiments, 
    summary, 
    loading, 
    error, 
    refresh 
  } = useExperiments();

  const tabs = [
    { id: 'pipeline', label: 'Training Pipeline' },
    { id: 'experiments', label: 'Experiments' },
    { id: 'optimization', label: 'Optimization' },
    { id: 'deployment', label: 'Deployment' }
  ];

  const startTraining = () => {
    setIsTraining(true);
    setTimeout(() => setIsTraining(false), 5000);
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
    if (mae === null || mae === undefined) return '€0';
    return `${(mae / 1000).toFixed(1)} k€`;
  };

  const formatDuration = (duration) => {
    if (!duration) return 'N/A';
    return duration;
  };

  const getScoreColor = (score) => {
    if (score >= 0.8) return 'text-green-600';
    if (score >= 0.6) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getR2GapColor = (gap) => {
    const gapValue = parseFloat(gap);
    if (gapValue <= 0.02) return 'bg-green-100 text-green-800 px-2 py-1 rounded';
    if (gapValue <= 0.10) return 'bg-yellow-100 text-yellow-800 px-2 py-1 rounded';
    return 'bg-red-100 text-red-800 px-2 py-1 rounded';
  };

  const getDiagnosticColor = (diagnostic) => {
    if (diagnostic === 'Excellent generalization') return 'text-green-600 font-medium';
    if (diagnostic === 'Good generalization') return 'text-blue-600 font-medium';
    if (diagnostic === 'Moderate overfitting') return 'text-yellow-600 font-medium';
    return 'text-red-600 font-medium';
  };

  const getGeneralizationDiagnostic = (r2_train, r2_test) => {
    if (!r2_train || !r2_test) return 'N/A';
    const gap = r2_train - r2_test;
    
    if (gap <= 0.02 && r2_test > 0.85) return 'Excellent generalization';
    if (gap <= 0.05 && r2_test > 0.75) return 'Good generalization';
    if (gap <= 0.10) return 'Moderate overfitting';
    return 'Strong overfitting';
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
    // Traitement des données pour le format du tableau
    const processedExperiments = experiments.map((exp, index) => {
      // Trier par R² test décroissant
      const sortedExperiments = [...experiments].sort((a, b) => (b.r2_test || 0) - (a.r2_test || 0));
      const rank = sortedExperiments.findIndex(e => e.id === exp.id) + 1;
      
      return {
        ...exp,
        rank,
        best: rank === 1 ? '✓' : '',
        model: 'CatBoost CV (All Features)',
        r2_gap: ((exp.r2_train || 0) - (exp.r2_test || 0)).toFixed(6),
        r2_gap_diagnostic: getGeneralizationDiagnostic(exp.r2_train, exp.r2_test),
        n_features: 2885
      };
    }).sort((a, b) => a.rank - b.rank);

    return (
      <div className="space-y-6">
        {/* Header avec statistiques */}
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
          
          {/* Statistiques de résumé */}
          {summary && (
            <div className="grid grid-cols-4 gap-4 mb-6">
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
                <div className="text-2xl font-bold text-gray-600">{summary.latest_experiment?.split('_')[1]?.split('T')[0]}</div>
                <div className="text-sm text-gray-600">Latest Date</div>
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

        {/* Tableau des expériences - Format exact selon l'image */}
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
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">MAE Train</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">RMSE Train</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">R² Train</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">MAE Test</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">RMSE Test</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">R² Test</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">R² Gap</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">R² Gap Diagnostic</th>
                    <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">N Features</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {processedExperiments.map((exp, index) => (
                    <tr key={exp.id} className={exp.rank === 1 ? 'bg-green-50' : 'hover:bg-gray-50'}>
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
                      <td className={`px-4 py-3 text-sm ${getDiagnosticColor(exp.r2_gap_diagnostic)}`}>
                        {exp.r2_gap_diagnostic}
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

  const renderTabContent = () => {
    switch (activeTab) {
      case 'pipeline': return renderPipelineTab();
      case 'experiments': return renderExperimentsTab();
      case 'optimization': return <div className="text-center py-8">Optimization tools coming soon...</div>;
      case 'deployment': return <div className="text-center py-8">Deployment options coming soon...</div>;
      default: return renderExperimentsTab();
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto px-4 py-8">
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
