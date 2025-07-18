/**
 * Model Training Agent Page
 * Azure ML integration and training pipeline management
 */

import React, { useState } from 'react';

const ModelTrainingPage = () => {
  const [activeTab, setActiveTab] = useState('pipeline');
  const [isTraining, setIsTraining] = useState(false);

  const tabs = [
    { id: 'pipeline', label: 'Training Pipeline' },
    { id: 'experiments', label: 'Experiments' },
    { id: 'optimization', label: 'Optimization' },
    { id: 'deployment', label: 'Deployment' }
  ];

  const startTraining = () => {
    setIsTraining(true);
    // Simulate training completion after 5 seconds
    setTimeout(() => setIsTraining(false), 5000);
  };

  const renderPipelineTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg border p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Azure ML Training Pipeline</h3>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-gray-700 mb-3">Configuration</h4>
            <div className="bg-gray-50 rounded-lg p-4 space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600">Compute Target:</span>
                <span className="font-medium">Tesla V100 (4 nodes)</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Environment:</span>
                <span className="font-medium">AzureML-sklearn-1.0</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Dataset:</span>
                <span className="font-medium">belgian_real_estate_v2</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Quality Gate:</span>
                <span className="font-medium">R² ≥ 0.85</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">Auto-scaling:</span>
                <span className="font-medium">0-4 nodes</span>
              </div>
            </div>
          </div>
          
          <div>
            <h4 className="font-medium text-gray-700 mb-3">Current Status</h4>
            <div className="bg-green-50 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-green-800 font-medium">Ready for Training</span>
                <div className="w-3 h-3 bg-green-500 rounded-full"></div>
              </div>
              <p className="text-green-600 text-sm">
                All prerequisites validated. GPU cluster available.
              </p>
              <div className="mt-3 text-sm text-green-700">
                <div>• Data validation: Passed</div>
                <div>• Feature engineering: Complete</div>
                <div>• Model registration: Ready</div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="mt-6 flex space-x-4">
          <button 
            onClick={startTraining}
            disabled={isTraining}
            className="bg-blue-600 text-white px-6 py-2 rounded-lg hover:bg-blue-700 disabled:opacity-50 flex items-center space-x-2"
          >
            {isTraining ? (
              <>
                <div className="animate-spin w-4 h-4 border-2 border-white border-t-transparent rounded-full"></div>
                <span>Training in Progress...</span>
              </>
            ) : (
              <span>Start New Training</span>
            )}
          </button>
          
          <button className="bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 transition-colors">
            View Metrics
          </button>
          
          <button className="bg-gray-600 text-white px-6 py-3 rounded-lg hover:bg-gray-700 transition-colors">
            Advanced Config
          </button>
        </div>
        
        {isTraining && (
          <div className="mt-6 bg-blue-50 rounded-lg p-4">
            <h4 className="font-medium text-gray-700 mb-3">Training Progress</h4>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Data Preprocessing</span>
                  <span>100%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-green-500 h-2 rounded-full w-full"></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Model Training</span>
                  <span>67%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-blue-500 h-2 rounded-full w-2/3 animate-pulse"></div>
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span>Validation</span>
                  <span>0%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div className="bg-gray-300 h-2 rounded-full w-0"></div>
                </div>
              </div>
            </div>
            
            <div className="mt-4 text-sm text-gray-600">
              <div>Estimated completion: 12 minutes</div>
              <div>Current R²: 0.823 (improving...)</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  const renderExperimentsTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg border p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Experiment History</h3>
        
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b">
                <th className="text-left p-2">Experiment</th>
                <th className="text-left p-2">Date</th>
                <th className="text-left p-2">R² Score</th>
                <th className="text-left p-2">MAE</th>
                <th className="text-left p-2">Status</th>
                <th className="text-left p-2">Duration</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b hover:bg-gray-50">
                <td className="p-2 font-medium">exp_20241215_003</td>
                <td className="p-2">2024-12-15</td>
                <td className="p-2"><span className="text-green-600 font-medium">0.891</span></td>
                <td className="p-2">€15,240</td>
                <td className="p-2"><span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">Complete</span></td>
                <td className="p-2">18m 32s</td>
              </tr>
              <tr className="border-b hover:bg-gray-50">
                <td className="p-2 font-medium">exp_20241214_012</td>
                <td className="p-2">2024-12-14</td>
                <td className="p-2"><span className="text-yellow-600 font-medium">0.847</span></td>
                <td className="p-2">€18,890</td>
                <td className="p-2"><span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">Complete</span></td>
                <td className="p-2">22m 15s</td>
              </tr>
              <tr className="border-b hover:bg-gray-50">
                <td className="p-2 font-medium">exp_20241213_008</td>
                <td className="p-2">2024-12-13</td>
                <td className="p-2"><span className="text-red-600 font-medium">0.823</span></td>
                <td className="p-2">€21,450</td>
                <td className="p-2"><span className="bg-red-100 text-red-800 px-2 py-1 rounded text-xs">Failed QG</span></td>
                <td className="p-2">16m 44s</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );

  const renderOptimizationTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg border p-6">
        <h4 className="font-medium text-gray-700 mb-4">Optuna Optimization Results</h4>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h5 className="font-medium text-gray-600 mb-3">Best Parameters</h5>
            <div className="bg-gray-50 rounded-lg p-4 space-y-2">
              <div className="flex justify-between">
                <span className="text-gray-600">n_estimators:</span>
                <span className="font-mono">847</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">max_depth:</span>
                <span className="font-mono">12</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">learning_rate:</span>
                <span className="font-mono">0.0847</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600">subsample:</span>
                <span className="font-mono">0.923</span>
              </div>
            </div>
          </div>
          
          <div>
            <h5 className="font-medium text-gray-600 mb-3">Optimization Stats</h5>
            <div className="bg-blue-50 rounded-lg p-4 space-y-2">
              <div><strong>247 trials</strong> completed</div>
              <div><strong>3.2 hours</strong> total time</div>
              <div><strong>+8.3%</strong> improvement</div>
              <div><strong>R² 0.891</strong> best score</div>
            </div>
          </div>
        </div>
        
        <div className="mt-6">
          <h5 className="font-medium text-gray-600 mb-3">Hyperparameter Importance</h5>
          <div className="space-y-2">
            <div className="flex items-center">
              <span className="w-24 text-sm text-gray-600">n_estimators</span>
              <div className="flex-1 bg-gray-200 rounded-full h-2 mx-3">
                <div className="bg-blue-500 h-2 rounded-full" style={{width: '89%'}}></div>
              </div>
              <span className="text-sm font-medium">89%</span>
            </div>
            <div className="flex items-center">
              <span className="w-24 text-sm text-gray-600">max_depth</span>
              <div className="flex-1 bg-gray-200 rounded-full h-2 mx-3">
                <div className="bg-green-500 h-2 rounded-full" style={{width: '67%'}}></div>
              </div>
              <span className="text-sm font-medium">67%</span>
            </div>
            <div className="flex items-center">
              <span className="w-24 text-sm text-gray-600">learning_rate</span>
              <div className="flex-1 bg-gray-200 rounded-full h-2 mx-3">
                <div className="bg-yellow-500 h-2 rounded-full" style={{width: '34%'}}></div>
              </div>
              <span className="text-sm font-medium">34%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  const renderDeploymentTab = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg border p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Model Deployment</h3>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div>
            <h4 className="font-medium text-gray-700 mb-3">Production Model</h4>
            <div className="bg-green-50 rounded-lg p-4">
              <div className="flex justify-between items-center mb-2">
                <span className="font-medium text-green-800">v2.1.3-prod</span>
                <span className="bg-green-100 text-green-800 px-2 py-1 rounded text-xs">Active</span>
              </div>
              <div className="text-sm text-green-600 space-y-1">
                <div>R² Score: 0.891</div>
                <div>Deployed: 2024-12-15</div>
                <div>Requests: 12,847 today</div>
                <div>Avg Response: 234ms</div>
              </div>
            </div>
          </div>
          
          <div>
            <h4 className="font-medium text-gray-700 mb-3">A/B Testing</h4>
            <div className="bg-blue-50 rounded-lg p-4">
              <div className="flex justify-between items-center mb-2">
                <span className="font-medium text-blue-800">Candidate Model</span>
                <span className="bg-blue-100 text-blue-800 px-2 py-1 rounded text-xs">Testing</span>
              </div>
              <div className="text-sm text-blue-600 space-y-1">
                <div>Traffic Split: 10%</div>
                <div>Performance: +2.3%</div>
                <div>Confidence: 94.2%</div>
                <div>ETA to decision: 2 days</div>
              </div>
            </div>
          </div>
        </div>
        
        <div className="mt-6 flex space-x-4">
          <button className="bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors">
            Deploy to Production
          </button>
          
          <button className="bg-green-600 text-white px-6 py-3 rounded-lg hover:bg-green-700 transition-colors">
            View Performance
          </button>
          
          <button className="bg-yellow-600 text-white px-6 py-3 rounded-lg hover:bg-yellow-700 transition-colors">
            Rollback
          </button>
        </div>
      </div>
    </div>
  );

  const renderActiveTab = () => {
    switch(activeTab) {
      case 'pipeline': return renderPipelineTab();
      case 'experiments': return renderExperimentsTab();
      case 'optimization': return renderOptimizationTab();
      case 'deployment': return renderDeploymentTab();
      default: return renderPipelineTab();
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">
            Model Training Agent
          </h1>
          <p className="text-gray-600">
            Azure ML integration for advanced model training and optimization
          </p>
        </div>

        {/* Tab Navigation */}
        <div className="mb-6">
          <div className="border-b border-gray-200">
            <nav className="flex space-x-8">
              {tabs.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`py-2 px-1 border-b-2 font-medium text-sm transition-colors ${
                    activeTab === tab.id
                      ? 'border-blue-500 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        {renderActiveTab()}
        
        {/* Quick Stats */}
        <div className="mt-8 grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="bg-white rounded-lg border p-4 text-center">
            <div className="text-2xl font-bold text-blue-600">€62/mo</div>
            <div className="text-sm text-gray-600">Azure ML Cost</div>
          </div>
          <div className="bg-white rounded-lg border p-4 text-center">
            <div className="text-2xl font-bold text-green-600">0.891</div>
            <div className="text-sm text-gray-600">Best R² Score</div>
          </div>
          <div className="bg-white rounded-lg border p-4 text-center">
            <div className="text-2xl font-bold text-purple-600">247</div>
            <div className="text-sm text-gray-600">Optuna Trials</div>
          </div>
          <div className="bg-white rounded-lg border p-4 text-center">
            <div className="text-2xl font-bold text-orange-600">18m</div>
            <div className="text-sm text-gray-600">Avg Training</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelTrainingPage;
