/**
 * Training Status Panel - Azure ML training monitoring
 */

import React, { useState, useEffect } from 'react';
import { Play, Pause, Square, TrendingUp, Clock, Cpu } from 'lucide-react';

const TrainingStatusPanel = () => {
  const [trainingJobs, setTrainingJobs] = useState([]);
  const [azureMLStatus, setAzureMLStatus] = useState('connected');

  useEffect(() => {
    // Demo training jobs data
    setTrainingJobs([
      {
        id: 'real-estate-opt-001',
        name: 'Real Estate Optimization',
        status: 'running',
        progress: 78,
        eta_minutes: 7,
        current_trial: 39,
        total_trials: 50,
        best_r2: 0.851,
        compute_target: 'gpu-cluster',
        started_at: '2025-07-15T15:30:00Z',
        target_r2: 0.85,
        current_gap: 0.03
      },
      {
        id: 'hyperopt-weekend-002', 
        name: 'Weekend Hyperparameter Sweep',
        status: 'completed',
        progress: 100,
        eta_minutes: 0,
        current_trial: 100,
        total_trials: 100,
        best_r2: 0.847,
        compute_target: 'gpu-cluster',
        started_at: '2025-07-13T09:00:00Z',
        completed_at: '2025-07-13T11:45:00Z',
        target_r2: 0.85,
        final_gap: 0.02
      }
    ]);

    // Simulate progress updates for running jobs
    const interval = setInterval(() => {
      setTrainingJobs(prev => 
        prev.map(job => {
          if (job.status === 'running' && job.progress < 100) {
            const newProgress = Math.min(100, job.progress + Math.random() * 3);
            const newTrial = Math.min(job.total_trials, job.current_trial + (Math.random() < 0.3 ? 1 : 0));
            const newEta = Math.max(0, job.eta_minutes - 0.2);
            const newR2 = job.best_r2 + (Math.random() - 0.5) * 0.002;
            
            return {
              ...job,
              progress: newProgress,
              current_trial: newTrial,
              eta_minutes: newEta,
              best_r2: Math.max(0.8, Math.min(0.9, newR2)),
              current_gap: Math.max(0.01, Math.min(0.08, job.current_gap + (Math.random() - 0.5) * 0.005))
            };
          }
          return job;
        })
      );
    }, 3000);

    return () => clearInterval(interval);
  }, []);

  const getStatusConfig = (status) => {
    const configs = {
      running: { color: 'blue', icon: Play, label: '🔵 Running' },
      completed: { color: 'green', icon: Square, label: '✅ Completed' },
      failed: { color: 'red', icon: Square, label: '❌ Failed' },
      cancelled: { color: 'gray', icon: Pause, label: '⏸️ Cancelled' },
      queued: { color: 'yellow', icon: Clock, label: '🟡 Queued' }
    };
    return configs[status] || configs.queued;
  };

  const formatDuration = (startTime, endTime = null) => {
    const start = new Date(startTime);
    const end = endTime ? new Date(endTime) : new Date();
    const diffMs = end - start;
    const hours = Math.floor(diffMs / (1000 * 60 * 60));
    const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
    
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes}m`;
  };

  const startNewTraining = async () => {
    try {
      const response = await fetch('/api/admin/training/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          target_r2: 0.85,
          max_trials: 50,
          compute_target: 'gpu-cluster'
        })
      });

      if (response.ok) {
        // Add new job to the list
        const newJob = {
          id: `training-${Date.now()}`,
          name: 'New Training Session',
          status: 'queued',
          progress: 0,
          eta_minutes: 15,
          current_trial: 0,
          total_trials: 50,
          best_r2: 0.0,
          compute_target: 'gpu-cluster',
          started_at: new Date().toISOString(),
          target_r2: 0.85,
          current_gap: 0.0
        };
        
        setTrainingJobs(prev => [newJob, ...prev]);
        alert('Training job submitted to Azure ML!');
      }
    } catch (error) {
      console.error('Failed to start training:', error);
      alert('Failed to start training job.');
    }
  };

  return (
    <div className="space-y-4">
      {/* Header with Azure ML Status */}
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold text-gray-800">⚡ Azure ML Training</h3>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${azureMLStatus === 'connected' ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
          <span className="text-xs text-gray-600">Azure ML</span>
        </div>
      </div>

      {/* Quick Start Training */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-3 border">
        <h4 className="font-medium text-gray-800 mb-2">🚀 Quick Start</h4>
        <div className="flex space-x-2">
          <button 
            onClick={startNewTraining}
            className="flex-1 bg-blue-600 text-white py-2 px-3 rounded text-sm hover:bg-blue-700 transition-colors flex items-center justify-center space-x-1"
          >
            <Play size={14} />
            <span>Start Training</span>
          </button>
          <button className="bg-green-600 text-white py-2 px-3 rounded text-sm hover:bg-green-700 transition-colors">
            📊 Templates
          </button>
        </div>
      </div>

      {/* Training Jobs List */}
      <div className="space-y-3">
        {trainingJobs.map((job) => {
          const statusConfig = getStatusConfig(job.status);
          const IconComponent = statusConfig.icon;
          
          return (
            <div key={job.id} className="bg-white rounded-lg border p-4">
              {/* Job Header */}
              <div className="flex justify-between items-center mb-3">
                <div>
                  <h4 className="font-medium text-gray-800">{job.name}</h4>
                  <p className="text-sm text-gray-600">ID: {job.id}</p>
                </div>
                <span className={`text-sm bg-${statusConfig.color}-100 text-${statusConfig.color}-800 px-2 py-1 rounded flex items-center space-x-1`}>
                  <IconComponent size={12} />
                  <span>{statusConfig.label}</span>
                </span>
              </div>

              {/* Progress Bar */}
              {job.status === 'running' && (
                <div className="mb-3">
                  <div className="flex justify-between text-sm mb-1">
                    <span>Progress: {job.current_trial}/{job.total_trials} trials</span>
                    <span>{job.progress.toFixed(1)}%</span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div 
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${job.progress}%` }}
                    ></div>
                  </div>
                </div>
              )}

              {/* Metrics Grid */}
              <div className="grid grid-cols-2 gap-3 mb-3">
                <div className="text-center p-2 bg-gray-50 rounded">
                  <div className={`text-lg font-bold ${job.best_r2 >= job.target_r2 ? 'text-green-600' : 'text-blue-600'}`}>
                    {job.best_r2.toFixed(3)}
                  </div>
                  <div className="text-xs text-gray-500">
                    Best R² {job.best_r2 >= job.target_r2 && '✅'}
                  </div>
                </div>
                <div className="text-center p-2 bg-gray-50 rounded">
                  <div className={`text-lg font-bold ${job.current_gap <= 0.05 ? 'text-green-600' : 'text-yellow-600'}`}>
                    {job.current_gap?.toFixed(3) || job.final_gap?.toFixed(3) || 'N/A'}
                  </div>
                  <div className="text-xs text-gray-500">
                    Val Gap {(job.current_gap <= 0.05 || job.final_gap <= 0.05) && '✅'}
                  </div>
                </div>
              </div>

              {/* Job Details */}
              <div className="text-sm text-gray-600 space-y-1">
                <div className="flex justify-between">
                  <span>Compute:</span>
                  <span className="flex items-center space-x-1">
                    <Cpu size={12} />
                    <span>{job.compute_target}</span>
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>Duration:</span>
                  <span>{formatDuration(job.started_at, job.completed_at)}</span>
                </div>
                {job.status === 'running' && (
                  <div className="flex justify-between">
                    <span>ETA:</span>
                    <span className="flex items-center space-x-1">
                      <Clock size={12} />
                      <span>{Math.ceil(job.eta_minutes)}m remaining</span>
                    </span>
                  </div>
                )}
              </div>

              {/* Quality Gate Status */}
              <div className="mt-3 p-2 bg-gray-50 rounded">
                <div className="text-xs text-gray-600 mb-1">Quality Gates:</div>
                <div className="flex space-x-4 text-sm">
                  <span className={job.best_r2 >= job.target_r2 ? 'text-green-600' : 'text-gray-500'}>
                    {job.best_r2 >= job.target_r2 ? '✅' : '⏳'} R² ≥ {job.target_r2}
                  </span>
                  <span className={(job.current_gap || job.final_gap || 1) <= 0.05 ? 'text-green-600' : 'text-gray-500'}>
                    {(job.current_gap || job.final_gap || 1) <= 0.05 ? '✅' : '⏳'} Gap ≤ 0.05
                  </span>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex space-x-2 mt-3">
                {job.status === 'running' && (
                  <button className="bg-red-600 text-white py-1 px-3 rounded text-sm hover:bg-red-700 transition-colors">
                    ⏹️ Stop
                  </button>
                )}
                {job.status === 'completed' && (
                  <button className="bg-green-600 text-white py-1 px-3 rounded text-sm hover:bg-green-700 transition-colors">
                    📥 Download Model
                  </button>
                )}
                <button className="bg-blue-600 text-white py-1 px-3 rounded text-sm hover:bg-blue-700 transition-colors">
                  📊 View Details
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Compute Status */}
      <div className="bg-gray-50 rounded-lg p-3 border">
        <h4 className="font-medium text-gray-700 mb-2">💻 Compute Resources</h4>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span>Cluster: gpu-cluster</span>
            <span className="text-green-600">✅ Available</span>
          </div>
          <div className="flex justify-between">
            <span>VM Size: Standard_NC6s_v3</span>
            <span className="text-blue-600">Tesla V100</span>
          </div>
          <div className="flex justify-between">
            <span>Active Nodes:</span>
            <span>1/4</span>
          </div>
          <div className="flex justify-between">
            <span>Est. Cost Today:</span>
            <span className="text-purple-600">€8.50</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TrainingStatusPanel;
