import React from 'react';
import './TrainingJobCard.css';

const TrainingJobCard = ({ job, onStop, showMachine = true }) => {
  const formatDate = (dateString) => {
    if (!dateString) return 'N/A';
    return new Date(dateString).toLocaleString('en-GB', {
      day: '2-digit',
      month: '2-digit',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatDuration = (startTime) => {
    const now = new Date();
    const start = new Date(startTime);
    const diffMs = now - start;
    const diffMins = Math.floor(diffMs / (1000 * 60));
    
    if (diffMins < 60) return `${diffMins} min`;
    const hours = Math.floor(diffMins / 60);
    const mins = diffMins % 60;
    return `${hours}h ${mins}min`;
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'running': return '#28a745';
      case 'queued': return '#ffc107';
      case 'completed': return '#17a2b8';
      case 'failed': return '#dc3545';
      case 'stopped': return '#6c757d';
      default: return '#6c757d';
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'running': return 'Running';
      case 'queued': return 'Queued';
      case 'completed': return 'Completed';
      case 'failed': return 'Failed';
      case 'stopped': return 'Stopped';
      default: return status;
    }
  };

  const generalizationScore = Math.max(0, Math.min(100, 100 - (job.current_gap * 1000)));
  
  let generalizationLabel = 'Excellent';
  if (generalizationScore < 30) generalizationLabel = 'Poor';
  else if (generalizationScore < 50) generalizationLabel = 'Fair';
  else if (generalizationScore < 70) generalizationLabel = 'Good';
  else if (generalizationScore < 90) generalizationLabel = 'Very Good';

  return (
    <div className="training-job-card">
      <div className="job-header">
        <div className="job-title-section">
          <h3 className="job-name">{job.name}</h3>
          <span 
            className="job-status" 
            style={{ backgroundColor: getStatusColor(job.status) }}
          >
            {getStatusText(job.status)}
          </span>
          {showMachine && (
            <span className="job-machine">
              <strong>Machine: {job.machine_name}</strong>
            </span>
          )}
        </div>
        
        <div className="job-actions">
          {job.status === 'running' && (
            <button 
              className="stop-btn" 
              onClick={() => onStop?.(job.id)}
              title="Stop training"
            >
              Stop
            </button>
          )}
        </div>
      </div>

      <div className="job-progress-section">
        <div className="progress-info">
          <span>Progress: {job.progress.toFixed(1)}%</span>
          {job.eta_minutes > 0 && (
            <span>ETA: {Math.ceil(job.eta_minutes)} min</span>
          )}
        </div>
        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${job.progress}%` }}
          />
        </div>
      </div>

      <div className="job-metrics">
        <div className="metrics-row">
          <div className="metric">
            <span className="metric-label">Trials:</span>
            <span className="metric-value">{job.current_trial}/{job.total_trials}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Best R²:</span>
            <span className="metric-value">{job.best_r2.toFixed(4)}</span>
          </div>
          <div className="metric">
            <span className="metric-label">Target R²:</span>
            <span className="metric-value">{job.target_r2.toFixed(3)}</span>
          </div>
        </div>

        <div className="metrics-row">
          <div className="metric">
            <span className="metric-label">Train/Val Gap:</span>
            <span className="metric-value">{(job.current_gap * 100).toFixed(2)}%</span>
          </div>
          <div className="metric">
            <span className="metric-label">Generalization Score:</span>
            <span className="metric-value">
              {generalizationScore.toFixed(1)} ({generalizationLabel})
            </span>
          </div>
        </div>
      </div>

      <div className="job-details">
        {!showMachine && (
          <div className="detail-row">
            <span className="detail-label">Machine:</span>
            <span className="detail-value"><strong>{job.machine_name}</strong></span>
          </div>
        )}
        <div className="detail-row">
          <span className="detail-label">Compute:</span>
          <span className="detail-value">{job.compute_target}</span>
        </div>
        <div className="detail-row">
          <span className="detail-label">Type:</span>
          <span className="detail-value">{job.model_type.toUpperCase()}</span>
        </div>
        <div className="detail-row">
          <span className="detail-label">Started:</span>
          <span className="detail-value">{formatDate(job.started_at)}</span>
        </div>
        <div className="detail-row">
          <span className="detail-label">Duration:</span>
          <span className="detail-value">{formatDuration(job.started_at)}</span>
        </div>
        {job.completed_at && (
          <div className="detail-row">
            <span className="detail-label">Completed:</span>
            <span className="detail-value">{formatDate(job.completed_at)}</span>
          </div>
        )}
      </div>

      {job.hyperparameters && (
        <div className="job-hyperparams">
          <h4>Hyperparameters:</h4>
          <div className="hyperparams-grid">
            {Object.entries(job.hyperparameters).map(([key, value]) => (
              <div key={key} className="hyperparam">
                <span className="hyperparam-key">{key}:</span>
                <span className="hyperparam-value">{value}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default TrainingJobCard;
