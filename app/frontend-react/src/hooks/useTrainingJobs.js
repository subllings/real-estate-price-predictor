import { useState, useEffect } from 'react';
import { PREDICTION_API_URL } from '../config/api';

const useTrainingJobs = () => {
  const [trainingJobs, setTrainingJobs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchTrainingJobs = async () => {
    try {
      setLoading(true);
      setError(null);
      
      // Pour l'instant, on utilise des données de demo
      // À terme, cela interrogera l'API backend qui lira le container TrainingJobs dans Cosmos DB
      const response = await fetch(`${PREDICTION_API_URL}/training-jobs`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch training jobs');
      }
      
      const data = await response.json();
      setTrainingJobs(data.training_jobs || []);
      
    } catch (err) {
      console.error('Error fetching training jobs:', err);
      
      // Données de fallback pour demo
      setTrainingJobs([
        {
          id: 'catboost-opt-001',
          name: 'CatBoost Hyperparameter Optimization',
          status: 'running',
          progress: 78.5,
          eta_minutes: 7,
          current_trial: 39,
          total_trials: 50,
          best_r2: 0.8512,
          target_r2: 0.85,
          current_gap: 0.0234,
          compute_target: 'Desktop-Intel-i7',
          machine_name: 'LAPTOP-DEV-01',
          started_at: new Date(Date.now() - 25 * 60 * 1000).toISOString(), // 25 min ago
          model_type: 'catboost',
          hyperparameters: {
            learning_rate: 0.1,
            depth: 8,
            iterations: 1000
          }
        },
        {
          id: 'catboost-distributed-002',
          name: 'Distributed CatBoost Training',
          status: 'running',
          progress: 45.2,
          eta_minutes: 12,
          current_trial: 23,
          total_trials: 75,
          best_r2: 0.8387,
          target_r2: 0.85,
          current_gap: 0.0456,
          compute_target: 'Azure-ML-Cluster',
          machine_name: 'gpu-cluster-node-2',
          started_at: new Date(Date.now() - 18 * 60 * 1000).toISOString(), // 18 min ago
          model_type: 'catboost',
          hyperparameters: {
            learning_rate: 0.08,
            depth: 10,
            iterations: 1500
          }
        },
        {
          id: 'xgboost-weekend-003',
          name: 'Weekend XGBoost Experiment',
          status: 'completed',
          progress: 100,
          eta_minutes: 0,
          current_trial: 100,
          total_trials: 100,
          best_r2: 0.8467,
          target_r2: 0.85,
          final_gap: 0.0298,
          compute_target: 'Desktop-RTX-3080',
          machine_name: 'DESKTOP-ML-02',
          started_at: new Date(Date.now() - 125 * 60 * 1000).toISOString(), // 2h ago
          completed_at: new Date(Date.now() - 15 * 60 * 1000).toISOString(), // 15 min ago
          model_type: 'xgboost'
        }
      ]);
      
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const startNewTraining = async (config = {}) => {
    try {
      const response = await fetch(`${PREDICTION_API_URL}/training-jobs/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_type: config.model_type || 'catboost',
          target_r2: config.target_r2 || 0.85,
          max_trials: config.max_trials || 50,
          compute_target: config.compute_target || 'local',
          machine_preference: config.machine_preference || 'auto'
        })
      });

      if (response.ok) {
        // Déterminer le nom de machine basé sur les préférences
        let machineName = 'CURRENT-MACHINE';
        if (config.machine_preference !== 'auto') {
          machineName = config.machine_preference;
        } else {
          // Auto-sélection basée sur le compute target
          switch (config.compute_target) {
            case 'azure-ml':
              machineName = 'AZURE-COMPUTE-' + Math.floor(Math.random() * 10).toString().padStart(2, '0');
              break;
            case 'desktop-cluster':
              machineName = 'DESKTOP-ML-' + Math.floor(Math.random() * 5).toString().padStart(2, '0');
              break;
            case 'laptop-cluster':
              machineName = 'LAPTOP-DEV-' + Math.floor(Math.random() * 3).toString().padStart(2, '0');
              break;
            default:
              machineName = window.navigator.userAgent.includes('Windows') ? 'WINDOWS-LOCAL' : 'LOCAL-MACHINE';
          }
        }

        const newJob = {
          id: `training-${Date.now()}`,
          name: `${(config.model_type || 'CatBoost').toUpperCase()} Training Session`,
          status: 'queued',
          progress: 0,
          eta_minutes: Math.floor(15 + Math.random() * 10), // 15-25 min
          current_trial: 0,
          total_trials: config.max_trials || 50,
          best_r2: 0.0,
          target_r2: config.target_r2 || 0.85,
          current_gap: 0.0,
          compute_target: config.compute_target || 'local',
          machine_name: machineName,
          started_at: new Date().toISOString(),
          model_type: config.model_type || 'catboost',
          hyperparameters: {
            learning_rate: 0.05 + Math.random() * 0.15,
            depth: Math.floor(6 + Math.random() * 8),
            iterations: Math.floor(800 + Math.random() * 400)
          }
        };
        
        setTrainingJobs(prev => [newJob, ...prev]);
        
        // Démarrer la simulation après 2 secondes (temps de démarrage)
        setTimeout(() => {
          setTrainingJobs(prev => 
            prev.map(job => 
              job.id === newJob.id 
                ? { ...job, status: 'running', progress: 1 }
                : job
            )
          );
        }, 2000);
        
        return { success: true, job: newJob };
      } else {
        throw new Error('Failed to start training job');
      }
    } catch (error) {
      console.error('Failed to start training:', error);
      return { success: false, error: error.message };
    }
  };

  const stopTrainingJob = async (jobId) => {
    try {
      const response = await fetch(`${PREDICTION_API_URL}/training-jobs/${jobId}/stop`, {
        method: 'POST'
      });

      if (response.ok) {
        setTrainingJobs(prev => 
          prev.map(job => 
            job.id === jobId 
              ? { ...job, status: 'stopped', completed_at: new Date().toISOString() }
              : job
          )
        );
        return { success: true };
      } else {
        throw new Error('Failed to stop training job');
      }
    } catch (error) {
      console.error('Failed to stop training:', error);
      return { success: false, error: error.message };
    }
  };

  useEffect(() => {
    fetchTrainingJobs();
    
    // Actualiser toutes les 5 secondes pour les jobs en cours
    const interval = setInterval(() => {
      // Simuler les mises à jour de progression pour la demo
      setTrainingJobs(prev => 
        prev.map(job => {
          if (job.status === 'running' && job.progress < 100) {
            const newProgress = Math.min(100, job.progress + Math.random() * 2);
            const newTrial = Math.min(job.total_trials, job.current_trial + (Math.random() < 0.3 ? 1 : 0));
            const newEta = Math.max(0, job.eta_minutes - 0.1);
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
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  return {
    trainingJobs,
    loading,
    error,
    refresh: fetchTrainingJobs,
    startNewTraining,
    stopTrainingJob
  };
};

export default useTrainingJobs;
