// Base API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8002';

// Helper function for API calls
const apiCall = async (endpoint, options = {}) => {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers
    },
    ...options
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  return response.json();
};

// Tuner Agent API endpoints
export const tunerApi = {
  // Lancer un tuner agent
  startTuner: async (config) => {
    return apiCall('/api/tuner/start', {
      method: 'POST',
      body: JSON.stringify(config)
    });
  },

  // Récupérer le statut d'un job
  getTunerStatus: async (jobId) => {
    return apiCall(`/api/tuner/status/${jobId}`);
  },

  // Lister tous les jobs tuner
  listTunerJobs: async () => {
    return apiCall('/api/tuner/list');
  }
};

// Training Jobs API (existing functionality only - local training via bash scripts)
export const trainingApi = {
  // Get all training jobs
  getTrainingJobs: async () => {
    return apiCall('/api/training/jobs');
  },

  // Start new training job
  startTraining: async (config) => {
    return apiCall('/api/training/start', {
      method: 'POST',
      body: JSON.stringify(config)
    });
  },

  // Stop training job
  stopTraining: async (jobId) => {
    return apiCall(`/api/training/stop/${jobId}`, {
      method: 'POST'
    });
  },

  // Get training job status
  getTrainingStatus: async (jobId) => {
    return apiCall(`/api/training/status/${jobId}`);
  }
};

// Experiments API 
export const experimentsApi = {
  // Get all experiments
  getExperiments: async () => {
    return apiCall('/api/experiments');
  },

  // Get experiment by ID
  getExperiment: async (experimentId) => {
    return apiCall(`/api/experiments/${experimentId}`);
  }
};

// Models API
export const modelsApi = {
  // Get available models
  getModels: async () => {
    return apiCall('/api/models');
  },

  // Get model details
  getModel: async (modelId) => {
    return apiCall(`/api/models/${modelId}`);
  }
};

export default {
  tunerApi,
  trainingApi,
  experimentsApi,
  modelsApi
};
