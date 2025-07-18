import { useState, useEffect } from 'react';

const API_BASE_URL = 'http://127.0.0.1:8000';

export const useExperiments = () => {
  const [experiments, setExperiments] = useState([]);
  const [summary, setSummary] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchExperiments = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/experiments`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setExperiments(data.experiments || []);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching experiments:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchSummary = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/experiments/summary`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setSummary(data);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching summary:', err);
    }
  };

  const refresh = () => {
    fetchExperiments();
    fetchSummary();
  };

  useEffect(() => {
    fetchExperiments();
    fetchSummary();
  }, []);

  return {
    experiments,
    summary,
    loading,
    error,
    refresh
  };
};

export const useExperimentDetails = (experimentId) => {
  const [experiment, setExperiment] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchExperiment = async () => {
    if (!experimentId) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/experiments/${experimentId}`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setExperiment(data);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching experiment details:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchExperiment();
  }, [experimentId]);

  return {
    experiment,
    loading,
    error,
    refetch: fetchExperiment
  };
};
