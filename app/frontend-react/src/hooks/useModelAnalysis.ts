import { useState, useEffect, useCallback } from 'react';

interface ModelDetails {
  model: string;
  timestamp: string;
  r2_train: number;
  r2_test: number;
  rmse_train: number;
  rmse_test: number;
  mae_train: number;
  mae_test: number;
  r2_gap: number;
  rmse_gap: number;
  category: string;
  interpretation: string;
  recommendation: string;
  color: string;
  n_features: number;
  is_perfect: boolean;
  experiment_name?: string;
}

interface ModelAnalysisData {
  models: ModelDetails[];
  categories: {
    [key: string]: {
      count: number;
      avg_r2_test: number;
      avg_rmse_test: number;
      color: string;
    };
  };
  performanceEvolution: Array<{
    date: string;
    avg_r2_test: number;
    avg_rmse_test: number;
    model_count: number;
  }>;
  bestModel: ModelDetails | null;
  totalModels: number;
  avgPerformance: {
    r2_test: number;
    rmse_test: number;
    mae_test: number;
  };
}

interface UseModelAnalysisOptions {
  autoRefresh?: boolean;
  refreshInterval?: number;
  filterCategory?: string;
  dateRange?: {
    start?: string;
    end?: string;
  };
}

const useModelAnalysis = (options: UseModelAnalysisOptions = {}) => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [lastRefresh, setLastRefresh] = useState(null);

  const {
    autoRefresh = false,
    refreshInterval = 30000, // 30 seconds
    filterCategory,
    dateRange
  } = options;

  // Build query parameters
  const buildQueryParams = useCallback(() => {
    const params = new URLSearchParams();
    if (filterCategory) params.append('category', filterCategory);
    if (dateRange?.start) params.append('start_date', dateRange.start);
    if (dateRange?.end) params.append('end_date', dateRange.end);
    return params.toString();
  }, [filterCategory, dateRange]);

  // Fetch analysis data
  const fetchAnalysisData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const queryParams = buildQueryParams();
      const url = `/api/models/analysis${queryParams ? `?${queryParams}` : ''}`;
      
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const analysisData = await response.json();
      setData(analysisData);
      setLastRefresh(new Date());
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Une erreur est survenue');
      console.error('Erreur lors du chargement des données:', err);
    } finally {
      setLoading(false);
    }
  }, [buildQueryParams]);

  // Fetch model categories
  const fetchCategories = useCallback(async () => {
    try {
      const response = await fetch('/api/models/categories');
      if (!response.ok) throw new Error('Erreur lors du chargement des catégories');
      return await response.json();
    } catch (err) {
      console.error('Erreur categories:', err);
      return {};
    }
  }, []);

  // Fetch performance evolution
  const fetchPerformanceEvolution = useCallback(async (days: number = 30) => {
    try {
      const response = await fetch(`/api/models/performance-evolution?days=${days}`);
      if (!response.ok) throw new Error('Erreur lors du chargement de l\'évolution');
      return await response.json();
    } catch (err) {
      console.error('Erreur evolution:', err);
      return [];
    }
  }, []);

  // Refresh data manually
  const refresh = useCallback(() => {
    fetchAnalysisData();
  }, [fetchAnalysisData]);

  // Get models by category
  const getModelsByCategory = useCallback((category: string) => {
    return data?.models.filter(model => model.category === category) || [];
  }, [data]);

  // Get best models (top 5 by R2 test)
  const getBestModels = useCallback((limit: number = 5) => {
    return data?.models
      .filter(model => !model.is_perfect) // Exclude potentially leaked models
      .sort((a, b) => b.r2_test - a.r2_test)
      .slice(0, limit) || [];
  }, [data]);

  // Get recent models
  const getRecentModels = useCallback((limit: number = 10) => {
    return data?.models
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, limit) || [];
  }, [data]);

  // Get model statistics
  const getModelStats = useCallback(() => {
    if (!data?.models.length) return null;

    const models = data.models;
    const validModels = models.filter(m => !m.is_perfect);

    return {
      total: models.length,
      valid: validModels.length,
      leaked: models.length - validModels.length,
      categories: Object.keys(data.categories).length,
      avgR2: validModels.reduce((sum, m) => sum + m.r2_test, 0) / validModels.length,
      avgRMSE: validModels.reduce((sum, m) => sum + m.rmse_test, 0) / validModels.length,
      bestR2: Math.max(...validModels.map(m => m.r2_test)),
      worstR2: Math.min(...validModels.map(m => m.r2_test))
    };
  }, [data]);

  // Filter models by performance
  const filterModelsByPerformance = useCallback((criteria: {
    minR2?: number;
    maxRMSE?: number;
    maxGap?: number;
  }) => {
    if (!data?.models) return [];

    return data.models.filter(model => {
      if (criteria.minR2 && model.r2_test < criteria.minR2) return false;
      if (criteria.maxRMSE && model.rmse_test > criteria.maxRMSE) return false;
      if (criteria.maxGap && model.r2_gap > criteria.maxGap) return false;
      return true;
    });
  }, [data]);

  // Search models
  const searchModels = useCallback((query: string) => {
    if (!data?.models || !query.trim()) return data?.models || [];

    const searchLower = query.toLowerCase();
    return data.models.filter(model =>
      model.model.toLowerCase().includes(searchLower) ||
      model.category.toLowerCase().includes(searchLower) ||
      model.experiment_name?.toLowerCase().includes(searchLower)
    );
  }, [data]);

  // Initial load
  useEffect(() => {
    fetchAnalysisData();
  }, [fetchAnalysisData]);

  // Auto refresh
  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(fetchAnalysisData, refreshInterval);
    return () => clearInterval(interval);
  }, [autoRefresh, refreshInterval, fetchAnalysisData]);

  return {
    // Data
    data,
    loading,
    error,
    lastRefresh,
    
    // Actions
    refresh,
    fetchCategories,
    fetchPerformanceEvolution,
    
    // Getters
    getModelsByCategory,
    getBestModels,
    getRecentModels,
    getModelStats,
    
    // Filters
    filterModelsByPerformance,
    searchModels,
    
    // Status
    isLoading: loading,
    hasError: !!error,
    hasData: !!data && data.models.length > 0,
    isEmpty: !!data && data.models.length === 0
  };
};

// Hook pour un modèle spécifique
const useModelDetails = (modelId: string) => {
  const [model, setModel] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchModelDetails = useCallback(async () => {
    if (!modelId) return;

    try {
      setLoading(true);
      setError(null);

      const response = await fetch(`/api/models/${encodeURIComponent(modelId)}`);
      
      if (!response.ok) {
        throw new Error(`Model not found: ${response.status}`);
      }
      
      const modelData = await response.json();
      setModel(modelData);
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erreur lors du chargement du modèle');
      console.error('Erreur model details:', err);
    } finally {
      setLoading(false);
    }
  }, [modelId]);

  useEffect(() => {
    fetchModelDetails();
  }, [fetchModelDetails]);

  return {
    model,
    loading,
    error,
    refresh: fetchModelDetails,
    isLoading: loading,
    hasError: !!error,
    notFound: !loading && !model && !error
  };
};

// Hook pour les comparaisons de modèles
const useModelComparison = (modelIds: string[]) => {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchModelsForComparison = useCallback(async () => {
    if (!modelIds.length) {
      setModels([]);
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const promises = modelIds.map(id => 
        fetch(`/api/models/${encodeURIComponent(id)}`).then(r => r.json())
      );
      
      const results = await Promise.all(promises);
      setModels(results.filter(Boolean)); // Remove any failed requests
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erreur lors de la comparaison');
      console.error('Erreur model comparison:', err);
    } finally {
      setLoading(false);
    }
  }, [modelIds]);

  // Generate comparison metrics
  const getComparisonMetrics = useCallback(() => {
    if (models.length < 2) return null;

    const metrics = ['r2_test', 'rmse_test', 'mae_test', 'r2_gap'] as const;
    
    return metrics.reduce((acc, metric) => {
      const values = models.map(m => m[metric]);
      acc[metric] = {
        best: Math.max(...values),
        worst: Math.min(...values),
        avg: values.reduce((sum, val) => sum + val, 0) / values.length,
        range: Math.max(...values) - Math.min(...values)
      };
      return acc;
    }, {} as Record<typeof metrics[number], {
      best: number;
      worst: number;
      avg: number;
      range: number;
    }>);
  }, [models]);

  useEffect(() => {
    fetchModelsForComparison();
  }, [fetchModelsForComparison]);

  return {
    models,
    loading,
    error,
    refresh: fetchModelsForComparison,
    comparisonMetrics: getComparisonMetrics(),
    canCompare: models.length >= 2,
    isLoading: loading,
    hasError: !!error
  };
};

export { useModelAnalysis, useModelDetails, useModelComparison };
export type { ModelDetails, ModelAnalysisData, UseModelAnalysisOptions };
