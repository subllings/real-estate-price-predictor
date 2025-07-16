/**
 * Hook React pour intégration automatique des métadonnées modèle
 * À utiliser dans votre application React
 */

import { useState, useEffect } from 'react';

// Types TypeScript
interface ModelMetadata {
  status: string;
  model_name: string;
  r2_test: number;
  mae_test: number;
  rmse_test: number;
  n_features: number;
  upload_timestamp: string;
  azure_url: string;
}

interface ModelHealth {
  status: 'healthy' | 'outdated' | 'unhealthy';
  checks: {
    model_file_exists: boolean;
    metadata_exists: boolean;
    model_age_hours: number;
    azure_connection: boolean;
  };
  timestamp: string;
}

// Hook principal pour les métadonnées du modèle
export const useModelMetadata = (apiBaseUrl: string = 'http://localhost:8000') => {
  const [metadata, setMetadata] = useState<ModelMetadata | null>(null);
  const [health, setHealth] = useState<ModelHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchModelInfo = async () => {
    try {
      setLoading(true);
      
      // Récupérer les métadonnées
      const metadataResponse = await fetch(`${apiBaseUrl}/model/info`);
      const metadataData = await metadataResponse.json();
      setMetadata(metadataData);

      // Récupérer l'état de santé
      const healthResponse = await fetch(`${apiBaseUrl}/model/health`);
      const healthData = await healthResponse.json();
      setHealth(healthData);
      
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erreur inconnue');
    } finally {
      setLoading(false);
    }
  };

  const forceSync = async () => {
    try {
      await fetch(`${apiBaseUrl}/model/sync`, { method: 'POST' });
      // Attendre un peu puis rafraîchir
      setTimeout(fetchModelInfo, 2000);
    } catch (err) {
      setError('Erreur lors de la synchronisation');
    }
  };

  useEffect(() => {
    fetchModelInfo();
    
    // Rafraîchir toutes les 5 minutes
    const interval = setInterval(fetchModelInfo, 5 * 60 * 1000);
    
    return () => clearInterval(interval);
  }, [apiBaseUrl]);

  return {
    metadata,
    health,
    loading,
    error,
    refresh: fetchModelInfo,
    forceSync
  };
};

// Composant Dashboard pour afficher les métriques
export const ModelDashboard: React.FC<{ apiBaseUrl?: string }> = ({ 
  apiBaseUrl = 'http://localhost:8000' 
}) => {
  const { metadata, health, loading, error, refresh, forceSync } = useModelMetadata(apiBaseUrl);

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        <span className="ml-2">Chargement modèle...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <h3 className="text-red-800 font-semibold">Erreur modèle</h3>
        <p className="text-red-600">{error}</p>
        <button 
          onClick={refresh}
          className="mt-2 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
        >
          Réessayer
        </button>
      </div>
    );
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'bg-green-100 text-green-800 border-green-200';
      case 'outdated': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'unhealthy': return 'bg-red-100 text-red-800 border-red-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold text-gray-800">🤖 Modèle ML Actuel</h2>
        <div className="flex gap-2">
          <button
            onClick={refresh}
            className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 text-sm"
          >
            🔄 Actualiser
          </button>
          <button
            onClick={forceSync}
            className="px-3 py-1 bg-purple-500 text-white rounded hover:bg-purple-600 text-sm"
          >
            ☁️ Sync Azure
          </button>
        </div>
      </div>

      {/* Statut de santé */}
      {health && (
        <div className={`border rounded-lg p-4 mb-6 ${getStatusColor(health.status)}`}>
          <div className="flex items-center justify-between">
            <span className="font-semibold">
              Statut: {health.status === 'healthy' ? '✅ Sain' : 
                       health.status === 'outdated' ? '⚠️ Obsolète' : '❌ Problème'}
            </span>
            <span className="text-sm">
              Âge: {health.checks.model_age_hours}h
            </span>
          </div>
        </div>
      )}

      {/* Métriques du modèle */}
      {metadata && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-blue-50 p-4 rounded-lg">
            <h3 className="text-sm font-semibold text-blue-800">R² Score</h3>
            <p className="text-2xl font-bold text-blue-600">
              {(metadata.r2_test * 100).toFixed(1)}%
            </p>
          </div>
          
          <div className="bg-green-50 p-4 rounded-lg">
            <h3 className="text-sm font-semibold text-green-800">RMSE</h3>
            <p className="text-2xl font-bold text-green-600">
              {Math.round(metadata.rmse_test).toLocaleString()}€
            </p>
          </div>
          
          <div className="bg-purple-50 p-4 rounded-lg">
            <h3 className="text-sm font-semibold text-purple-800">MAE</h3>
            <p className="text-2xl font-bold text-purple-600">
              {Math.round(metadata.mae_test).toLocaleString()}€
            </p>
          </div>
          
          <div className="bg-orange-50 p-4 rounded-lg">
            <h3 className="text-sm font-semibold text-orange-800">Features</h3>
            <p className="text-2xl font-bold text-orange-600">
              {metadata.n_features}
            </p>
          </div>
        </div>
      )}

      {/* Informations détaillées */}
      {metadata && (
        <div className="mt-6 border-t pt-4">
          <h3 className="font-semibold text-gray-700 mb-2">Détails</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <span className="font-medium">Modèle:</span> {metadata.model_name}
            </div>
            <div>
              <span className="font-medium">Source:</span> {
                metadata.azure_url.includes('azure') ? '☁️ Azure' : '💻 Local'
              }
            </div>
            <div>
              <span className="font-medium">Dernière MAJ:</span> {
                new Date(metadata.upload_timestamp).toLocaleString('fr-FR')
              }
            </div>
            <div>
              <span className="font-medium">Statut:</span> {
                metadata.status === 'active' ? '✅ Actif' : '⚠️ Inactif'
              }
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

// Hook pour utiliser le modèle dans les prédictions
export const usePrediction = (apiBaseUrl: string = 'http://localhost:8000') => {
  const [predicting, setPredicting] = useState(false);
  const [lastPrediction, setLastPrediction] = useState<any>(null);

  const predict = async (houseData: any) => {
    setPredicting(true);
    try {
      const response = await fetch(`${apiBaseUrl}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(houseData)
      });
      
      const result = await response.json();
      setLastPrediction(result);
      return result;
    } catch (error) {
      console.error('Erreur prédiction:', error);
      throw error;
    } finally {
      setPredicting(false);
    }
  };

  return { predict, predicting, lastPrediction };
};

export default ModelDashboard;
