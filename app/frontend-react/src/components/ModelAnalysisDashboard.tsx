import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ScatterChart, Scatter, LineChart, Line, PieChart, Pie, Cell } from 'recharts';

// Types pour les modèles et métriques
interface ModelAnalysis {
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
}

interface ModelMetrics {
  total_models: number;
  best_r2: number;
  mean_r2: number;
  best_model: string;
  production_ready_count: number;
  models_summary: ModelAnalysis[];
}

// Hook pour récupérer les données des modèles
const useModelAnalysis = (apiBaseUrl: string = 'http://localhost:8000') => {
  const [models, setModels] = useState<ModelAnalysis[]>([]);
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchModelData = async () => {
    try {
      setLoading(true);
      
      // Récupérer les données depuis l'API ou fichier JSON
      const response = await fetch(`${apiBaseUrl}/reports/dashboard_summary.json`);
      const data = await response.json();
      
      // Analyser les modèles pour ajouter les catégories
      const analyzedModels = data.models_summary.map((model: any) => {
        const r2Gap = model.r2_train - model.r2_test;
        const analysis = analyzeGeneralization(model.r2_train, model.r2_test, model.rmse_train, model.rmse_test);
        
        return {
          ...model,
          r2_gap: r2Gap,
          rmse_gap: model.rmse_test - model.rmse_train,
          ...analysis
        };
      });
      
      setModels(analyzedModels);
      setMetrics(data);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Erreur de chargement');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModelData();
    // Actualiser toutes les 30 secondes
    const interval = setInterval(fetchModelData, 30000);
    return () => clearInterval(interval);
  }, [apiBaseUrl]);

  return { models, metrics, loading, error, refresh: fetchModelData };
};

// Fonction d'analyse de généralisation
const analyzeGeneralization = (r2Train: number, r2Test: number, rmseTrain: number, rmseTest: number) => {
  const r2Gap = r2Train - r2Test;
  
  if (r2Test < 0.5) {
    return {
      category: "Underfitting",
      color: "#ff6b6b",
      interpretation: "Modèle trop simple, performances insuffisantes",
      recommendation: "Augmenter la complexité, plus de features"
    };
  } else if (r2Gap > 0.15) {
    return {
      category: "Strong overfitting",
      color: "#ff9f43",
      interpretation: "Modèle mémorise les données d'entraînement",
      recommendation: "Réduire complexité, régularisation"
    };
  } else if (r2Gap > 0.08) {
    return {
      category: "Moderate overfitting",
      color: "#feca57",
      interpretation: "Léger surapprentissage, acceptable",
      recommendation: "Surveiller, possible régularisation légère"
    };
  } else if (r2Gap < 0.02 && r2Test > 0.7) {
    return {
      category: "Good generalization",
      color: "#48dbfb",
      interpretation: "Excellent équilibre train/test",
      recommendation: "Modèle optimal, prêt pour production"
    };
  } else if (r2Test > 0.6 && r2Gap < 0.05) {
    return {
      category: "Light overfitting",
      color: "#0be881",
      interpretation: "Bon modèle avec généralisation correcte",
      recommendation: "Acceptable pour production"
    };
  } else {
    return {
      category: "Moderate underfitting",
      color: "#a55eea",
      interpretation: "Performances moyennes, marge d'amélioration",
      recommendation: "Optimiser features et hyperparameters"
    };
  }
};

// Composant principal du dashboard
const ModelAnalysisDashboard: React.FC<{ apiBaseUrl?: string }> = ({ 
  apiBaseUrl = 'http://localhost:8000' 
}) => {
  const { models, metrics, loading, error, refresh } = useModelAnalysis(apiBaseUrl);
  const [selectedCategory, setSelectedCategory] = useState<string>('all');
  const [sortBy, setSortBy] = useState<'r2_test' | 'timestamp' | 'r2_gap'>('r2_test');

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500 mx-auto"></div>
          <p className="mt-4 text-lg">Chargement de l'analyse des modèles...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-red-50 flex items-center justify-center">
        <div className="text-center p-8">
          <h2 className="text-2xl font-bold text-red-800 mb-4">Erreur de chargement</h2>
          <p className="text-red-600 mb-4">{error}</p>
          <button 
            onClick={refresh}
            className="px-6 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
          >
            Réessayer
          </button>
        </div>
      </div>
    );
  }

  // Filtrer les modèles selon la catégorie sélectionnée
  const filteredModels = selectedCategory === 'all' 
    ? models 
    : models.filter(model => model.category === selectedCategory);

  // Trier les modèles
  const sortedModels = [...filteredModels].sort((a, b) => {
    if (sortBy === 'timestamp') {
      return new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime();
    }
    return b[sortBy] - a[sortBy];
  });

  // Données pour les graphiques
  const categoryData = Object.entries(
    models.reduce((acc, model) => {
      acc[model.category] = (acc[model.category] || 0) + 1;
      return acc;
    }, {} as Record<string, number>)
  ).map(([name, value]) => ({ name, value }));

  const performanceData = models.map(model => ({
    name: model.model.substring(0, 15) + '...',
    r2_test: model.r2_test,
    r2_gap: model.r2_gap,
    category: model.category,
    color: model.color
  }));

  const colors = {
    "Good generalization": "#48dbfb",
    "Light overfitting": "#0be881",
    "Moderate overfitting": "#feca57",
    "Strong overfitting": "#ff9f43",
    "Underfitting": "#ff6b6b",
    "Moderate underfitting": "#a55eea"
  };

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      {/* Header */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-800">🤖 Analyse des Modèles ML</h1>
            <p className="text-gray-600 mt-2">Dashboard complet de performance et généralisation</p>
          </div>
          <button
            onClick={refresh}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center gap-2"
          >
            🔄 Actualiser
          </button>
        </div>
        
        {/* Métriques globales */}
        {metrics && (
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mt-6">
            <div className="bg-blue-50 p-4 rounded-lg text-center">
              <h3 className="text-sm font-semibold text-blue-800">Total Modèles</h3>
              <p className="text-2xl font-bold text-blue-600">{metrics.total_models}</p>
            </div>
            <div className="bg-green-50 p-4 rounded-lg text-center">
              <h3 className="text-sm font-semibold text-green-800">Meilleur R²</h3>
              <p className="text-2xl font-bold text-green-600">{(metrics.best_r2 * 100).toFixed(1)}%</p>
            </div>
            <div className="bg-purple-50 p-4 rounded-lg text-center">
              <h3 className="text-sm font-semibold text-purple-800">R² Moyen</h3>
              <p className="text-2xl font-bold text-purple-600">{(metrics.mean_r2 * 100).toFixed(1)}%</p>
            </div>
            <div className="bg-orange-50 p-4 rounded-lg text-center">
              <h3 className="text-sm font-semibold text-orange-800">Production Ready</h3>
              <p className="text-2xl font-bold text-orange-600">{metrics.production_ready_count}</p>
            </div>
            <div className="bg-gray-50 p-4 rounded-lg text-center">
              <h3 className="text-sm font-semibold text-gray-800">Meilleur Modèle</h3>
              <p className="text-xs font-bold text-gray-600 truncate">{metrics.best_model}</p>
            </div>
          </div>
        )}
      </div>

      {/* Filtres et tri */}
      <div className="bg-white rounded-lg shadow-lg p-4 mb-6">
        <div className="flex flex-wrap gap-4 items-center">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Catégorie</label>
            <select
              value={selectedCategory}
              onChange={(e) => setSelectedCategory(e.target.value)}
              className="border border-gray-300 rounded-md px-3 py-2"
            >
              <option value="all">Toutes les catégories</option>
              {Object.keys(colors).map(category => (
                <option key={category} value={category}>{category}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Trier par</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="border border-gray-300 rounded-md px-3 py-2"
            >
              <option value="r2_test">R² Test</option>
              <option value="timestamp">Date</option>
              <option value="r2_gap">Gap R²</option>
            </select>
          </div>
          <div className="text-sm text-gray-600">
            {filteredModels.length} modèle(s) affiché(s)
          </div>
        </div>
      </div>

      {/* Graphiques */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        {/* Distribution des catégories */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold mb-4">📊 Distribution des Catégories</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie
                data={categoryData}
                cx="50%"
                cy="50%"
                outerRadius={80}
                dataKey="value"
                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
              >
                {categoryData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={colors[entry.name as keyof typeof colors] || '#8884d8'} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Performance vs Gap */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold mb-4">🎯 Performance vs Généralisation</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ScatterChart data={performanceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="r2_test" domain={[0, 1]} label={{ value: 'R² Test', position: 'insideBottom', offset: -5 }} />
              <YAxis dataKey="r2_gap" label={{ value: 'R² Gap', angle: -90, position: 'insideLeft' }} />
              <Tooltip 
                formatter={(value, name) => [typeof value === 'number' ? value.toFixed(3) : value, name]}
                labelFormatter={(label) => `Modèle: ${label}`}
              />
              <Scatter dataKey="r2_gap" fill="#8884d8">
                {performanceData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Évolution temporelle */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <h3 className="text-lg font-semibold mb-4">📈 Évolution des Performances</h3>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={models.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis 
              dataKey="timestamp" 
              tickFormatter={(value) => new Date(value).toLocaleDateString()}
            />
            <YAxis domain={[0, 1]} />
            <Tooltip 
              labelFormatter={(value) => new Date(value).toLocaleString()}
              formatter={(value: any) => [(value * 100).toFixed(1) + '%', 'R² Test']}
            />
            <Line type="monotone" dataKey="r2_test" stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Table des modèles */}
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        <div className="p-6 border-b">
          <h3 className="text-lg font-semibold">📋 Détail des Modèles</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Modèle</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">R² Test</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Gap R²</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">RMSE</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Catégorie</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Date</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Recommandation</th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {sortedModels.map((model, index) => (
                <tr key={index} className={index % 2 === 0 ? 'bg-white' : 'bg-gray-50'}>
                  <td className="px-6 py-4 text-sm font-medium text-gray-900 max-w-xs truncate">
                    {model.model}
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-900">
                    <span className={`font-semibold ${model.r2_test > 0.8 ? 'text-green-600' : model.r2_test > 0.6 ? 'text-yellow-600' : 'text-red-600'}`}>
                      {(model.r2_test * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-900">
                    <span className={`font-semibold ${model.r2_gap < 0.05 ? 'text-green-600' : model.r2_gap < 0.1 ? 'text-yellow-600' : 'text-red-600'}`}>
                      {model.r2_gap.toFixed(3)}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-900">
                    {Math.round(model.rmse_test).toLocaleString()}€
                  </td>
                  <td className="px-6 py-4 text-sm">
                    <span 
                      className="inline-flex px-2 py-1 text-xs font-semibold rounded-full text-white"
                      style={{ backgroundColor: model.color }}
                    >
                      {model.category}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-900">
                    {new Date(model.timestamp).toLocaleDateString()}
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-700 max-w-xs">
                    <div title={model.recommendation} className="truncate">
                      {model.recommendation}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default ModelAnalysisDashboard;
