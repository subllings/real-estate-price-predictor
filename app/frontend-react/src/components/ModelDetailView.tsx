import React, { useState } from 'react';
// Note: Install recharts with: npm install recharts @types/recharts
// import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

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

interface ModelDetailViewProps {
  model: ModelDetails;
  onClose: () => void;
}

const ModelDetailView = ({ model, onClose }) => {
  const [activeTab, setActiveTab] = useState('overview');

  // Données pour le graphique de comparaison Train vs Test
  const comparisonData = [
    {
      metric: 'R²',
      train: model.r2_train,
      test: model.r2_test,
      unit: '%',
      ideal: 0.8
    },
    {
      metric: 'RMSE',
      train: model.rmse_train,
      test: model.rmse_test,
      unit: '€',
      ideal: 50000
    },
    {
      metric: 'MAE', 
      train: model.mae_train,
      test: model.mae_test,
      unit: '€',
      ideal: 35000
    }
  ];

  // Données pour le radar chart de performance
  const radarData = [
    {
      metric: 'Précision',
      value: model.r2_test * 100,
      fullMark: 100
    },
    {
      metric: 'Généralisation',
      value: Math.max(0, 100 - (model.r2_gap * 1000)),
      fullMark: 100
    },
    {
      metric: 'Stabilité',
      value: Math.max(0, 100 - Math.abs(model.rmse_gap) / 1000),
      fullMark: 100
    },
    {
      metric: 'Efficacité',
      value: Math.max(0, 100 - model.rmse_test / 1000),
      fullMark: 100
    }
  ];

  const getStatusIcon = (category: string) => {
    const icons = {
      "Good generalization": "🎯",
      "Light overfitting": "⚡",
      "Moderate overfitting": "⚠️",
      "Strong overfitting": "🔴",
      "Underfitting": "📉",
      "Moderate underfitting": "📊"
    };
    return icons[category as keyof typeof icons] || "🤖";
  };

  const getScoreColor = (score: number, metric: string) => {
    if (metric === 'R²') {
      if (score > 0.8) return 'text-green-600';
      if (score > 0.6) return 'text-yellow-600';
      return 'text-red-600';
    } else {
      if (score < 40000) return 'text-green-600';
      if (score < 60000) return 'text-yellow-600';
      return 'text-red-600';
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-2xl max-w-6xl w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="p-6 border-b bg-gradient-to-r from-blue-600 to-purple-600 text-white rounded-t-lg">
          <div className="flex justify-between items-center">
            <div>
              <h2 className="text-2xl font-bold flex items-center gap-2">
                {getStatusIcon(model.category)} Détail du Modèle
              </h2>
              <p className="text-blue-100 mt-1 font-mono text-sm truncate max-w-md">
                {model.model}
              </p>
            </div>
            <button
              onClick={onClose}
              className="text-white hover:text-gray-200 text-2xl font-bold"
            >
              ✕
            </button>
          </div>
          
          {/* Status badge */}
          <div className="mt-4">
            <span 
              className="inline-flex px-4 py-2 rounded-full text-sm font-semibold text-white shadow-lg"
              style={{ backgroundColor: model.color }}
            >
              {model.category}
            </span>
          </div>
        </div>

        {/* Navigation tabs */}
        <div className="border-b">
          <nav className="flex space-x-8 px-6">
            {[
              { id: 'overview', label: '📊 Vue d\'ensemble', icon: '📊' },
              { id: 'metrics', label: '📈 Métriques', icon: '📈' },
              { id: 'analysis', label: '🔍 Analyse', icon: '🔍' },
              { id: 'recommendations', label: '💡 Recommandations', icon: '💡' }
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={`py-4 px-2 border-b-2 font-medium text-sm ${
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

        {/* Content */}
        <div className="p-6">
          {activeTab === 'overview' && (
            <div className="space-y-6">
              {/* Métriques principales */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-blue-50 p-4 rounded-lg border">
                  <h4 className="text-sm font-semibold text-blue-800">R² Test</h4>
                  <p className={`text-2xl font-bold ${getScoreColor(model.r2_test, 'R²')}`}>
                    {(model.r2_test * 100).toFixed(1)}%
                  </p>
                </div>
                <div className="bg-green-50 p-4 rounded-lg border">
                  <h4 className="text-sm font-semibold text-green-800">RMSE Test</h4>
                  <p className={`text-2xl font-bold ${getScoreColor(model.rmse_test, 'RMSE')}`}>
                    {Math.round(model.rmse_test).toLocaleString()}€
                  </p>
                </div>
                <div className="bg-purple-50 p-4 rounded-lg border">
                  <h4 className="text-sm font-semibold text-purple-800">Gap R²</h4>
                  <p className={`text-2xl font-bold ${model.r2_gap < 0.05 ? 'text-green-600' : model.r2_gap < 0.1 ? 'text-yellow-600' : 'text-red-600'}`}>
                    {model.r2_gap.toFixed(3)}
                  </p>
                </div>
                <div className="bg-orange-50 p-4 rounded-lg border">
                  <h4 className="text-sm font-semibold text-orange-800">Features</h4>
                  <p className="text-2xl font-bold text-orange-600">
                    {model.n_features}
                  </p>
                </div>
              </div>

              {/* Graphique radar de performance */}
              <div className="bg-gray-50 p-6 rounded-lg">
                <h4 className="text-lg font-semibold mb-4">🎯 Performance Globale</h4>
                <ResponsiveContainer width="100%" height={300}>
                  <RadarChart data={radarData}>
                    <PolarGrid />
                    <PolarAngleAxis dataKey="metric" />
                    <PolarRadiusAxis domain={[0, 100]} />
                    <Radar 
                      name="Performance" 
                      dataKey="value" 
                      stroke="#3b82f6" 
                      fill="#3b82f6" 
                      fillOpacity={0.3}
                      strokeWidth={2}
                    />
                    <Tooltip formatter={(value) => [`${Number(value).toFixed(1)}%`, 'Score']} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {activeTab === 'metrics' && (
            <div className="space-y-6">
              {/* Comparaison Train vs Test */}
              <div className="bg-gray-50 p-6 rounded-lg">
                <h4 className="text-lg font-semibold mb-4">📊 Comparaison Train vs Test</h4>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={comparisonData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="metric" />
                    <YAxis />
                    <Tooltip 
                      formatter={(value: any, name) => [
                        typeof value === 'number' ? 
                          (name === 'train' || name === 'test' ? 
                            (value < 1 ? (value * 100).toFixed(1) + '%' : Math.round(value).toLocaleString() + '€')
                            : value
                          ) : value,
                        name === 'train' ? 'Entraînement' : name === 'test' ? 'Test' : name
                      ]}
                    />
                    <Bar dataKey="train" fill="#3b82f6" name="train" />
                    <Bar dataKey="test" fill="#10b981" name="test" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Tableau détaillé des métriques */}
              <div className="bg-white border rounded-lg overflow-hidden">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Métrique</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Entraînement</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Test</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Gap</th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Évaluation</th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    <tr>
                      <td className="px-6 py-4 text-sm font-medium text-gray-900">R² Score</td>
                      <td className="px-6 py-4 text-sm text-gray-900">{(model.r2_train * 100).toFixed(1)}%</td>
                      <td className="px-6 py-4 text-sm text-gray-900">{(model.r2_test * 100).toFixed(1)}%</td>
                      <td className="px-6 py-4 text-sm text-gray-900">{model.r2_gap.toFixed(3)}</td>
                      <td className="px-6 py-4 text-sm">
                        <span className={`px-2 py-1 rounded-full text-xs ${
                          model.r2_gap < 0.05 ? 'bg-green-100 text-green-800' :
                          model.r2_gap < 0.1 ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {model.r2_gap < 0.05 ? 'Excellent' : model.r2_gap < 0.1 ? 'Bon' : 'Problématique'}
                        </span>
                      </td>
                    </tr>
                    <tr>
                      <td className="px-6 py-4 text-sm font-medium text-gray-900">RMSE</td>
                      <td className="px-6 py-4 text-sm text-gray-900">{Math.round(model.rmse_train).toLocaleString()}€</td>
                      <td className="px-6 py-4 text-sm text-gray-900">{Math.round(model.rmse_test).toLocaleString()}€</td>
                      <td className="px-6 py-4 text-sm text-gray-900">{Math.round(model.rmse_gap).toLocaleString()}€</td>
                      <td className="px-6 py-4 text-sm">
                        <span className={`px-2 py-1 rounded-full text-xs ${
                          model.rmse_test < 40000 ? 'bg-green-100 text-green-800' :
                          model.rmse_test < 60000 ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {model.rmse_test < 40000 ? 'Excellent' : model.rmse_test < 60000 ? 'Acceptable' : 'Élevé'}
                        </span>
                      </td>
                    </tr>
                    <tr>
                      <td className="px-6 py-4 text-sm font-medium text-gray-900">MAE</td>
                      <td className="px-6 py-4 text-sm text-gray-900">{Math.round(model.mae_train).toLocaleString()}€</td>
                      <td className="px-6 py-4 text-sm text-gray-900">{Math.round(model.mae_test).toLocaleString()}€</td>
                      <td className="px-6 py-4 text-sm text-gray-900">{Math.round(model.mae_test - model.mae_train).toLocaleString()}€</td>
                      <td className="px-6 py-4 text-sm">
                        <span className={`px-2 py-1 rounded-full text-xs ${
                          model.mae_test < 30000 ? 'bg-green-100 text-green-800' :
                          model.mae_test < 45000 ? 'bg-yellow-100 text-yellow-800' :
                          'bg-red-100 text-red-800'
                        }`}>
                          {model.mae_test < 30000 ? 'Très bon' : model.mae_test < 45000 ? 'Correct' : 'Perfectible'}
                        </span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {activeTab === 'analysis' && (
            <div className="space-y-6">
              {/* Diagnostic principal */}
              <div 
                className="p-6 rounded-lg border-l-4 text-white"
                style={{ backgroundColor: model.color, borderLeftColor: model.color }}
              >
                <h4 className="text-lg font-semibold mb-2">{getStatusIcon(model.category)} {model.category}</h4>
                <p className="text-lg">{model.interpretation}</p>
              </div>

              {/* Analyse détaillée */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-blue-50 p-6 rounded-lg">
                  <h5 className="font-semibold text-blue-800 mb-3">🎯 Analyse de Performance</h5>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-center gap-2">
                      <span className={`w-3 h-3 rounded-full ${model.r2_test > 0.8 ? 'bg-green-500' : model.r2_test > 0.6 ? 'bg-yellow-500' : 'bg-red-500'}`}></span>
                      R² Test: {model.r2_test > 0.8 ? 'Excellente' : model.r2_test > 0.6 ? 'Bonne' : 'Insuffisante'} précision
                    </li>
                    <li className="flex items-center gap-2">
                      <span className={`w-3 h-3 rounded-full ${model.rmse_test < 50000 ? 'bg-green-500' : model.rmse_test < 70000 ? 'bg-yellow-500' : 'bg-red-500'}`}></span>
                      RMSE: Erreur {model.rmse_test < 50000 ? 'faible' : model.rmse_test < 70000 ? 'modérée' : 'élevée'}
                    </li>
                    <li className="flex items-center gap-2">
                      <span className={`w-3 h-3 rounded-full ${model.n_features < 100 ? 'bg-green-500' : model.n_features < 200 ? 'bg-yellow-500' : 'bg-red-500'}`}></span>
                      Complexité: {model.n_features < 100 ? 'Simple' : model.n_features < 200 ? 'Modérée' : 'Élevée'} ({model.n_features} features)
                    </li>
                  </ul>
                </div>

                <div className="bg-green-50 p-6 rounded-lg">
                  <h5 className="font-semibold text-green-800 mb-3">⚖️ Analyse de Généralisation</h5>
                  <ul className="space-y-2 text-sm">
                    <li className="flex items-center gap-2">
                      <span className={`w-3 h-3 rounded-full ${model.r2_gap < 0.05 ? 'bg-green-500' : model.r2_gap < 0.1 ? 'bg-yellow-500' : 'bg-red-500'}`}></span>
                      Gap R²: {model.r2_gap < 0.05 ? 'Excellent' : model.r2_gap < 0.1 ? 'Acceptable' : 'Problématique'} équilibre
                    </li>
                    <li className="flex items-center gap-2">
                      <span className={`w-3 h-3 rounded-full ${Math.abs(model.rmse_gap) < 5000 ? 'bg-green-500' : Math.abs(model.rmse_gap) < 10000 ? 'bg-yellow-500' : 'bg-red-500'}`}></span>
                      Stabilité RMSE: {Math.abs(model.rmse_gap) < 5000 ? 'Très stable' : Math.abs(model.rmse_gap) < 10000 ? 'Stable' : 'Instable'}
                    </li>
                    <li className="flex items-center gap-2">
                      <span className={`w-3 h-3 rounded-full ${['Good generalization', 'Light overfitting'].includes(model.category) ? 'bg-green-500' : 'bg-red-500'}`}></span>
                      Production: {['Good generalization', 'Light overfitting'].includes(model.category) ? 'Prêt' : 'Non recommandé'}
                    </li>
                  </ul>
                </div>
              </div>

              {/* Métriques comparatives */}
              <div className="bg-gray-50 p-6 rounded-lg">
                <h5 className="font-semibold text-gray-800 mb-4">📊 Benchmarks Secteur Immobilier</h5>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="text-center p-4 bg-white rounded">
                    <h6 className="text-sm font-medium text-gray-600">R² Attendu</h6>
                    <p className="text-lg font-bold">70-85%</p>
                    <p className={`text-sm ${model.r2_test >= 0.7 ? 'text-green-600' : 'text-red-600'}`}>
                      Votre modèle: {(model.r2_test * 100).toFixed(1)}%
                    </p>
                  </div>
                  <div className="text-center p-4 bg-white rounded">
                    <h6 className="text-sm font-medium text-gray-600">RMSE Acceptable</h6>
                    <p className="text-lg font-bold">40-60k€</p>
                    <p className={`text-sm ${model.rmse_test <= 60000 ? 'text-green-600' : 'text-red-600'}`}>
                      Votre modèle: {Math.round(model.rmse_test / 1000)}k€
                    </p>
                  </div>
                  <div className="text-center p-4 bg-white rounded">
                    <h6 className="text-sm font-medium text-gray-600">Gap R² Max</h6>
                    <p className="text-lg font-bold">&lt; 10%</p>
                    <p className={`text-sm ${model.r2_gap <= 0.1 ? 'text-green-600' : 'text-red-600'}`}>
                      Votre modèle: {(model.r2_gap * 100).toFixed(1)}%
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'recommendations' && (
            <div className="space-y-6">
              {/* Recommandation principale */}
              <div className="bg-blue-50 border border-blue-200 p-6 rounded-lg">
                <h4 className="text-lg font-semibold text-blue-800 mb-3">💡 Recommandation Principale</h4>
                <p className="text-blue-700 text-lg">{model.recommendation}</p>
              </div>

              {/* Actions spécifiques */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-green-50 p-6 rounded-lg">
                  <h5 className="font-semibold text-green-800 mb-3">✅ Actions Recommandées</h5>
                  <ul className="space-y-2 text-sm text-green-700">
                    {model.category === 'Good generalization' && (
                      <>
                        <li>• Déployer en production immédiatement</li>
                        <li>• Surveiller les performances en continu</li>
                        <li>• Documenter la configuration</li>
                        <li>• Préparer les tests A/B</li>
                      </>
                    )}
                    {model.category === 'Light overfitting' && (
                      <>
                        <li>• Acceptable pour production avec monitoring</li>
                        <li>• Considérer une légère régularisation</li>
                        <li>• Valider sur données récentes</li>
                        <li>• Surveiller la dérive du modèle</li>
                      </>
                    )}
                    {['Moderate overfitting', 'Strong overfitting'].includes(model.category) && (
                      <>
                        <li>• Augmenter la régularisation</li>
                        <li>• Réduire la complexité du modèle</li>
                        <li>• Obtenir plus de données d'entraînement</li>
                        <li>• Revoir la sélection de features</li>
                      </>
                    )}
                    {['Underfitting', 'Moderate underfitting'].includes(model.category) && (
                      <>
                        <li>• Augmenter la complexité du modèle</li>
                        <li>• Ajouter plus de features</li>
                        <li>• Optimiser les hyperparamètres</li>
                        <li>• Vérifier la qualité des données</li>
                      </>
                    )}
                  </ul>
                </div>

                <div className="bg-yellow-50 p-6 rounded-lg">
                  <h5 className="font-semibold text-yellow-800 mb-3">⚠️ Points d'Attention</h5>
                  <ul className="space-y-2 text-sm text-yellow-700">
                    {model.r2_gap > 0.1 && <li>• Gap R² élevé - risque d'overfitting</li>}
                    {model.rmse_test > 70000 && <li>• RMSE élevé - précision insuffisante</li>}
                    {model.n_features > 200 && <li>• Beaucoup de features - risque de complexité</li>}
                    {model.r2_test < 0.6 && <li>• R² faible - performances insuffisantes</li>}
                    <li>• Valider régulièrement sur nouvelles données</li>
                    <li>• Monitorer la dérive des performances</li>
                  </ul>
                </div>
              </div>

              {/* Prochaines étapes */}
              <div className="bg-purple-50 p-6 rounded-lg">
                <h5 className="font-semibold text-purple-800 mb-3">🚀 Prochaines Étapes</h5>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h6 className="font-medium text-purple-700 mb-2">Court terme (1-2 semaines)</h6>
                    <ul className="text-sm text-purple-600 space-y-1">
                      <li>• Valider sur données de validation</li>
                      <li>• Tests de performance en conditions réelles</li>
                      <li>• Documentation technique complète</li>
                    </ul>
                  </div>
                  <div>
                    <h6 className="font-medium text-purple-700 mb-2">Moyen terme (1-3 mois)</h6>
                    <ul className="text-sm text-purple-600 space-y-1">
                      <li>• Monitoring continu des performances</li>
                      <li>• Collecte de feedback utilisateurs</li>
                      <li>• Optimisation basée sur données réelles</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="border-t px-6 py-4 bg-gray-50 rounded-b-lg">
          <div className="flex justify-between items-center text-sm text-gray-600">
            <div>
              <span className="font-medium">Analysé le:</span> {new Date(model.timestamp).toLocaleString('fr-FR')}
            </div>
            <div className="flex gap-4">
              <span className="font-medium">Expérience:</span> {model.experiment_name || 'Standard'}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ModelDetailView;
