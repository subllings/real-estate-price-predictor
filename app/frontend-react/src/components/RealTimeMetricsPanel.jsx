/**
 * Real-time Metrics Panel - Live system monitoring for demo
 */

import React, { useState, useEffect } from 'react';
import { Activity, Users, Zap, CheckCircle, Clock } from 'lucide-react';

const RealTimeMetricsPanel = () => {
  const [metrics, setMetrics] = useState({});
  const [logs, setLogs] = useState([]);
  const [isConnected, setIsConnected] = useState(false);

  // Simulate real-time data for demo
  useEffect(() => {
    // Initial data
    setMetrics({
      predictions_per_hour: 145,
      avg_response_time: 245,
      success_rate: 99.2,
      active_users: 23,
      total_requests_today: 1247,
      api_health: 'healthy',
      db_connection: 'connected',
      azure_ml_status: 'running'
    });

    setLogs([
      { timestamp: new Date().toLocaleTimeString(), message: '✅ Prediction request completed (€420,000)', level: 'info' },
      { timestamp: new Date(Date.now() - 5000).toLocaleTimeString(), message: '🔄 Model v2.4.0 processing A/B test request', level: 'info' },
      { timestamp: new Date(Date.now() - 12000).toLocaleTimeString(), message: '📊 Azure ML training progress: 85% complete', level: 'success' },
      { timestamp: new Date(Date.now() - 18000).toLocaleTimeString(), message: '💬 LLM chat response generated successfully', level: 'info' },
      { timestamp: new Date(Date.now() - 25000).toLocaleTimeString(), message: '📄 Document uploaded: "Antwerp Market Report Q3"', level: 'info' }
    ]);

    setIsConnected(true);

    // Simulate live updates
    const interval = setInterval(() => {
      // Update metrics with small variations
      setMetrics(prev => ({
        ...prev,
        predictions_per_hour: prev.predictions_per_hour + Math.floor(Math.random() * 10 - 5),
        avg_response_time: prev.avg_response_time + Math.floor(Math.random() * 50 - 25),
        active_users: Math.max(1, prev.active_users + Math.floor(Math.random() * 6 - 3)),
        total_requests_today: prev.total_requests_today + Math.floor(Math.random() * 3)
      }));

      // Add new log entry occasionally
      if (Math.random() < 0.3) {
        const newLog = {
          timestamp: new Date().toLocaleTimeString(),
          message: getRandomLogMessage(),
          level: 'info'
        };
        
        setLogs(prev => [newLog, ...prev.slice(0, 9)]);
      }
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  const getRandomLogMessage = () => {
    const messages = [
      '✅ Property prediction completed successfully',
      '🔄 Model inference request processed',
      '💬 AI chat response generated',
      '📊 Performance metrics updated',
      '🔍 RAG search completed for market data',
      '⚡ Cache hit for property features',
      '📈 Model performance within targets',
      '🚀 Health check passed on all services'
    ];
    return messages[Math.floor(Math.random() * messages.length)];
  };

  const MetricCard = ({ title, value, icon: Icon, color, subtitle }) => (
    <div className={`bg-white rounded-lg border p-3 hover:shadow-md transition-shadow`}>
      <div className="flex items-center justify-between mb-2">
        <div className={`p-2 rounded-lg bg-${color}-100`}>
          <Icon className={`w-4 h-4 text-${color}-600`} />
        </div>
        <div className="text-right">
          <div className={`text-lg font-bold text-${color}-600`}>{value}</div>
          {subtitle && <div className="text-xs text-gray-500">{subtitle}</div>}
        </div>
      </div>
      <h4 className="text-sm font-medium text-gray-700">{title}</h4>
    </div>
  );

  const StatusIndicator = ({ status, label }) => {
    const colors = {
      healthy: 'green',
      connected: 'green', 
      running: 'blue',
      warning: 'yellow',
      error: 'red'
    };
    
    const color = colors[status] || 'gray';
    
    return (
      <div className="flex items-center justify-between py-2 px-3 bg-white rounded border">
        <span className="text-sm text-gray-700">{label}</span>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full bg-${color}-400 animate-pulse`}></div>
          <span className={`text-xs font-medium text-${color}-600 uppercase`}>
            {status}
          </span>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-4">
      {/* Connection Status */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold text-gray-800">📊 Live System Metrics</h3>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`}></div>
          <span className="text-xs text-gray-600">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
      </div>
      
      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 gap-3">
        <MetricCard 
          title="Predictions/Hour" 
          value={metrics.predictions_per_hour || 0}
          icon={Activity}
          color="blue"
        />
        <MetricCard 
          title="Avg Response" 
          value={`${metrics.avg_response_time || 0}ms`}
          icon={Zap}
          color="green"
        />
        <MetricCard 
          title="Success Rate" 
          value={`${metrics.success_rate || 0}%`}
          icon={CheckCircle}
          color="emerald"
        />
        <MetricCard 
          title="Active Users" 
          value={metrics.active_users || 0}
          icon={Users}
          color="purple"
        />
      </div>

      {/* System Status */}
      <div>
        <h4 className="font-medium text-gray-700 mb-2">🔧 System Health</h4>
        <div className="space-y-2">
          <StatusIndicator status={metrics.api_health} label="API Service" />
          <StatusIndicator status={metrics.db_connection} label="Database" />
          <StatusIndicator status={metrics.azure_ml_status} label="Azure ML" />
        </div>
      </div>

      {/* Live Activity Log */}
      <div>
        <h4 className="font-medium text-gray-700 mb-2 flex items-center space-x-2">
          <Clock size={16} />
          <span>🔄 Live Activity</span>
        </h4>
        <div className="bg-gray-900 text-green-400 p-3 rounded text-xs font-mono h-40 overflow-y-auto">
          {logs.map((log, index) => (
            <div key={index} className="mb-1 flex">
              <span className="text-gray-500 mr-2">[{log.timestamp}]</span>
              <span className={
                log.level === 'success' ? 'text-green-400' :
                log.level === 'warning' ? 'text-yellow-400' :
                log.level === 'error' ? 'text-red-400' :
                'text-green-400'
              }>
                {log.message}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Performance Summary */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-3 border">
        <h4 className="font-medium text-gray-800 mb-2">📈 Today's Summary</h4>
        <div className="grid grid-cols-2 gap-2 text-sm">
          <div>
            <span className="text-gray-600">Total Requests:</span>
            <span className="font-medium text-blue-600 ml-1">
              {metrics.total_requests_today || 0}
            </span>
          </div>
          <div>
            <span className="text-gray-600">Uptime:</span>
            <span className="font-medium text-green-600 ml-1">
              99.8%
            </span>
          </div>
          <div>
            <span className="text-gray-600">Peak Users:</span>
            <span className="font-medium text-purple-600 ml-1">
              47
            </span>
          </div>
          <div>
            <span className="text-gray-600">Avg Load:</span>
            <span className="font-medium text-orange-600 ml-1">
              12%
            </span>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="flex space-x-2">
        <button className="flex-1 bg-blue-600 text-white py-2 px-3 rounded text-sm hover:bg-blue-700 transition-colors">
          📊 Full Dashboard
        </button>
        <button className="flex-1 bg-green-600 text-white py-2 px-3 rounded text-sm hover:bg-green-700 transition-colors">
          📥 Export Logs
        </button>
      </div>
    </div>
  );
};

export default RealTimeMetricsPanel;
