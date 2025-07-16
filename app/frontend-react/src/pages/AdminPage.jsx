// AdminPage.jsx
import React, { useState } from 'react';
import ModelRegistry from '../components/ModelRegistry/ModelRegistry';

const AdminPage = () => {
  const [activeTab, setActiveTab] = useState('models');

  const tabs = [
    { id: 'models', label: '🤖 Model Registry', component: ModelRegistry },
    { id: 'documents', label: '📁 Documents', component: () => <div className="coming-soon">📄 Document Management - Coming Soon</div> },
    { id: 'monitoring', label: '📊 Monitoring', component: () => <div className="coming-soon">📈 Performance Monitoring - Coming Soon</div> },
    { id: 'settings', label: '⚙️ Settings', component: () => <div className="coming-soon">🔧 System Settings - Coming Soon</div> }
  ];

  const ActiveComponent = tabs.find(tab => tab.id === activeTab)?.component || ModelRegistry;

  return (
    <div style={{ minHeight: '100vh', backgroundColor: '#f8f9fa' }}>
      {/* Navigation Header */}
      <div style={{ 
        backgroundColor: '#fff',
        borderBottom: '1px solid #e9ecef',
        padding: '1rem 0',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 2rem' }}>
          <h1 style={{ margin: '0 0 1rem 0', color: '#2c3e50' }}>
            🔧 Real Estate AI - Admin Dashboard
          </h1>
          <nav style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap' }}>
            {tabs.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                style={{
                  padding: '0.5rem 1rem',
                  border: 'none',
                  borderRadius: '6px',
                  cursor: 'pointer',
                  backgroundColor: activeTab === tab.id ? '#007bff' : '#e9ecef',
                  color: activeTab === tab.id ? 'white' : '#495057',
                  transition: 'all 0.2s ease',
                  fontWeight: activeTab === tab.id ? 'bold' : 'normal'
                }}
                onMouseEnter={(e) => {
                  if (activeTab !== tab.id) {
                    e.target.style.backgroundColor = '#dee2e6';
                  }
                }}
                onMouseLeave={(e) => {
                  if (activeTab !== tab.id) {
                    e.target.style.backgroundColor = '#e9ecef';
                  }
                }}
              >
                {tab.label}
              </button>
            ))}
            <button
              onClick={() => window.location.href = '/'}
              style={{
                padding: '0.5rem 1rem',
                border: '1px solid #28a745',
                borderRadius: '6px',
                cursor: 'pointer',
                backgroundColor: 'transparent',
                color: '#28a745',
                transition: 'all 0.2s ease',
                marginLeft: 'auto'
              }}
              onMouseEnter={(e) => {
                e.target.style.backgroundColor = '#28a745';
                e.target.style.color = 'white';
              }}
              onMouseLeave={(e) => {
                e.target.style.backgroundColor = 'transparent';
                e.target.style.color = '#28a745';
              }}
            >
              🏠 Back to App
            </button>
          </nav>
        </div>
      </div>

      {/* Main Content */}
      <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
        <ActiveComponent />
      </div>

      {/* Coming Soon Styling */}
      <style jsx>{`
        .coming-soon {
          display: flex;
          align-items: center;
          justify-content: center;
          height: 300px;
          font-size: 1.5rem;
          color: #6c757d;
          background-color: #f8f9fa;
          border-radius: 8px;
          margin: 2rem;
          border: 2px dashed #dee2e6;
        }
      `}</style>
    </div>
  );
};

export default AdminPage;
