import React, { useState } from 'react';
import AdminPanel from './components/AdminPanel/AdminPanel';

const TestAdminPanel = () => {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <div style={{ padding: '20px', background: '#f0f0f0', minHeight: '100vh' }}>
      <h1>Test AdminPanel</h1>
      <button 
        onClick={() => setIsVisible(!isVisible)}
        style={{
          padding: '10px 20px',
          background: '#007bff',
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: 'pointer',
          margin: '10px 0'
        }}
      >
        {isVisible ? 'Hide' : 'Show'} Admin Panel
      </button>
      
      {isVisible && (
        <AdminPanel 
          isExpanded={isVisible}
          onToggle={() => setIsVisible(!isVisible)}
          onClose={() => setIsVisible(false)}
        />
      )}
    </div>
  );
};

export default TestAdminPanel;
