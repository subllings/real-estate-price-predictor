import React from 'react';

const SimpleAdminPanelTest = ({ isExpanded, onClose }) => {
  console.log('SimpleAdminPanelTest rendered with isExpanded:', isExpanded);
  
  return (
    <div style={{
      position: 'fixed',
      top: 0,
      right: 0,
      width: '400px',
      height: '100vh',
      backgroundColor: 'rgba(102, 126, 234, 0.9)',
      color: 'white',
      padding: '20px',
      boxShadow: '0 0 20px rgba(0,0,0,0.5)',
      zIndex: 1000,
      transform: isExpanded ? 'translateX(0)' : 'translateX(100%)',
      transition: 'transform 0.3s ease'
    }}>
      <h2>Simple Admin Panel Test</h2>
      <p>isExpanded: {isExpanded ? 'true' : 'false'}</p>
      <button 
        onClick={onClose}
        style={{
          padding: '10px 20px',
          backgroundColor: '#ff4444',
          color: 'white',
          border: 'none',
          borderRadius: '5px',
          cursor: 'pointer',
          marginTop: '20px'
        }}
      >
        Close
      </button>
    </div>
  );
};

export default SimpleAdminPanelTest;
