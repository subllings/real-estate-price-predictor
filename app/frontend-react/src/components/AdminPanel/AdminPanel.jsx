import React, { useState, useEffect, useRef } from "react";
import "./AdminPanel.css";
import { 
  X, 
  Settings, 
  BarChart, 
  FileText, 
  Activity, 
  MessageSquare, 
  Maximize2,
  Minimize2,
  RotateCcw,
  Copy,
  Download,
  Trash2
} from 'lucide-react';

const AdminPanel = ({ isExpanded, onToggle, onClose }) => {
  console.log('AdminPanel rendered with props:', { isExpanded, onToggle, onClose });
  
  const [activeTab, setActiveTab] = useState('prompts');
  const [isDetached, setIsDetached] = useState(false);
  const [prompts, setPrompts] = useState([]);
  
  // États pour le redimensionnement
  const [panelWidth, setPanelWidth] = useState(400);
  const [panelHeight, setPanelHeight] = useState(600);
  const [isResizing, setIsResizing] = useState(false);
  const [startX, setStartX] = useState(0);
  const [startY, setStartY] = useState(0);
  const [startWidth, setStartWidth] = useState(400);
  const [startHeight, setStartHeight] = useState(600);
  const [resizeDirection, setResizeDirection] = useState('');
  
  // États pour le drag & drop
  const [isDragging, setIsDragging] = useState(false);
  const [dragOffset, setDragOffset] = useState({ x: 0, y: 0 });
  const [panelPosition, setPanelPosition] = useState({ x: 0, y: 0 });
  
  // Références
  const panelRef = useRef(null);
  const headerRef = useRef(null);
  const promptsContainerRef = useRef(null);

  // Onglets disponibles
  const tabs = [
    { id: 'prompts', label: 'Prompt Visualization', icon: MessageSquare },
    { id: 'models', label: 'Models', icon: BarChart },
    { id: 'training', label: 'Training', icon: Activity },
    { id: 'documents', label: 'Documents', icon: FileText },
    { id: 'monitoring', label: 'Monitor', icon: Settings }
  ];

  // Écouter les prompts LLM globalement
  useEffect(() => {
    const handlePromptSent = (event) => {
      const { prompt, type, timestamp } = event.detail;
      const newPrompt = {
        id: Date.now(),
        prompt,
        type,
        timestamp: timestamp || new Date().toISOString(),
        length: prompt.length
      };
      setPrompts(prev => [...prev, newPrompt]);
      
      // Auto-scroll vers le bas
      setTimeout(() => {
        if (promptsContainerRef.current) {
          promptsContainerRef.current.scrollTop = promptsContainerRef.current.scrollHeight;
        }
      }, 100);
    };

    window.addEventListener('llm-prompt-sent', handlePromptSent);
    return () => window.removeEventListener('llm-prompt-sent', handlePromptSent);
  }, []);

  // Gestion du redimensionnement
  const handleResizeStart = (e, direction) => {
    e.preventDefault();
    setIsResizing(true);
    setResizeDirection(direction);
    setStartX(e.clientX);
    setStartY(e.clientY);
    setStartWidth(panelWidth);
    setStartHeight(panelHeight);
    
    document.addEventListener('mousemove', handleResize);
    document.addEventListener('mouseup', handleResizeEnd);
  };

  const handleResize = (e) => {
    if (!isResizing) return;
    
    if (resizeDirection.includes('right')) {
      const newWidth = startWidth + (e.clientX - startX);
      setPanelWidth(Math.max(300, Math.min(window.innerWidth - 50, newWidth)));
    }
    if (resizeDirection.includes('left')) {
      const newWidth = startWidth - (e.clientX - startX);
      setPanelWidth(Math.max(300, Math.min(window.innerWidth - 50, newWidth)));
    }
    if (resizeDirection.includes('bottom')) {
      const newHeight = startHeight + (e.clientY - startY);
      setPanelHeight(Math.max(400, Math.min(window.innerHeight - 100, newHeight)));
    }
    if (resizeDirection.includes('top')) {
      const newHeight = startHeight - (e.clientY - startY);
      setPanelHeight(Math.max(400, Math.min(window.innerHeight - 100, newHeight)));
    }
  };

  const handleResizeEnd = () => {
    setIsResizing(false);
    setResizeDirection('');
    document.removeEventListener('mousemove', handleResize);
    document.removeEventListener('mouseup', handleResizeEnd);
  };

  // Gestion du drag & drop
  const handleDragStart = (e) => {
    if (!isDetached) return;
    
    setIsDragging(true);
    const rect = panelRef.current.getBoundingClientRect();
    setDragOffset({
      x: e.clientX - rect.left,
      y: e.clientY - rect.top
    });
    
    document.addEventListener('mousemove', handleDrag);
    document.addEventListener('mouseup', handleDragEnd);
  };

  const handleDrag = (e) => {
    if (!isDragging || !isDetached) return;
    
    const newX = e.clientX - dragOffset.x;
    const newY = e.clientY - dragOffset.y;
    
    setPanelPosition({
      x: Math.max(0, Math.min(window.innerWidth - panelWidth, newX)),
      y: Math.max(0, Math.min(window.innerHeight - panelHeight, newY))
    });
  };

  const handleDragEnd = () => {
    setIsDragging(false);
    document.removeEventListener('mousemove', handleDrag);
    document.removeEventListener('mouseup', handleDragEnd);
  };

  // Fonctions utilitaires pour les prompts
  const clearPrompts = () => {
    setPrompts([]);
  };

  const copyPrompt = (prompt) => {
    navigator.clipboard.writeText(prompt);
    // Notification simple
    const notification = document.createElement('div');
    notification.textContent = 'Prompt copied to clipboard!';
    notification.className = 'prompt-copy-notification';
    document.body.appendChild(notification);
    setTimeout(() => notification.remove(), 2000);
  };

  const exportPrompts = () => {
    const data = JSON.stringify(prompts, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `llm-prompts-${new Date().toISOString().split('T')[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Style dynamique basé sur l'état
  const panelStyle = {
    width: panelWidth,
    height: isDetached ? panelHeight : '100vh',
    position: isDetached ? 'fixed' : 'fixed',
    top: isDetached ? panelPosition.y : '0',
    left: isDetached ? panelPosition.x : 'auto',
    right: isDetached ? 'auto' : '0',
    zIndex: isDetached ? 10000 : 1000,
    transform: isDetached ? 'none' : (isExpanded ? 'translateX(0)' : 'translateX(100%)'),
    boxShadow: isDetached ? '0 20px 60px rgba(0,0,0,0.3)' : 'none',
    borderRadius: isDetached ? '12px' : '0',
    overflow: 'hidden'
  };

  const renderPromptVisualization = () => (
    <div className="prompt-visualization-container">
      <div className="prompt-controls">
        <button onClick={clearPrompts} className="prompt-control-btn">
          <Trash2 size={16} />
          Clear All
        </button>
        <button onClick={exportPrompts} className="prompt-control-btn">
          <Download size={16} />
          Export
        </button>
        <span className="prompt-count">{prompts.length} prompts</span>
      </div>
      
      <div 
        ref={promptsContainerRef}
        className="prompts-container"
        style={{ height: isDetached ? panelHeight - 120 : 'calc(100vh - 120px)' }}
      >
        {prompts.length === 0 ? (
          <div className="no-prompts">
            <MessageSquare size={48} />
            <p>No prompts sent yet</p>
            <p className="text-sm text-gray-500">LLM prompts will appear here when sent</p>
          </div>
        ) : (
          prompts.map((promptData) => (
            <div key={promptData.id} className="prompt-item">
              <div className="prompt-header">
                <div className="prompt-info">
                  <span className="prompt-type">{promptData.type}</span>
                  <span className="prompt-timestamp">
                    {new Date(promptData.timestamp).toLocaleTimeString()}
                  </span>
                  <span className="prompt-length">{promptData.length} chars</span>
                </div>
                <button 
                  onClick={() => copyPrompt(promptData.prompt)}
                  className="copy-btn"
                  title="Copy prompt"
                >
                  <Copy size={14} />
                </button>
              </div>
              <div className="prompt-content">
                <pre>{promptData.prompt}</pre>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );

  const renderTabContent = () => {
    switch (activeTab) {
      case 'prompts':
        return renderPromptVisualization();
      case 'models':
        return <div className="tab-content">Models Management Panel</div>;
      case 'training':
        return <div className="tab-content">Training Status Panel</div>;
      case 'documents':
        return <div className="tab-content">Documents Panel</div>;
      case 'monitoring':
        return <div className="tab-content">Real-time Monitoring Panel</div>;
      default:
        return <div className="tab-content">Select a tab</div>;
    }
  };

  return (
    <div 
      ref={panelRef}
      className={`admin-panel ${isExpanded ? 'expanded' : ''} ${isDetached ? 'detached' : ''}`}
      style={panelStyle}
    >
      {/* Resize handles */}
      {isExpanded && (
        <>
          <div 
            className="resize-handle resize-handle-left"
            onMouseDown={(e) => handleResizeStart(e, 'left')}
          />
          <div 
            className="resize-handle resize-handle-right"
            onMouseDown={(e) => handleResizeStart(e, 'right')}
          />
          {isDetached && (
            <>
              <div 
                className="resize-handle resize-handle-top"
                onMouseDown={(e) => handleResizeStart(e, 'top')}
              />
              <div 
                className="resize-handle resize-handle-bottom"
                onMouseDown={(e) => handleResizeStart(e, 'bottom')}
              />
              <div 
                className="resize-handle resize-handle-top-left"
                onMouseDown={(e) => handleResizeStart(e, 'top-left')}
              />
              <div 
                className="resize-handle resize-handle-top-right"
                onMouseDown={(e) => handleResizeStart(e, 'top-right')}
              />
              <div 
                className="resize-handle resize-handle-bottom-left"
                onMouseDown={(e) => handleResizeStart(e, 'bottom-left')}
              />
              <div 
                className="resize-handle resize-handle-bottom-right"
                onMouseDown={(e) => handleResizeStart(e, 'bottom-right')}
              />
            </>
          )}
        </>
      )}

      {/* Header */}
      <div 
        ref={headerRef}
        className="admin-panel-header"
        onMouseDown={handleDragStart}
        style={{ cursor: isDetached ? 'move' : 'default' }}
      >
        <div className="admin-panel-title">
          <Settings size={20} />
          <span>Admin Panel</span>
        </div>
        
        <div className="admin-panel-controls">
          <button 
            onClick={() => setIsDetached(!isDetached)}
            className="control-btn"
            title={isDetached ? 'Attach to side' : 'Detach from side'}
          >
            {isDetached ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
          <button 
            onClick={onToggle}
            className="control-btn"
            title="Toggle panel"
          >
            <RotateCcw size={16} />
          </button>
          <button 
            onClick={onClose}
            className="control-btn close-btn"
            title="Close panel"
          >
            <X size={16} />
          </button>
        </div>
      </div>

      {/* Tabs */}
      <div className="admin-panel-tabs">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          return (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`}
            >
              <Icon size={16} />
              <span>{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Content */}
      <div className="admin-panel-content">
        {renderTabContent()}
      </div>
    </div>
  );
};

export default AdminPanel;
