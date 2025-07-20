import React, { useState, useEffect, useRef, useCallback } from "react";
import "./AdminPanel.css";
import UserProfilesAdmin from './UserProfilesAdmin';
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
  Trash2,
  Users
} from 'lucide-react';

const AdminPanel = ({ isExpanded, onToggle, onClose }) => {
  console.log('AdminPanel rendered with props:', { isExpanded, onToggle, onClose });
  
  const [activeTab, setActiveTab] = useState('prompts');
  const [isDetached, setIsDetached] = useState(false);
  const [prompts, setPrompts] = useState([]);
  
  // États pour le redimensionnement et drag & drop
  const [panelWidth, setPanelWidth] = useState(400);
  const [panelHeight, setPanelHeight] = useState(600);
  const [isResizing, setIsResizing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
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
    { id: 'users', label: 'User Profiles', icon: Users },
    { id: 'monitoring', label: 'Monitor', icon: Settings }
  ];

  // Écouter les prompts LLM globalement
  useEffect(() => {
    console.log('🎯 AdminPanel: Setting up event listener for llmPromptSent');
    
    const handlePromptSent = (event) => {
      console.log('📨 AdminPanel: Event received!', event);
      console.log('📨 AdminPanel: Event detail:', event.detail);
      
      if (!event.detail) {
        console.error('❌ AdminPanel: Event detail is missing!');
        return;
      }
      
      const { prompt, type, timestamp } = event.detail;
      
      if (!prompt) {
        console.error('❌ AdminPanel: Prompt is missing from event detail!');
        return;
      }
      
      const newPrompt = {
        id: Date.now(),
        prompt,
        type: type || 'UNKNOWN',
        timestamp: timestamp || new Date().toISOString(),
        length: prompt.length
      };
      
      console.log('📝 AdminPanel: Adding new prompt:', newPrompt);
      setPrompts(prev => {
        const updated = [...prev, newPrompt];
        console.log('📊 AdminPanel: Total prompts:', updated.length);
        return updated;
      });
      
      // Auto-scroll vers le bas
      setTimeout(() => {
        if (promptsContainerRef.current) {
          promptsContainerRef.current.scrollTop = promptsContainerRef.current.scrollHeight;
        }
      }, 100);
    };

    // Ajouter l'événement IMMÉDIATEMENT
    window.addEventListener('llmPromptSent', handlePromptSent);
    console.log('✅ AdminPanel: Event listener added successfully');
    
    // Marquer l'AdminPanel comme prêt
    window.adminPanelReady = true;
    console.log('✅ AdminPanel: Marked as ready');
    
    return () => {
      console.log('🚮 AdminPanel: Removing event listener');
      window.removeEventListener('llmPromptSent', handlePromptSent);
      window.adminPanelReady = false;
    };
  }, []);

  // Fonction pour obtenir la classe CSS du curseur
  const getResizeCursorClass = (direction) => {
    switch (direction) {
      case 'left':
      case 'right':
        return 'resize-ew';
      case 'top':
      case 'bottom':
        return 'resize-ns';
      case 'top-left':
        return 'resize-nw';
      case 'top-right':
        return 'resize-ne';
      case 'bottom-left':
        return 'resize-sw';
      case 'bottom-right':
        return 'resize-se';
      default:
        return '';
    }
  };

  // Gestion du redimensionnement
  const handleResizeStart = useCallback((e, direction) => {
    e.preventDefault();
    e.stopPropagation();
    
    setIsResizing(true);
    
    // Ajouter la classe curseur au body
    const cursorClass = `resizing-${getResizeCursorClass(direction).replace('resize-', '')}`;
    document.body.classList.add(cursorClass);
    
    const startX = e.clientX;
    const startY = e.clientY;
    const startWidth = panelWidth;
    const startHeight = panelHeight;
    const startPosX = panelPosition.x;
    const startPosY = panelPosition.y;
    
    const handleResizeMove = (e) => {
      const deltaX = e.clientX - startX;
      const deltaY = e.clientY - startY;
      
      let newWidth = startWidth;
      let newHeight = startHeight;
      let newX = startPosX;
      let newY = startPosY;
      
      // Redimensionnement horizontal
      if (direction === 'right' || direction.includes('right')) {
        newWidth = Math.max(300, startWidth + deltaX);
      } else if (direction === 'left' || direction.includes('left')) {
        const potentialWidth = startWidth - deltaX;
        if (potentialWidth >= 300) {
          newWidth = potentialWidth;
          newX = startPosX + deltaX;
        } else {
          newWidth = 300;
          newX = startPosX + startWidth - 300;
        }
      }
      
      // Redimensionnement vertical
      if (direction === 'bottom' || direction.includes('bottom')) {
        newHeight = Math.max(400, startHeight + deltaY);
      } else if (direction === 'top' || direction.includes('top')) {
        const potentialHeight = startHeight - deltaY;
        if (potentialHeight >= 400) {
          newHeight = potentialHeight;
          newY = startPosY + deltaY;
        } else {
          newHeight = 400;
          newY = startPosY + startHeight - 400;
        }
      }
      
      // Contraintes de fenêtre
      newWidth = Math.min(newWidth, window.innerWidth - 50);
      newHeight = Math.min(newHeight, window.innerHeight - 100);
      
      // Contraintes de position
      newX = Math.max(0, Math.min(window.innerWidth - newWidth, newX));
      newY = Math.max(0, Math.min(window.innerHeight - newHeight, newY));
      
      // Appliquer les changements immédiatement
      setPanelWidth(newWidth);
      setPanelHeight(newHeight);
      setPanelPosition({ x: newX, y: newY });
    };
    
    const handleResizeStop = () => {
      setIsResizing(false);
      
      // Retirer la classe curseur du body
      document.body.classList.remove(cursorClass);
      
      document.removeEventListener('mousemove', handleResizeMove);
      document.removeEventListener('mouseup', handleResizeStop);
    };
    
    document.addEventListener('mousemove', handleResizeMove);
    document.addEventListener('mouseup', handleResizeStop);
  }, [panelWidth, panelHeight, panelPosition]);

  // Gestion du drag & drop
  const handleDragStart = useCallback((e) => {
    if (!isDetached) return;
    e.preventDefault();
    e.stopPropagation();
    
    setIsDragging(true);
    document.body.classList.add('dragging-panel');
    
    const rect = panelRef.current.getBoundingClientRect();
    const offsetX = e.clientX - rect.left;
    const offsetY = e.clientY - rect.top;
    
    const handleDragMove = (e) => {
      const newX = e.clientX - offsetX;
      const newY = e.clientY - offsetY;
      
      // Contraintes pour garder le panel dans la fenêtre
      const constrainedX = Math.max(0, Math.min(window.innerWidth - panelWidth, newX));
      const constrainedY = Math.max(0, Math.min(window.innerHeight - panelHeight, newY));
      
      setPanelPosition({ x: constrainedX, y: constrainedY });
    };
    
    const handleDragStop = () => {
      setIsDragging(false);
      document.body.classList.remove('dragging-panel');
      document.removeEventListener('mousemove', handleDragMove);
      document.removeEventListener('mouseup', handleDragStop);
    };
    
    document.addEventListener('mousemove', handleDragMove);
    document.addEventListener('mouseup', handleDragStop);
  }, [isDetached, panelWidth, panelHeight]);

  // Cleanup des event listeners quand le composant est démonté
  useEffect(() => {
    return () => {
      // Nettoyer les classes CSS du body
      document.body.classList.remove('dragging-panel', 'resizing-ew', 'resizing-ns', 'resizing-nw', 'resizing-ne', 'resizing-sw', 'resizing-se');
    };
  }, []);

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

  // Fonction de test pour envoyer un prompt
  const sendTestPrompt = () => {
    console.log('🧪 AdminPanel: Sending test prompt...');
    window.dispatchEvent(new CustomEvent('llmPromptSent', {
      detail: {
        type: 'TEST_PROMPT',
        prompt: 'Test prompt sent from AdminPanel - ' + new Date().toLocaleTimeString(),
        timestamp: new Date().toISOString(),
        metadata: {
          source: 'adminPanel',
          test: true
        }
      }
    }));
  };

  // Style dynamique basé sur l'état
  const panelStyle = {
    width: panelWidth,
    height: isDetached ? panelHeight : '100vh',
    position: 'fixed',
    top: isDetached ? panelPosition.y : '0',
    left: isDetached ? panelPosition.x : 'auto',
    right: isDetached ? 'auto' : '0',
    zIndex: isDetached ? 10000 : 1000,
    transform: isDetached ? 'none' : (isExpanded ? 'translateX(0)' : 'translateX(100%)'),
    boxShadow: isDetached ? '0 20px 60px rgba(0,0,0,0.3)' : 'none',
    borderRadius: isDetached ? '12px' : '0',
    overflow: 'hidden',
    userSelect: isDragging || isResizing ? 'none' : 'auto',
    pointerEvents: 'auto'
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
        <button onClick={sendTestPrompt} className="prompt-control-btn" style={{ backgroundColor: '#28a745' }}>
          <MessageSquare size={16} />
          Test Prompt
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
      case 'users':
        return <UserProfilesAdmin />;
      default:
        return <div className="tab-content">Select a tab</div>;
    }
  };

  return (
    <div 
      ref={panelRef}
      className={`admin-panel ${isExpanded ? 'expanded' : ''} ${isDetached ? 'detached' : ''} ${isDragging ? 'dragging' : ''} ${isResizing ? 'resizing' : ''}`}
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
