import React, { useState, useEffect, useCallback } from 'react';
import './DocumentUploadPage.css';

const DocumentUploadPage = () => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploadProgress, setUploadProgress] = useState({});
  const [uploadStatus, setUploadStatus] = useState('');
  const [isUploading, setIsUploading] = useState(false);
  const [documents, setDocuments] = useState([]);
  const [indexStats, setIndexStats] = useState({
    total_documents: 0,
    total_chunks: 0,
    total_size_bytes: 0,
    index_size_mb: 0,
    last_updated: 'Never'
  });
  const [dragOver, setDragOver] = useState(false);

  // API Base URL
  const API_BASE_URL = process.env.NODE_ENV === 'production' 
    ? 'https://realestate-llm-api-agent.azurewebsites.net'
    : 'http://localhost:8001';

  // Load documents and stats on component mount
  useEffect(() => {
    loadDocuments();
    loadIndexStats();
  }, []);

  const loadDocuments = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/documents`);
      if (response.ok) {
        const data = await response.json();
        setDocuments(data.documents || []);
      } else {
        console.error('Failed to load documents');
      }
    } catch (error) {
      console.error('Error loading documents:', error);
    }
  };

  const loadIndexStats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/index_stats`);
      if (response.ok) {
        const data = await response.json();
        setIndexStats(data);
      } else {
        console.error('Failed to load index stats');
      }
    } catch (error) {
      console.error('Error loading index stats:', error);
    }
  };

  const handleFileSelection = (files) => {
    const fileArray = Array.from(files);
    const validFiles = fileArray.filter(file => {
      const validTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
      return validTypes.includes(file.type) || file.name.toLowerCase().endsWith('.pdf') || file.name.toLowerCase().endsWith('.docx') || file.name.toLowerCase().endsWith('.txt');
    });

    if (validFiles.length !== fileArray.length) {
      setUploadStatus('Certains fichiers ont été ignorés. Seuls les fichiers PDF, DOCX et TXT sont supportés.');
    }

    setSelectedFiles(prev => [...prev, ...validFiles.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      progress: 0
    }))]);
  };

  const handleFileInputChange = (event) => {
    handleFileSelection(event.target.files);
  };

  const handleDragOver = useCallback((event) => {
    event.preventDefault();
    setDragOver(true);
  }, []);

  const handleDragLeave = useCallback((event) => {
    event.preventDefault();
    setDragOver(false);
  }, []);

  const handleDrop = useCallback((event) => {
    event.preventDefault();
    setDragOver(false);
    handleFileSelection(event.dataTransfer.files);
  }, []);

  const removeFile = (fileId) => {
    setSelectedFiles(prev => prev.filter(f => f.id !== fileId));
    setUploadProgress(prev => {
      const newProgress = { ...prev };
      delete newProgress[fileId];
      return newProgress;
    });
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileIcon = (fileName) => {
    const extension = fileName.toLowerCase().split('.').pop();
    switch (extension) {
      case 'pdf':
        return '📄';
      case 'docx':
        return '📝';
      case 'txt':
        return '📃';
      default:
        return '📄';
    }
  };

  const uploadFiles = async () => {
    if (selectedFiles.length === 0) return;

    setIsUploading(true);
    setUploadStatus('Traitement des documents...');

    try {
      for (const fileItem of selectedFiles) {
        const formData = new FormData();
        formData.append('file', fileItem.file);
        formData.append('document_type', 'esg_document');

        setUploadProgress(prev => ({ ...prev, [fileItem.id]: 0 }));

        const response = await fetch(`${API_BASE_URL}/upload_document`, {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          setUploadProgress(prev => ({ ...prev, [fileItem.id]: 100 }));
          const result = await response.json();
          console.log('Upload successful:', result);
        } else {
          const error = await response.json();
          throw new Error(error.detail || 'Upload failed');
        }
      }

      setUploadStatus(`✅ ${selectedFiles.length} document(s) traité(s) avec succès!`);
      setSelectedFiles([]);
      setUploadProgress({});
      
      // Reload documents and stats
      await loadDocuments();
      await loadIndexStats();

    } catch (error) {
      console.error('Upload error:', error);
      setUploadStatus(`❌ Erreur lors du traitement: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const deleteDocument = async (documentId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/documents/${documentId}`, {
        method: 'DELETE'
      });

      if (response.ok) {
        await loadDocuments();
        await loadIndexStats();
        setUploadStatus('✅ Document supprimé avec succès');
      } else {
        throw new Error('Failed to delete document');
      }
    } catch (error) {
      console.error('Delete error:', error);
      setUploadStatus('❌ Erreur lors de la suppression');
    }
  };

  return (
    <div className="document-upload-page">
      {/* Page Header */}
      <div className="page-header">
        <h1>📚 Système RAG de Traitement des Documents</h1>
        <p className="page-subtitle">
          Téléchargez vos documents ESG, légaux et énergétiques pour une analyse intelligente
        </p>
      </div>

      {/* RAG Explanation */}
      <div className="rag-explanation">
        <h3>🔍 Comment fonctionne notre système RAG (Retrieval-Augmented Generation)</h3>
        <div className="rag-steps">
          <div className="rag-step">
            <div className="step-number">1</div>
            <div className="step-content">
              <strong>📄 Extraction du Contenu</strong>
              <p>Le système extrait automatiquement le texte de vos documents PDF, DOCX et TXT avec une précision maximale.</p>
            </div>
          </div>
          <div className="rag-step">
            <div className="step-number">2</div>
            <div className="step-content">
              <strong>🧩 Segmentation Intelligente</strong>
              <p>Le contenu est découpé en segments logiques optimisés pour l'analyse et la recherche sémantique.</p>
            </div>
          </div>
          <div className="rag-step">
            <div className="step-number">3</div>
            <div className="step-content">
              <strong>🧠 Indexation FAISS</strong>
              <p>Chaque segment est transformé en embeddings vectoriels et indexé dans une base de données FAISS haute performance.</p>
            </div>
          </div>
          <div className="rag-step">
            <div className="step-number">4</div>
            <div className="step-content">
              <strong>💡 Analyse Azure OpenAI</strong>
              <p>Les requêtes utilisent GPT-4 et text-embedding-ada-002 pour des réponses contextualisées et précises.</p>
            </div>
          </div>
        </div>
      </div>

      {/* Upload Section */}
      <div className="upload-section">
        <div 
          className={`upload-area ${dragOver ? 'drag-over' : ''}`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <input
            type="file"
            multiple
            accept=".pdf,.docx,.txt"
            onChange={handleFileInputChange}
            className="file-input-hidden"
            id="file-input"
          />
          <label htmlFor="file-input" className="upload-button">
            📎 Choisir des Fichiers
          </label>
          <p>ou glissez-déposez vos documents ici</p>
          <div className="supported-formats">
            <strong>Formats supportés:</strong> PDF, DOCX, TXT
          </div>
        </div>

        {/* Selected Files */}
        {selectedFiles.length > 0 && (
          <div className="selected-files">
            <h4>📋 Fichiers Sélectionnés ({selectedFiles.length})</h4>
            {selectedFiles.map((fileItem) => (
              <div key={fileItem.id} className="file-item">
                <span className="file-icon">{getFileIcon(fileItem.file.name)}</span>
                <div className="file-details">
                  <div className="file-name">{fileItem.file.name}</div>
                  <div className="file-size">{formatFileSize(fileItem.file.size)}</div>
                </div>
                <div className="progress-container">
                  <div className="progress-bar">
                    <div 
                      className="progress-fill" 
                      style={{ width: `${uploadProgress[fileItem.id] || 0}%` }}
                    ></div>
                  </div>
                  <span className="progress-text">{uploadProgress[fileItem.id] || 0}%</span>
                </div>
                <button 
                  className="remove-file-btn"
                  onClick={() => removeFile(fileItem.id)}
                  disabled={isUploading}
                >
                  ✕
                </button>
              </div>
            ))}
            
            <button 
              className="upload-submit-btn"
              onClick={uploadFiles}
              disabled={isUploading || selectedFiles.length === 0}
            >
              {isUploading ? '⏳ Traitement en cours...' : `🚀 Traiter ${selectedFiles.length} document(s)`}
            </button>
          </div>
        )}

        {/* Status Message */}
        {uploadStatus && (
          <div className={`status-message ${uploadStatus.includes('✅') ? 'success' : 'error'}`}>
            {uploadStatus}
          </div>
        )}
      </div>

      {/* Index Statistics */}
      <div className="index-stats">
        <h3>📊 Statistiques de l'Index FAISS</h3>
        <div className="stats-grid">
          <div className="stat-item">
            <span className="stat-value">{indexStats.total_documents}</span>
            <span className="stat-label">Documents</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{indexStats.total_chunks}</span>
            <span className="stat-label">Segments</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{formatFileSize(indexStats.total_size_bytes)}</span>
            <span className="stat-label">Taille Totale</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{indexStats.index_size_mb} MB</span>
            <span className="stat-label">Index FAISS</span>
          </div>
        </div>
      </div>

      {/* Documents List */}
      <div className="documents-list">
        <h3>📁 Documents Indexés ({documents.length})</h3>
        {documents.length === 0 ? (
          <div className="empty-state">
            <p>🕳️ Aucun document indexé. Commencez par télécharger vos premiers documents!</p>
          </div>
        ) : (
          <div className="documents-grid">
            {documents.map((doc) => (
              <div key={doc.id} className="document-card">
                <div className="document-header">
                  <span className="doc-icon">{getFileIcon(doc.filename)}</span>
                  <div className="doc-info">
                    <h4 className="doc-name">{doc.filename}</h4>
                    <p className="doc-meta">
                      {formatFileSize(doc.size_bytes)} • {doc.chunks_count} segments • {new Date(doc.upload_time).toLocaleDateString('fr-FR')}
                    </p>
                  </div>
                  <button 
                    className="delete-doc-btn"
                    onClick={() => deleteDocument(doc.id)}
                    title="Supprimer le document"
                  >
                    🗑️
                  </button>
                </div>
                
                <div className="document-preview">
                  <p>{doc.content_preview}</p>
                </div>
                
                <div className="document-tags">
                  {doc.tags.map((tag, index) => (
                    <span key={index} className="tag">{tag}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Technical Stack */}
      <div className="tech-stack">
        <h3>⚙️ Architecture Technique</h3>
        <div className="stack-items">
          <div className="stack-item">
            <strong>🔍 FAISS Vector Database:</strong> Indexation vectorielle haute performance pour la recherche sémantique
          </div>
          <div className="stack-item">
            <strong>🦜 LangChain Framework:</strong> Orchestration des workflows RAG et gestion des embeddings
          </div>
          <div className="stack-item">
            <strong>🧠 Azure OpenAI:</strong> GPT-4 pour la génération et text-embedding-ada-002 pour la vectorisation
          </div>
          <div className="stack-item">
            <strong>📄 PyMuPDF + python-docx:</strong> Extraction de texte multi-format avec préservation de la structure
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocumentUploadPage;
