import React, { useState, useEffect } from 'react';
import './DocumentUploadPage.css';

const DocumentUploadPage = () => {
  const [activeTab, setActiveTab] = useState('upload');
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [vectorStore, setVectorStore] = useState('faiss');
  const [ragEnabled, setRagEnabled] = useState(true);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const [documents, setDocuments] = useState([]);
  const [stats, setStats] = useState({
    fileCount: 0,
    totalChunks: 0,
    embeddingSize: 0,
    vectorIndexSize: 0
  });

  // API Base URL for LLM backend - try multiple endpoints
  const API_ENDPOINTS = [
    'http://127.0.0.1:8010',
    'http://localhost:8010'
  ];
  
  const [currentApiUrl, setCurrentApiUrl] = useState(API_ENDPOINTS[0]);

  const [apiStatus, setApiStatus] = useState('checking'); // 'checking', 'online', 'offline'

  // Load documents and stats on component mount
  useEffect(() => {
    const initializeApi = async () => {
      setApiStatus('checking');
      const workingEndpoint = await testApiConnectivity();
      if (workingEndpoint) {
        setApiStatus('online');
        loadDocuments();
        loadStats();
      } else {
        setApiStatus('offline');
      }
    };
    
    initializeApi();
  }, []);

  // Test API connectivity and find working endpoint
  const testApiConnectivity = async () => {
    for (const endpoint of API_ENDPOINTS) {
      try {
        const response = await fetch(`${endpoint}/health`, { 
          method: 'GET',
          timeout: 3000 
        });
        if (response.ok) {
          setCurrentApiUrl(endpoint);
          return endpoint;
        }
      } catch (error) {
        continue;
      }
    }
    return null;
  };

  // Load documents from backend
  const loadDocuments = async () => {
    try {
      const workingEndpoint = await testApiConnectivity();
      if (!workingEndpoint) {
        setUploadStatus('Backend API is not available. Please start the LLM backend service.');
        return;
      }
      
      const response = await fetch(`${workingEndpoint}/documents`);
      if (response.ok) {
        const data = await response.json();
        setDocuments(data.documents || []);
      }
    } catch (error) {
      console.error('Error loading documents:', error);
      setUploadStatus('Error connecting to backend service.');
    }
  };

  // Load statistics from backend
  const loadStats = async () => {
    try {
      if (!currentApiUrl) return;
      
      const response = await fetch(`${currentApiUrl}/stats`);
      if (response.ok) {
        const data = await response.json();
        setStats(data);
      }
    } catch (error) {
      console.error('Error loading stats:', error);
    }
  };

  // Handle drag and drop
  const [isDragging, setIsDragging] = useState(false);

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      handleFileSelection(files);
    }
  };

  // Handle file selection
  const handleFileSelection = (files) => {
    const fileArray = Array.from(files);
    const validFiles = fileArray.filter(file => {
      const validTypes = ['application/pdf', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain'];
      return validTypes.includes(file.type) || 
             file.name.toLowerCase().endsWith('.pdf') || 
             file.name.toLowerCase().endsWith('.docx') || 
             file.name.toLowerCase().endsWith('.txt');
    });

    if (validFiles.length !== fileArray.length) {
      setUploadStatus('Some files were ignored. Only PDF, DOCX and TXT files are supported.');
    }

    setSelectedFiles(prev => [...prev, ...validFiles.map(file => ({
      id: Math.random().toString(36).substr(2, 9),
      file,
      status: 'pending'
    }))]);
  };

  // Handle file upload
  const handleUpload = async () => {
    if (selectedFiles.length === 0) return;

    setIsUploading(true);
    setUploadStatus('Testing API connectivity...');

    // Test API connectivity first
    const workingEndpoint = await testApiConnectivity();
    if (!workingEndpoint) {
      setUploadStatus('Error: Backend API is not available. Please start the LLM backend service on port 8010 or 8000.');
      setIsUploading(false);
      return;
    }

    setUploadStatus('Processing documents...');

    console.log('Upload Configuration:', {
      vector_store: vectorStore,
      with_rag: ragEnabled,
      files: selectedFiles.length,
      endpoint: workingEndpoint
    });

    try {
      for (const fileItem of selectedFiles) {
        const formData = new FormData();
        formData.append('file', fileItem.file);
        formData.append('vector_store', vectorStore);
        formData.append('with_rag', ragEnabled.toString());

        const response = await fetch(`${workingEndpoint}/upload_document`, {
          method: 'POST',
          body: formData,
        });

        if (response.ok) {
          const result = await response.json();
          console.log('Upload successful:', result);
          setSelectedFiles(prev => 
            prev.map(f => f.id === fileItem.id ? {...f, status: 'completed'} : f)
          );
        } else {
          const errorText = await response.text();
          let errorMessage = 'Upload failed';
          
          try {
            const errorJson = JSON.parse(errorText);
            errorMessage = errorJson.detail || errorJson.message || errorMessage;
          } catch {
            errorMessage = errorText || errorMessage;
          }
          
          setSelectedFiles(prev => 
            prev.map(f => f.id === fileItem.id ? {...f, status: 'error'} : f)
          );
          throw new Error(`${response.status}: ${errorMessage}`);
        }
      }

      setUploadStatus(`Successfully processed ${selectedFiles.length} document(s)!`);
      
      // Reload documents and stats
      await loadDocuments();
      await loadStats();
      
      // Clear selection after successful upload
      setTimeout(() => {
        setSelectedFiles([]);
        setUploadStatus('');
      }, 3000);

    } catch (error) {
      console.error('Upload error:', error);
      setUploadStatus(`Error during processing: ${error.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  // Remove file from selection
  const removeFile = (fileId) => {
    setSelectedFiles(prev => prev.filter(f => f.id !== fileId));
  };

  // Format file size
  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="document-upload-page">
      {/* Page Header */}
      <div className="page-header">
        <h1>RAG Document Processing System</h1>
        <p className="page-subtitle">
          Upload your ESG, legal and energy documents for intelligent analysis
        </p>
        
        {/* API Status Indicator */}
        <div className={`api-status ${apiStatus}`}>
          <span className="status-dot"></span>
          {apiStatus === 'checking' && 'Checking API connection...'}
          {apiStatus === 'online' && `Backend API connected (${currentApiUrl})`}
          {apiStatus === 'offline' && 'Backend API unavailable - Please start the LLM service'}
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="tab-container">
        <div className="tab-header">
          <button 
            className={`tab-button ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            Upload & Vectorization
          </button>
          <button 
            className={`tab-button ${activeTab === 'library' ? 'active' : ''}`}
            onClick={() => setActiveTab('library')}
          >
            Document Library
          </button>
        </div>

        {/* Tab Content */}
        <div className="tab-content">
          {activeTab === 'upload' && (
            <div className="upload-tab">
              {/* Configuration Section */}
              <div className="config-section">
                <h3>Configuration</h3>
                
                {/* Vector Store Selection */}
                <div className="config-group">
                  <label className="config-label">Vector Store</label>
                  <select 
                    value={vectorStore} 
                    onChange={(e) => setVectorStore(e.target.value)}
                    className="config-select"
                  >
                    <option value="faiss">FAISS (Local)</option>
                    <option value="azure">Azure Cognitive Search (Cloud)</option>
                  </select>
                  <p className="config-description">
                    {vectorStore === 'faiss' 
                      ? 'High-performance local vector store (ideal for dev/testing). Ultra-fast similarity search. No cloud dependency. Local storage only. Should run inside a container in production.'
                      : 'Fully managed search index on Azure. Built-in scaling and security. Ideal for production use. Works with Azure OpenAI embeddings.'
                    }
                  </p>
                </div>

                {/* RAG Toggle */}
                <div className="config-group">
                  <label className="config-label">RAG Mode</label>
                  <div className="rag-checkbox-container">
                    <label className="checkbox-label">
                      <input
                        type="checkbox"
                        checked={ragEnabled}
                        onChange={(e) => setRagEnabled(e.target.checked)}
                        className="rag-checkbox"
                      />
                      <span className="checkmark"></span>
                      Enable RAG (Retrieval-Augmented Generation)
                    </label>
                  </div>
                  <p className="config-description">
                    {ragEnabled 
                      ? 'Documents will be embedded using Azure OpenAI and indexed into the selected vector store.'
                      : 'Files are stored only, no vectorization or indexing performed.'
                    }
                  </p>
                </div>
              </div>

              {/* File Upload Section */}
              <div className="upload-section">
                <h3>Upload Documents</h3>
                
                <div 
                  className={`file-input-area ${isDragging ? 'dragging' : ''}`}
                  onDragEnter={handleDragEnter}
                  onDragLeave={handleDragLeave}
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                >
                  <input
                    type="file"
                    multiple
                    accept=".pdf,.docx,.txt"
                    onChange={(e) => handleFileSelection(e.target.files)}
                    className="file-input"
                    id="file-input"
                  />
                  <label htmlFor="file-input" className="file-input-label">
                    {isDragging ? 'Drop files here' : 'Choose Files or Drag & Drop'}
                  </label>
                  <p className="supported-formats">
                    Supported formats: PDF, DOCX, TXT
                  </p>
                </div>

                {/* Selected Files */}
                {selectedFiles.length > 0 && (
                  <div className="selected-files">
                    <h4>Selected Files ({selectedFiles.length})</h4>
                    {selectedFiles.map((fileItem) => (
                      <div key={fileItem.id} className="file-item">
                        <div className="file-info">
                          <span className="file-name">{fileItem.file.name}</span>
                          <span className="file-size">{formatFileSize(fileItem.file.size)}</span>
                          <span className={`file-status ${fileItem.status}`}>
                            {fileItem.status === 'pending' ? 'Ready' : 
                             fileItem.status === 'completed' ? 'Processed' : 'Error'}
                          </span>
                        </div>
                        <button 
                          className="remove-file-btn"
                          onClick={() => removeFile(fileItem.id)}
                          disabled={isUploading}
                        >
                          Remove
                        </button>
                      </div>
                    ))}
                    
                    <button 
                      className="upload-btn"
                      onClick={handleUpload}
                      disabled={isUploading || selectedFiles.length === 0}
                    >
                      {isUploading ? 'Processing...' : `Upload ${selectedFiles.length} File(s)`}
                    </button>
                  </div>
                )}

                {/* Upload Status */}
                {uploadStatus && (
                  <div className={`upload-status ${uploadStatus.includes('Error') ? 'error' : 'success'}`}>
                    {uploadStatus}
                  </div>
                )}
              </div>

              {/* Statistics Display */}
              <div className="stats-section">
                <h3>Processing Statistics</h3>
                <div className="stats-grid">
                  <div className="stat-item">
                    <span className="stat-value">{stats.fileCount}</span>
                    <span className="stat-label">Files Uploaded</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-value">{stats.totalChunks}</span>
                    <span className="stat-label">Document Chunks</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-value">{stats.embeddingSize} MB</span>
                    <span className="stat-label">Embedding Size</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-value">{stats.vectorIndexSize} MB</span>
                    <span className="stat-label">Vector Index Size</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'library' && (
            <div className="library-tab">
              {/* Document List */}
              <div className="documents-section">
                <h3>Uploaded Documents ({documents.length})</h3>
                {documents.length === 0 ? (
                  <div className="empty-state">
                    <p>No documents uploaded yet. Start by uploading your first documents in the Upload tab.</p>
                  </div>
                ) : (
                  <div className="documents-list">
                    {documents.map((doc) => (
                      <div key={doc.id} className="document-card">
                        <div className="document-info">
                          <h4 className="document-name">{doc.filename}</h4>
                          <p className="document-meta">
                            {formatFileSize(doc.size)} • {new Date(doc.upload_date).toLocaleDateString()}
                          </p>
                          <span className={`vectorization-status ${doc.vectorized ? 'vectorized' : 'not-vectorized'}`}>
                            {doc.vectorized ? 'Vectorized' : 'Not Vectorized'}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Feature Descriptions */}
              <div className="features-section">
                <h3>Available Features</h3>
                <div className="feature-grid">
                  <div className="feature-card">
                    <h4>Semantic Search</h4>
                    <p>Find relevant information across all your documents using natural language queries powered by vector similarity.</p>
                  </div>
                  <div className="feature-card">
                    <h4>ESG Report Extraction</h4>
                    <p>Automatic extraction and analysis of Environmental, Social, and Governance metrics from uploaded documents.</p>
                  </div>
                  <div className="feature-card">
                    <h4>Real-Time LLM QA</h4>
                    <p>Ask questions about your documents and get contextualized answers using GPT-4 and retrieved content.</p>
                  </div>
                  <div className="feature-card">
                    <h4>Data Security</h4>
                    <p>Your documents are processed securely with enterprise-grade encryption and compliance standards.</p>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Process Explanation */}
      <div className="process-explanation">
        <h3>How the RAG System Works</h3>
        <div className="process-steps">
          <div className="process-step">
            <div className="step-number">1</div>
            <div className="step-content">
              <h4>Choose Vector Store</h4>
              <p>Select between FAISS (local) or Azure Cognitive Search (cloud) for document indexing.</p>
            </div>
          </div>
          <div className="process-step">
            <div className="step-number">2</div>
            <div className="step-content">
              <h4>Enable or Disable RAG</h4>
              <p>If enabled, documents are embedded and indexed. Otherwise, they are stored only.</p>
            </div>
          </div>
          <div className="process-step">
            <div className="step-number">3</div>
            <div className="step-content">
              <h4>Smart Processing</h4>
              <p>Uses Azure OpenAI for vectorization and indexes into the selected store.</p>
            </div>
          </div>
          <div className="process-step">
            <div className="step-number">4</div>
            <div className="step-content">
              <h4>Ask Questions</h4>
              <p>Uses Prompt Flow + GPT-4 to answer your queries based on stored content.</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocumentUploadPage;
