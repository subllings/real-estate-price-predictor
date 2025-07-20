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
      setUploadStatus('Some files were ignored. Only PDF, DOCX and TXT files are supported.');
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

  const getFileType = (fileName) => {
    const extension = fileName.toLowerCase().split('.').pop();
    switch (extension) {
      case 'pdf':
        return 'PDF';
      case 'docx':
        return 'DOCX';
      case 'txt':
        return 'TXT';
      default:
        return 'FILE';
    }
  };

  const uploadFiles = async () => {
    if (selectedFiles.length === 0) return;

    setIsUploading(true);
    setUploadStatus('Processing documents...');

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

      setUploadStatus(`Successfully processed ${selectedFiles.length} document(s)!`);
      setSelectedFiles([]);
      setUploadProgress({});
      
      // Reload documents and stats
      await loadDocuments();
      await loadIndexStats();

    } catch (error) {
      console.error('Upload error:', error);
      setUploadStatus(`Error during processing: ${error.message}`);
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
        setUploadStatus('Document deleted successfully');
      } else {
        throw new Error('Failed to delete document');
      }
    } catch (error) {
      console.error('Delete error:', error);
      setUploadStatus('Error during deletion');
    }
  };

  return (
    <div className="document-upload-page">
      {/* Page Header */}
      <div className="page-header">
        <h1>RAG Document Processing System</h1>
        <p className="page-subtitle">
          Upload your ESG, legal and energy documents for intelligent analysis
        </p>
      </div>

      {/* RAG Explanation */}
      <div className="rag-explanation">
        <h3>How our RAG (Retrieval-Augmented Generation) system works</h3>
        <div className="rag-steps">
          <div className="rag-step">
            <div className="step-number">1</div>
            <div className="step-content">
              <strong>Content Extraction</strong>
              <p>The system automatically extracts text from your PDF, DOCX and TXT documents with maximum precision.</p>
            </div>
          </div>
          <div className="rag-step">
            <div className="step-number">2</div>
            <div className="step-content">
              <strong>Intelligent Segmentation</strong>
              <p>Content is split into logical segments optimized for analysis and semantic search.</p>
            </div>
          </div>
          <div className="rag-step">
            <div className="step-number">3</div>
            <div className="step-content">
              <strong>FAISS Indexing</strong>
              <p>Each segment is transformed into vector embeddings and indexed in a high-performance FAISS database.</p>
            </div>
          </div>
          <div className="rag-step">
            <div className="step-number">4</div>
            <div className="step-content">
              <strong>Azure OpenAI Analysis</strong>
              <p>Queries use GPT-4 and text-embedding-ada-002 for contextualized and accurate responses.</p>
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
            Choose Files
          </label>
          <p>or drag and drop your documents here</p>
          <div className="supported-formats">
            <strong>Supported formats:</strong> PDF, DOCX, TXT
          </div>
        </div>

        {/* Selected Files */}
        {selectedFiles.length > 0 && (
          <div className="selected-files">
            <h4>Selected Files ({selectedFiles.length})</h4>
            {selectedFiles.map((fileItem) => (
              <div key={fileItem.id} className="file-item">
                <span className="file-type">{getFileType(fileItem.file.name)}</span>
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
                  Remove
                </button>
              </div>
            ))}
            
            <button 
              className="upload-submit-btn"
              onClick={uploadFiles}
              disabled={isUploading || selectedFiles.length === 0}
            >
              {isUploading ? 'Processing...' : `Process ${selectedFiles.length} document(s)`}
            </button>
          </div>
        )}

        {/* Status Message */}
        {uploadStatus && (
          <div className={`status-message ${uploadStatus.includes('Successfully') ? 'success' : 'error'}`}>
            {uploadStatus}
          </div>
        )}
      </div>

      {/* Index Statistics */}
      <div className="index-stats">
        <h3>FAISS Index Statistics</h3>
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
            <span className="stat-label">Total Size</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{indexStats.index_size_mb} MB</span>
            <span className="stat-label">FAISS Index</span>
          </div>
        </div>
      </div>

      {/* Documents List */}
      <div className="documents-list">
        <h3>Indexed Documents ({documents.length})</h3>
        {documents.length === 0 ? (
          <div className="empty-state">
            <p>No documents indexed. Start by uploading your first documents!</p>
          </div>
        ) : (
          <div className="documents-grid">
            {documents.map((doc) => (
              <div key={doc.id} className="document-card">
                <div className="document-header">
                  <span className="doc-type">{getFileType(doc.filename)}</span>
                  <div className="doc-info">
                    <h4 className="doc-name">{doc.filename}</h4>
                    <p className="doc-meta">
                      {formatFileSize(doc.size_bytes)} • {doc.chunks_count} segments • {new Date(doc.upload_time).toLocaleDateString('en-US')}
                    </p>
                  </div>
                  <button 
                    className="delete-doc-btn"
                    onClick={() => deleteDocument(doc.id)}
                    title="Delete document"
                  >
                    Delete
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
        <h3>Technical Architecture</h3>
        <div className="stack-items">
          <div className="stack-item">
            <strong>FAISS Vector Database:</strong> High-performance vector indexing for semantic search
          </div>
          <div className="stack-item">
            <strong>LangChain Framework:</strong> RAG workflow orchestration and embeddings management
          </div>
          <div className="stack-item">
            <strong>Azure OpenAI:</strong> GPT-4 for generation and text-embedding-ada-002 for vectorization
          </div>
          <div className="stack-item">
            <strong>PyMuPDF + python-docx:</strong> Multi-format text extraction with structure preservation
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocumentUploadPage;
