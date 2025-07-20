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

    setSelectedFiles(prevFiles => [...prevFiles, ...validFiles]);
  };

  const removeFile = (index) => {
    setSelectedFiles(prevFiles => prevFiles.filter((_, i) => i !== index));
  };

  const uploadDocuments = async () => {
    if (selectedFiles.length === 0) {
      setStatusMessage('⚠️ Please select at least one document to upload.');
      return;
    }

    setUploading(true);
    setStatusMessage('');
    const newProgress = {};

    try {
      for (let i = 0; i < selectedFiles.length; i++) {
        const file = selectedFiles[i];
        const formData = new FormData();
        formData.append('file', file);

        // Update progress for current file
        newProgress[file.name] = 0;
        setUploadProgress({ ...newProgress });

        try {
          const response = await axios.post(UPLOAD_API_URL, formData, {
            headers: {
              'Content-Type': 'multipart/form-data',
            },
            onUploadProgress: (progressEvent) => {
              const percentCompleted = Math.round(
                (progressEvent.loaded * 100) / progressEvent.total
              );
              newProgress[file.name] = percentCompleted;
              setUploadProgress({ ...newProgress });
            },
          });

          newProgress[file.name] = 100;
          setUploadProgress({ ...newProgress });

          console.log(`Upload successful for ${file.name}:`, response.data);
        } catch (error) {
          console.error(`Upload failed for ${file.name}:`, error);
          newProgress[file.name] = -1; // Error state
          setUploadProgress({ ...newProgress });
        }
      }

      // Refresh the document list and index stats
      await fetchUploadedDocuments();
      await fetchIndexStats();
      
      setStatusMessage('✅ All documents processed and indexed successfully! Ready for RAG queries.');
      setSelectedFiles([]);
      setUploadProgress({});
      
    } catch (error) {
      console.error('Upload process failed:', error);
      setStatusMessage('❌ Upload process failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  const deleteDocument = async (docId) => {
    try {
      await axios.delete(`${DOCUMENTS_API_URL}/${docId}`);
      await fetchUploadedDocuments();
      await fetchIndexStats();
      setStatusMessage('🗑️ Document removed from index successfully.');
    } catch (error) {
      console.error('Failed to delete document:', error);
      setStatusMessage('❌ Failed to delete document.');
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getFileIcon = (filename) => {
    const extension = filename.split('.').pop().toLowerCase();
    switch (extension) {
      case 'pdf': return '📄';
      case 'txt': return '📝';
      case 'docx': return '📘';
      default: return '📎';
    }
  };

  return (
    <div className="document-upload-page">
      <div className="page-header">
        <h1>📤 ESG Document Upload & RAG Indexing</h1>
        <p className="page-subtitle">
          Upload ESG documents to enhance AI responses with your specific content
        </p>
      </div>

      {/* RAG Context Explanation */}
      <div className="rag-explanation">
        <h3>🧠 How RAG (Retrieval-Augmented Generation) Works</h3>
        <div className="rag-steps">
          <div className="rag-step">
            <span className="step-number">1</span>
            <div className="step-content">
              <strong>Document Upload</strong>
              <p>Your documents are processed and split into semantic chunks</p>
            </div>
          </div>
          <div className="rag-step">
            <span className="step-number">2</span>
            <div className="step-content">
              <strong>Vector Indexing</strong>
              <p>Text chunks are converted to embeddings using Azure OpenAI and stored in FAISS</p>
            </div>
          </div>
          <div className="rag-step">
            <span className="step-number">3</span>
            <div className="step-content">
              <strong>Smart Retrieval</strong>
              <p>When you ask questions, relevant document sections are automatically found</p>
            </div>
          </div>
          <div className="rag-step">
            <span className="step-number">4</span>
            <div className="step-content">
              <strong>Enhanced Responses</strong>
              <p>AI generates answers using both its knowledge and your documents</p>
            </div>
          </div>
        </div>
      </div>

      {/* Upload Section */}
      <div className="upload-section">
        <div className="upload-area">
          <input
            type="file"
            id="file-input"
            multiple
            accept=".pdf,.txt,.docx"
            onChange={handleFileSelect}
            className="file-input-hidden"
          />
          <label htmlFor="file-input" className="upload-button">
            📁 Select ESG Documents
          </label>
          <div className="supported-formats">
            Supported formats: PDF, TXT, DOCX
          </div>
        </div>

        {selectedFiles.length > 0 && (
          <div className="selected-files">
            <h4>Selected Files ({selectedFiles.length})</h4>
            {selectedFiles.map((file, index) => (
              <div key={index} className="file-item">
                <span className="file-icon">{getFileIcon(file.name)}</span>
                <div className="file-details">
                  <span className="file-name">{file.name}</span>
                  <span className="file-size">{formatFileSize(file.size)}</span>
                </div>
                {uploadProgress[file.name] !== undefined && (
                  <div className="progress-container">
                    <div className="progress-bar">
                      <div 
                        className="progress-fill"
                        style={{ 
                          width: `${uploadProgress[file.name]}%`,
                          backgroundColor: uploadProgress[file.name] === -1 ? '#dc3545' : '#28a745'
                        }}
                      ></div>
                    </div>
                    <span className="progress-text">
                      {uploadProgress[file.name] === -1 ? 'Error' : `${uploadProgress[file.name]}%`}
                    </span>
                  </div>
                )}
                <button 
                  onClick={() => removeFile(index)}
                  className="remove-file-btn"
                  disabled={uploading}
                >
                  ✕
                </button>
              </div>
            ))}
            
            <button 
              onClick={uploadDocuments}
              disabled={uploading || selectedFiles.length === 0}
              className="upload-submit-btn"
            >
              {uploading ? '⏳ Processing & Indexing...' : '🚀 Upload & Index Documents'}
            </button>
          </div>
        )}

        {statusMessage && (
          <div className={`status-message ${statusMessage.includes('❌') ? 'error' : 'success'}`}>
            {statusMessage}
          </div>
        )}
      </div>

      {/* Index Statistics */}
      {indexStats && (
        <div className="index-stats">
          <h3>📊 Vector Index Statistics</h3>
          <div className="stats-grid">
            <div className="stat-item">
              <span className="stat-value">{indexStats.total_documents || 0}</span>
              <span className="stat-label">Documents Indexed</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{indexStats.total_chunks || 0}</span>
              <span className="stat-label">Text Chunks</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{indexStats.vector_dimension || 0}</span>
              <span className="stat-label">Vector Dimension</span>
            </div>
            <div className="stat-item">
              <span className="stat-value">{formatFileSize(indexStats.index_size_bytes || 0)}</span>
              <span className="stat-label">Index Size</span>
            </div>
          </div>
        </div>
      )}

      {/* Uploaded Documents List */}
      <div className="documents-list">
        <h3>📚 Indexed Documents ({uploadedDocuments.length})</h3>
        {uploadedDocuments.length === 0 ? (
          <div className="empty-state">
            <p>No documents uploaded yet. Upload your first ESG document to get started!</p>
          </div>
        ) : (
          <div className="documents-grid">
            {uploadedDocuments.map((doc, index) => (
              <div key={index} className="document-card">
                <div className="document-header">
                  <span className="doc-icon">{getFileIcon(doc.filename)}</span>
                  <div className="doc-info">
                    <h4 className="doc-name">{doc.filename}</h4>
                    <p className="doc-meta">
                      {formatFileSize(doc.size)} • {doc.chunks} chunks • 
                      {new Date(doc.uploaded_at).toLocaleDateString()}
                    </p>
                  </div>
                  <button 
                    onClick={() => deleteDocument(doc.id)}
                    className="delete-doc-btn"
                    title="Remove from index"
                  >
                    🗑️
                  </button>
                </div>
                <div className="document-preview">
                  <p>{doc.preview || 'Document content available for RAG queries...'}</p>
                </div>
                <div className="document-tags">
                  {doc.tags && doc.tags.map((tag, i) => (
                    <span key={i} className="tag">{tag}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Technical Stack Info */}
      <div className="tech-stack">
        <h3>🔧 Technical Stack</h3>
        <div className="stack-items">
          <div className="stack-item">
            <strong>Vector Store:</strong> FAISS (Facebook AI Similarity Search)
          </div>
          <div className="stack-item">
            <strong>Embeddings:</strong> Azure OpenAI text-embedding-ada-002
          </div>
          <div className="stack-item">
            <strong>LLM:</strong> Azure OpenAI GPT-4 / GPT-3.5-turbo
          </div>
          <div className="stack-item">
            <strong>Framework:</strong> LangChain + FastAPI + React
          </div>
        </div>
      </div>
    </div>
  );
};

export default DocumentUploadPage;
