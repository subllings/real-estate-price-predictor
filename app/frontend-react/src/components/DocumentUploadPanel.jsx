/**
 * Document Upload Panel - RAG document management
 */

import React, { useState, useRef } from 'react';
import { Upload, FileText, CheckCircle, AlertCircle, Trash2 } from 'lucide-react';

const DocumentUploadPanel = () => {
  const [documents, setDocuments] = useState([
    {
      id: 1,
      name: 'Belgian Housing Market Q3 2025.pdf',
      size: '2.4 MB',
      status: 'processed',
      category: 'market_intelligence',
      uploaded_at: '2025-07-15T10:30:00Z',
      chunks: 45,
      vectors: 45
    },
    {
      id: 2,
      name: 'Antwerp Zoning Regulations.docx',
      size: '1.8 MB', 
      status: 'processing',
      category: 'regulatory',
      uploaded_at: '2025-07-15T14:20:00Z',
      chunks: 32,
      vectors: 28
    },
    {
      id: 3,
      name: 'Property Tax Guide 2025.pdf',
      size: '890 KB',
      status: 'processed',
      category: 'fiscal',
      uploaded_at: '2025-07-14T16:45:00Z',
      chunks: 23,
      vectors: 23
    }
  ]);

  const [uploading, setUploading] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState('market_intelligence');
  const fileInputRef = useRef(null);

  const categories = [
    { id: 'market_intelligence', label: '📈 Market Intelligence', color: 'blue' },
    { id: 'regulatory', label: '🏛️ Regulatory', color: 'green' },
    { id: 'fiscal', label: '💰 Fiscal', color: 'purple' },
    { id: 'legal', label: '📜 Legal', color: 'red' }
  ];

  const handleFileUpload = async (event) => {
    const files = Array.from(event.target.files);
    if (files.length === 0) return;

    setUploading(true);

    for (const file of files) {
      // Simulate upload process
      const newDoc = {
        id: Date.now() + Math.random(),
        name: file.name,
        size: formatFileSize(file.size),
        status: 'uploading',
        category: selectedCategory,
        uploaded_at: new Date().toISOString(),
        chunks: 0,
        vectors: 0
      };

      setDocuments(prev => [newDoc, ...prev]);

      // Simulate processing
      setTimeout(() => {
        setDocuments(prev => 
          prev.map(doc => 
            doc.id === newDoc.id 
              ? { ...doc, status: 'processing', chunks: Math.floor(Math.random() * 30) + 10 }
              : doc
          )
        );
      }, 2000);

      setTimeout(() => {
        setDocuments(prev => 
          prev.map(doc => 
            doc.id === newDoc.id 
              ? { ...doc, status: 'processed', vectors: doc.chunks }
              : doc
          )
        );
      }, 5000);
    }

    setUploading(false);
    // Clear file input
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  const getStatusConfig = (status) => {
    const configs = {
      uploading: { color: 'blue', icon: Upload, label: '📤 Uploading' },
      processing: { color: 'yellow', icon: AlertCircle, label: '🔄 Processing' },
      processed: { color: 'green', icon: CheckCircle, label: '✅ Processed' },
      failed: { color: 'red', icon: AlertCircle, label: '❌ Failed' }
    };
    return configs[status] || configs.processing;
  };

  const getCategoryConfig = (categoryId) => {
    return categories.find(cat => cat.id === categoryId) || categories[0];
  };

  const deleteDocument = (docId) => {
    if (confirm('Are you sure you want to delete this document?')) {
      setDocuments(prev => prev.filter(doc => doc.id !== docId));
    }
  };

  const reprocessDocument = async (docId) => {
    setDocuments(prev => 
      prev.map(doc => 
        doc.id === docId 
          ? { ...doc, status: 'processing', vectors: 0 }
          : doc
      )
    );

    // Simulate reprocessing
    setTimeout(() => {
      setDocuments(prev => 
        prev.map(doc => 
          doc.id === docId 
            ? { ...doc, status: 'processed', vectors: doc.chunks }
            : doc
        )
      );
    }, 3000);
  };

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-gray-800">📄 Document Management</h3>
      
      {/* Upload Section */}
      <div className="bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg p-4 border border-dashed border-blue-300">
        <h4 className="font-medium text-gray-800 mb-3">📤 Upload Documents</h4>
        
        {/* Category Selection */}
        <div className="mb-3">
          <label className="text-sm text-gray-600 mb-2 block">Category:</label>
          <select 
            value={selectedCategory}
            onChange={(e) => setSelectedCategory(e.target.value)}
            className="w-full p-2 border rounded text-sm"
          >
            {categories.map(cat => (
              <option key={cat.id} value={cat.id}>{cat.label}</option>
            ))}
          </select>
        </div>

        {/* File Upload */}
        <div className="flex space-x-2">
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.docx,.txt,.md"
            onChange={handleFileUpload}
            className="hidden"
          />
          <button 
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            className="flex-1 bg-blue-600 text-white py-2 px-3 rounded text-sm hover:bg-blue-700 transition-colors disabled:opacity-50 flex items-center justify-center space-x-1"
          >
            <Upload size={14} />
            <span>{uploading ? 'Uploading...' : 'Choose Files'}</span>
          </button>
          <button className="bg-green-600 text-white py-2 px-3 rounded text-sm hover:bg-green-700 transition-colors">
            📎 Bulk Upload
          </button>
        </div>
        
        <p className="text-xs text-gray-500 mt-2">
          Supported: PDF, DOCX, TXT, MD • Max 10MB per file
        </p>
      </div>

      {/* Documents List */}
      <div className="space-y-3">
        <div className="flex justify-between items-center">
          <h4 className="font-medium text-gray-700">📚 Document Library</h4>
          <span className="text-sm text-gray-500">
            {documents.length} documents • {documents.reduce((sum, doc) => sum + doc.vectors, 0)} vectors
          </span>
        </div>

        {documents.map((doc) => {
          const statusConfig = getStatusConfig(doc.status);
          const categoryConfig = getCategoryConfig(doc.category);
          const IconComponent = statusConfig.icon;
          
          return (
            <div key={doc.id} className="bg-white rounded-lg border p-3 hover:shadow-md transition-shadow">
              {/* Document Header */}
              <div className="flex justify-between items-start mb-2">
                <div className="flex-1">
                  <h5 className="font-medium text-gray-800 text-sm mb-1">{doc.name}</h5>
                  <div className="flex items-center space-x-2 text-xs text-gray-500">
                    <span>{doc.size}</span>
                    <span>•</span>
                    <span className={`px-2 py-1 rounded text-${categoryConfig.color}-700 bg-${categoryConfig.color}-100`}>
                      {categoryConfig.label}
                    </span>
                  </div>
                </div>
                <span className={`text-xs bg-${statusConfig.color}-100 text-${statusConfig.color}-800 px-2 py-1 rounded flex items-center space-x-1`}>
                  <IconComponent size={10} />
                  <span>{statusConfig.label}</span>
                </span>
              </div>

              {/* Processing Progress */}
              {doc.status === 'processing' && (
                <div className="mb-2">
                  <div className="w-full bg-gray-200 rounded-full h-1">
                    <div 
                      className="bg-blue-600 h-1 rounded-full transition-all duration-500"
                      style={{ width: `${(doc.vectors / doc.chunks) * 100}%` }}
                    ></div>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    Vectorizing: {doc.vectors}/{doc.chunks} chunks
                  </p>
                </div>
              )}

              {/* Document Stats */}
              <div className="grid grid-cols-3 gap-2 mb-2 text-xs">
                <div className="text-center p-1 bg-gray-50 rounded">
                  <div className="font-medium text-blue-600">{doc.chunks}</div>
                  <div className="text-gray-500">Chunks</div>
                </div>
                <div className="text-center p-1 bg-gray-50 rounded">
                  <div className="font-medium text-green-600">{doc.vectors}</div>
                  <div className="text-gray-500">Vectors</div>
                </div>
                <div className="text-center p-1 bg-gray-50 rounded">
                  <div className="font-medium text-purple-600">
                    {new Date(doc.uploaded_at).toLocaleDateString()}
                  </div>
                  <div className="text-gray-500">Uploaded</div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex space-x-1">
                {doc.status === 'processed' && (
                  <button 
                    onClick={() => reprocessDocument(doc.id)}
                    className="flex-1 bg-blue-600 text-white py-1 px-2 rounded text-xs hover:bg-blue-700 transition-colors"
                  >
                    🔄 Reprocess
                  </button>
                )}
                <button className="flex-1 bg-green-600 text-white py-1 px-2 rounded text-xs hover:bg-green-700 transition-colors">
                  👁️ Preview
                </button>
                <button 
                  onClick={() => deleteDocument(doc.id)}
                  className="bg-red-600 text-white py-1 px-2 rounded text-xs hover:bg-red-700 transition-colors"
                >
                  <Trash2 size={10} />
                </button>
              </div>
            </div>
          );
        })}
      </div>

      {/* Vector Store Stats */}
      <div className="bg-gray-50 rounded-lg p-3 border">
        <h4 className="font-medium text-gray-700 mb-2">🗂️ Vector Store Status</h4>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div>
            <span className="text-gray-600">Total Documents:</span>
            <span className="font-medium text-blue-600 ml-1">{documents.length}</span>
          </div>
          <div>
            <span className="text-gray-600">Total Vectors:</span>
            <span className="font-medium text-green-600 ml-1">
              {documents.reduce((sum, doc) => sum + doc.vectors, 0)}
            </span>
          </div>
          <div>
            <span className="text-gray-600">Index Size:</span>
            <span className="font-medium text-purple-600 ml-1">847 MB</span>
          </div>
          <div>
            <span className="text-gray-600">Search Latency:</span>
            <span className="font-medium text-orange-600 ml-1">45ms</span>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div className="flex space-x-2">
        <button className="flex-1 bg-blue-600 text-white py-2 px-3 rounded text-sm hover:bg-blue-700 transition-colors">
          🔍 Search Test
        </button>
        <button className="flex-1 bg-green-600 text-white py-2 px-3 rounded text-sm hover:bg-green-700 transition-colors">
          📊 Analytics
        </button>
        <button className="bg-red-600 text-white py-2 px-3 rounded text-sm hover:bg-red-700 transition-colors">
          🗑️ Cleanup
        </button>
      </div>
    </div>
  );
};

export default DocumentUploadPanel;
