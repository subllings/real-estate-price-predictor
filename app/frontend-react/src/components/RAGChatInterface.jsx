```javascript
// ...existing code...

const uploadDocument = async (file) => {
  setIsUploading(true);
  const formData = new FormData();
  formData.append('file', file);

  try {
    // Use hybrid endpoint when Azure Search is selected
    const endpoint = vectorStore === 'azure_search' ? '/upload_document_hybrid' : '/upload_document';
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error('Upload failed');
    }

    const result = await response.json();
    setUploadStatus(`✅ ${result.message}`);
    
    // Show hybrid details
    console.log('Hybrid upload result:', result);
    setUploadStatus(`✅ ${result.message} (${result.embedding_provider} + ${result.storage})`);
    
    // Refresh document list
    fetchDocuments();
  } catch (error) {
    setUploadStatus(`❌ Upload failed: ${error.message}`);
    console.error('Upload error:', error);
  } finally {
    setIsUploading(false);
  }
};

const askQuestion = async () => {
  if (!question.trim()) return;

  setIsProcessing(true);
  setAnswer('');

  try {
    // Use hybrid query endpoint
    const response = await fetch(`${API_BASE_URL}/query_hybrid`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query: question }),
    });

    if (!response.ok) {
      throw new Error('Query failed');
    }

    const result = await response.json();
    
    // Format hybrid response
    let formattedAnswer = `**Query Results (${result.embedding_provider} + ${result.storage}):**\n\n`;
    
    if (result.results && result.results.length > 0) {
      result.results.forEach((doc, index) => {
        formattedAnswer += `**Document ${index + 1}:** ${doc.filename}\n`;
        formattedAnswer += `**Score:** ${doc.score?.toFixed(3)}\n`;
        formattedAnswer += `**Content:** ${doc.content}\n\n`;
      });
    } else {
      formattedAnswer += "No relevant documents found.";
    }

    setAnswer(formattedAnswer);
  } catch (error) {
    setAnswer(`❌ Error: ${error.message}`);
    console.error('Query error:', error);
  } finally {
    setIsProcessing(false);
  }
};

// ...existing code...

// Update the Vector Store display
<div className="config-item">
  <label>Vector Store:</label>
  <select value={vectorStore} onChange={(e) => setVectorStore(e.target.value)}>
    <option value="faiss">FAISS (Local)</option>
    <option value="azure_search">Azure Cognitive Search (Hybrid)</option>
  </select>
</div>

// ...existing code...
```