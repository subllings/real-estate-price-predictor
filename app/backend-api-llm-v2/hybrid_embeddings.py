import os
from openai import OpenAI
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv

load_dotenv()

class HybridEmbeddingService:
    def __init__(self):
        # OpenAI standard for embeddings
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Azure Search for storage
        self.search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        self.search_key = os.getenv("AZURE_SEARCH_API_KEY")
        self.index_name = os.getenv("AZURE_SEARCH_INDEX_NAME", "documents")
        
        self.search_client = SearchClient(
            endpoint=self.search_endpoint,
            index_name=self.index_name,
            credential=AzureKeyCredential(self.search_key)
        )
    
    def create_embeddings(self, text: str):
        """Create embeddings using OpenAI standard API"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    
    def store_in_azure_search(self, doc_id: str, content: str, embeddings: list):
        """Store document and embeddings in Azure Cognitive Search"""
        document = {
            "id": doc_id,
            "content": content,
            "contentVector": embeddings
        }
        
        result = self.search_client.upload_documents([document])
        return result
