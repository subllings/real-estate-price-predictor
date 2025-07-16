import os
import pickle
import json
from datetime import datetime
from azure.storage.blob import BlobServiceClient, BlobClient
from azure.core.exceptions import ResourceNotFoundError
from dotenv import load_dotenv
from typing import Optional, Dict, Any, List
import pandas as pd

load_dotenv()

class AzureModelStorage:
    """
    Gestionnaire de stockage des modèles PKL sur Azure Blob Storage
    Permet upload, download automatique et injection dans FastAPI
    """
    
    def __init__(self):
        self.connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.container_name = os.getenv("AZURE_MODELS_CONTAINER", "ml-models")
        
        if not self.connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING manquant dans .env")
        
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self._ensure_container_exists()
    
    def _ensure_container_exists(self):
        """Créer le container s'il n'existe pas"""
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            container_client.get_container_properties()
        except ResourceNotFoundError:
            self.blob_service_client.create_container(self.container_name)
            print(f"[INFO] Container '{self.container_name}' créé sur Azure")
    
    def upload_model(self, model_path: str, model_metadata: Dict[str, Any]) -> str:
        """
        Upload un modèle PKL vers Azure avec métadonnées
        
        Args:
            model_path: Chemin local vers le fichier .pkl
            model_metadata: Métadonnées du modèle (R², MAE, RMSE, etc.)
        
        Returns:
            URL Azure du modèle uploadé
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modèle non trouvé: {model_path}")
        
        # Générer nom unique pour Azure
        model_name = os.path.basename(model_path)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        azure_blob_name = f"models/{timestamp}_{model_name}"
        
        try:
            # Upload du fichier PKL
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=azure_blob_name
            )
            
            with open(model_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)
            
            # Upload des métadonnées (JSON)
            metadata_blob_name = azure_blob_name.replace('.pkl', '_metadata.json')
            metadata_blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=metadata_blob_name
            )
            
            enhanced_metadata = {
                **model_metadata,
                "upload_timestamp": datetime.utcnow().isoformat(),
                "azure_blob_name": azure_blob_name,
                "local_path": model_path,
                "model_size_bytes": os.path.getsize(model_path)
            }
            
            metadata_blob_client.upload_blob(
                json.dumps(enhanced_metadata, indent=2).encode(),
                overwrite=True
            )
            
            blob_url = blob_client.url
            print(f"[✅] Modèle uploadé: {azure_blob_name}")
            print(f"[✅] URL: {blob_url}")
            
            return blob_url
            
        except Exception as e:
            print(f"[❌] Erreur upload Azure: {e}")
            raise
    
    def download_model(self, azure_blob_name: str, local_path: str) -> bool:
        """
        Download un modèle depuis Azure vers un chemin local
        
        Args:
            azure_blob_name: Nom du blob sur Azure
            local_path: Chemin de destination local
        
        Returns:
            True si succès, False sinon
        """
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=azure_blob_name
            )
            
            # Créer le dossier de destination
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            with open(local_path, "wb") as download_file:
                download_file.write(blob_client.download_blob().readall())
            
            print(f"[✅] Modèle téléchargé: {local_path}")
            return True
            
        except Exception as e:
            print(f"[❌] Erreur download Azure: {e}")
            return False
    
    def get_best_model_info(self) -> Optional[Dict[str, Any]]:
        """
        Récupérer les infos du meilleur modèle basé sur R²
        
        Returns:
            Dictionnaire avec infos du meilleur modèle ou None
        """
        try:
            # Lister tous les fichiers métadonnées
            container_client = self.blob_service_client.get_container_client(self.container_name)
            metadata_blobs = [
                blob for blob in container_client.list_blobs(name_starts_with="models/")
                if blob.name.endswith("_metadata.json")
            ]
            
            if not metadata_blobs:
                return None
            
            best_model = None
            best_r2 = -float('inf')
            
            for blob in metadata_blobs:
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=blob.name
                )
                
                metadata_json = blob_client.download_blob().readall().decode()
                metadata = json.loads(metadata_json)
                
                # Récupérer R² (plusieurs formats possibles)
                r2 = metadata.get("r2_test") or metadata.get("r2") or metadata.get("metrics", {}).get("test", {}).get("r2", 0)
                
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = metadata
            
            if best_model:
                print(f"[✅] Meilleur modèle trouvé: R² = {best_r2:.4f}")
            
            return best_model
            
        except Exception as e:
            print(f"[❌] Erreur recherche meilleur modèle: {e}")
            return None
    
    def download_best_model(self, local_path: str = "models/current_best_model.pkl") -> bool:
        """
        Télécharger automatiquement le meilleur modèle pour FastAPI
        
        Args:
            local_path: Chemin de destination (par défaut pour FastAPI)
        
        Returns:
            True si succès, False sinon
        """
        best_model_info = self.get_best_model_info()
        
        if not best_model_info:
            print("[❌] Aucun modèle trouvé sur Azure")
            return False
        
        azure_blob_name = best_model_info.get("azure_blob_name")
        if not azure_blob_name:
            print("[❌] Nom blob Azure manquant dans les métadonnées")
            return False
        
        success = self.download_model(azure_blob_name, local_path)
        
        if success:
            # Sauvegarder aussi les métadonnées pour l'API
            metadata_path = local_path.replace('.pkl', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(best_model_info, f, indent=2)
            
            print(f"[✅] Meilleur modèle prêt pour FastAPI: {local_path}")
            print(f"[📊] R² = {best_model_info.get('r2_test', 'N/A')}")
        
        return success
    
    def list_all_models(self) -> List[Dict[str, Any]]:
        """
        Lister tous les modèles disponibles sur Azure avec métadonnées
        
        Returns:
            Liste des modèles avec leurs métadonnées
        """
        try:
            container_client = self.blob_service_client.get_container_client(self.container_name)
            metadata_blobs = [
                blob for blob in container_client.list_blobs(name_starts_with="models/")
                if blob.name.endswith("_metadata.json")
            ]
            
            models = []
            for blob in metadata_blobs:
                blob_client = self.blob_service_client.get_blob_client(
                    container=self.container_name,
                    blob=blob.name
                )
                
                metadata_json = blob_client.download_blob().readall().decode()
                metadata = json.loads(metadata_json)
                models.append(metadata)
            
            # Trier par R² décroissant
            models.sort(key=lambda x: x.get("r2_test", 0), reverse=True)
            
            print(f"[✅] {len(models)} modèles trouvés sur Azure")
            return models
            
        except Exception as e:
            print(f"[❌] Erreur listing modèles: {e}")
            return []


# Fonction helper pour intégration avec le model_saver existant
def upload_model_to_azure(model_path: str, metrics: Dict[str, Any]) -> Optional[str]:
    """
    Helper function pour upload automatique depuis model_saver
    
    Args:
        model_path: Chemin vers le modèle .pkl
        metrics: Métadonnées du modèle
    
    Returns:
        URL Azure ou None si erreur
    """
    try:
        storage = AzureModelStorage()
        return storage.upload_model(model_path, metrics)
    except Exception as e:
        print(f"[❌] Upload Azure échoué: {e}")
        return None


# Fonction pour injection automatique dans FastAPI
def ensure_best_model_available(api_model_path: str = "models/current_best_model.pkl") -> bool:
    """
    S'assurer que le meilleur modèle est disponible pour l'API
    
    Args:
        api_model_path: Chemin où l'API s'attend à trouver le modèle
    
    Returns:
        True si modèle disponible, False sinon
    """
    try:
        # Vérifier si on a déjà un modèle local récent
        if os.path.exists(api_model_path):
            # Vérifier l'âge du fichier (si < 1h, on garde)
            file_age = datetime.now().timestamp() - os.path.getmtime(api_model_path)
            if file_age < 3600:  # 1 heure
                print(f"[✅] Modèle local récent disponible: {api_model_path}")
                return True
        
        # Sinon, télécharger le meilleur depuis Azure
        storage = AzureModelStorage()
        return storage.download_best_model(api_model_path)
        
    except Exception as e:
        print(f"[❌] Erreur availability check: {e}")
        return False
