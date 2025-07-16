#!/usr/bin/env python3
"""
Test de la configuration Azure Storage
"""
try:
    print("🔄 Test de l'Azure Storage...")
    from utils.azure_model_storage import AzureModelStorage
    
    print("📦 Initialisation du stockage...")
    storage = AzureModelStorage()
    
    print("✅ Azure Storage configuré avec succès!")
    print(f"📦 Container: {storage.container_name}")
    print("🔗 Connexion établie vers Azure Blob Storage")
    
    # Test de listing des modèles
    print("📊 Listing des modèles disponibles...")
    models = storage.list_all_models()
    print(f"📊 Modèles disponibles: {len(models)}")
    
    if models:
        print("\n🏆 Modèles trouvés:")
        for i, model in enumerate(models[:3], 1):  # Afficher les 3 premiers
            r2 = model.get('r2_test', 'N/A')
            timestamp = model.get('upload_timestamp', 'N/A')
            print(f"  {i}. R² = {r2} | {timestamp}")
    else:
        print("📝 Aucun modèle trouvé (normal si premier usage)")
    
    print("\n🎉 Configuration Azure Storage VALIDÉE!")
    
except ImportError as e:
    print(f"❌ Erreur d'import: {e}")
    print("💡 Vérifiez que azure-storage-blob est installé: pip install azure-storage-blob")
    
except Exception as e:
    print(f"❌ Erreur: {e}")
    print("💡 Vérifiez votre connection string Azure dans .env")
