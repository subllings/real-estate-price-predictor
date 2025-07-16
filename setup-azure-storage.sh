#!/bin/bash
# Configuration automatique d'Azure Storage pour votre projet ML

echo "🚀 Configuration Azure Storage pour real-estate-price-predictor"
echo "=================================================="

# Variables
STORAGE_ACCOUNT="immoelizastorage"
RESOURCE_GROUP="immoeliza-ml"
CONTAINER_NAME="ml-models"

echo "📊 Récupération de la connection string..."

# Récupérer la connection string
CONNECTION_STRING=$(az storage account show-connection-string \
    --name $STORAGE_ACCOUNT \
    --resource-group $RESOURCE_GROUP \
    --query connectionString \
    --output tsv)

if [ -z "$CONNECTION_STRING" ]; then
    echo "❌ Erreur: Impossible de récupérer la connection string"
    echo "💡 Vérifiez que vous êtes bien connecté à Azure et que le storage account existe"
    exit 1
fi

echo "✅ Connection string récupérée avec succès!"

# Créer le container pour les modèles ML
echo "📦 Création du container '$CONTAINER_NAME'..."
az storage container create \
    --name $CONTAINER_NAME \
    --connection-string "$CONNECTION_STRING" \
    --public-access off

if [ $? -eq 0 ]; then
    echo "✅ Container '$CONTAINER_NAME' créé avec succès!"
else
    echo "⚠️  Container '$CONTAINER_NAME' existe déjà ou erreur de création"
fi

# Mettre à jour le fichier .env
echo "📝 Mise à jour du fichier .env..."

# Backup du .env existant
if [ -f .env ]; then
    cp .env .env.backup
    echo "💾 Backup du .env créé (.env.backup)"
fi

# Ajouter ou mettre à jour les variables Azure
ENV_FILE=".env"

# Fonction pour ajouter/mettre à jour une variable dans .env
update_env_var() {
    local var_name=$1
    local var_value=$2
    
    if grep -q "^$var_name=" "$ENV_FILE" 2>/dev/null; then
        # Variable existe, la mettre à jour
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s|^$var_name=.*|$var_name=\"$var_value\"|" "$ENV_FILE"
        else
            # Linux/Windows
            sed -i "s|^$var_name=.*|$var_name=\"$var_value\"|" "$ENV_FILE"
        fi
        echo "🔄 Mis à jour: $var_name"
    else
        # Variable n'existe pas, l'ajouter
        echo "" >> "$ENV_FILE"
        echo "# Azure Storage Configuration" >> "$ENV_FILE"
        echo "$var_name=\"$var_value\"" >> "$ENV_FILE"
        echo "➕ Ajouté: $var_name"
    fi
}

# Mettre à jour les variables Azure
update_env_var "AZURE_STORAGE_CONNECTION_STRING" "$CONNECTION_STRING"
update_env_var "AZURE_MODELS_CONTAINER" "$CONTAINER_NAME"

echo ""
echo "🎉 Configuration Azure Storage terminée avec succès!"
echo "=================================================="
echo "📋 Résumé de la configuration:"
echo "   • Storage Account: $STORAGE_ACCOUNT"
echo "   • Resource Group: $RESOURCE_GROUP" 
echo "   • Container: $CONTAINER_NAME"
echo "   • Fichier .env mis à jour"
echo ""
echo "🚀 Prochaines étapes:"
echo "   1. Vérifiez votre fichier .env"
echo "   2. Testez l'upload d'un modèle avec python -c \"from utils.azure_model_storage import AzureModelStorage; print('✅ Azure Storage configuré!')\""
echo "   3. Lancez votre entraînement nocturne avec upload automatique"
echo ""
echo "💡 Votre système est maintenant prêt pour l'upload automatique des modèles vers Azure!"
