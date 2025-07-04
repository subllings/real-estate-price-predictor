#!/bin/bash

# Make this script executable: chmod +x cloud/azure/azure_deploy_frontend.sh
# Run with: ./cloud/azure/azure_deploy_frontend.sh

clear
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

RESOURCE_GROUP="realestate-api-rg-north"
LOCATION="northeurope"
ACR_NAME="realestateacrneo2"
APP_SERVICE_PLAN="realestate-api-plan-north"
WEBAPP_NAME="realestate-ui"

FRONTEND_IMAGE="$ACR_NAME.azurecr.io/real-estate-frontend:latest"
DOCKERFILE_PATH="app/frontend-streamlit/Dockerfile.azure"
FRONTEND_DIR="app/frontend-streamlit"

echo "Creating ACR if needed..."
az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null || {
  az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic --location "$LOCATION"
}

echo "Logging into ACR..."
az acr login --name "$ACR_NAME"

echo "Building Docker image..."
docker build -f "$DOCKERFILE_PATH" -t "$FRONTEND_IMAGE" "$FRONTEND_DIR"

echo "Pushing image to ACR..."
docker push "$FRONTEND_IMAGE"

echo "Retrieving ACR credentials..."
az acr update -n "$ACR_NAME" --admin-enabled true
ACR_USERNAME=$(az acr credential show -n "$ACR_NAME" --query "username" -o tsv)
ACR_PASSWORD=$(az acr credential show -n "$ACR_NAME" --query "passwords[0].value" -o tsv)

echo "Creating Web App for Streamlit UI..."
az webapp create \
  --resource-group "$RESOURCE_GROUP" \
  --plan "$APP_SERVICE_PLAN" \
  --name "$WEBAPP_NAME" \
  --deployment-container-image-name "$FRONTEND_IMAGE"

echo "Configuring container credentials..."
az webapp config container set \
  --name "$WEBAPP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --docker-custom-image-name "$FRONTEND_IMAGE" \
  --docker-registry-server-url "https://$ACR_NAME.azurecr.io" \
  --docker-registry-server-user "$ACR_USERNAME" \
  --docker-registry-server-password "$ACR_PASSWORD"

echo "Setting required environment variables (port 8501 + API_URL)..."
az webapp config appsettings set \
  --name "$WEBAPP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --settings PORT=8501 API_URL=https://realestate-api.azurewebsites.net

echo "Restarting frontend Web App..."
az webapp restart \
  --name "$WEBAPP_NAME" \
  --resource-group "$RESOURCE_GROUP"

echo ""
echo "Streamlit frontend deployed:"
echo "https://$WEBAPP_NAME.azurewebsites.net"

az webapp log config \
  --name realestate-ui \
  --resource-group realestate-api-rg-north \
  --docker-container-logging filesystem



