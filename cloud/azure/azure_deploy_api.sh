#!/bin/bash

# Make this script executable: chmod +x cloud/azure/azure_deploy_api.sh
# Run with: ./cloud/azure/azure_deploy_api.sh

set -e

# Register Microsoft.Web provider (if not already registered)
echo "Registering Microsoft.Web provider (if needed)..."
az provider register --namespace Microsoft.Web

# Resolve script path (if run from project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Configuration
RESOURCE_GROUP="realestate-api-rg-north"
LOCATION="northeurope"
ACR_NAME="realestateacrneo2"
APP_SERVICE_PLAN="realestate-api-plan-north"
WEBAPP_NAME="realestate-api"

BACKEND_IMAGE="$ACR_NAME.azurecr.io/real-estate-backend:latest"
DOCKERFILE_PATH="app/backend/Dockerfile.azure"
BACKEND_DIR="app/backend"

echo "Using configuration:"
echo "RESOURCE_GROUP=$RESOURCE_GROUP"
echo "LOCATION=$LOCATION"
echo "ACR_NAME=$ACR_NAME"
echo "APP_SERVICE_PLAN=$APP_SERVICE_PLAN"
echo "WEBAPP_NAME=$WEBAPP_NAME"

# Step 1: Create the resource group
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

# Step 2: Create the ACR if it doesn't exist
az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null || {
  echo "Creating Azure Container Registry..."
  az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic --location "$LOCATION"
}

# Step 3: Login to ACR
echo "Logging in to ACR..."
az acr login --name "$ACR_NAME"

# Step 4: Build Docker image
echo "Building backend Docker image..."
docker build -f "$DOCKERFILE_PATH" -t "$BACKEND_IMAGE" "$BACKEND_DIR"

# Step 5: Push image to ACR
echo "Pushing backend image to ACR..."
docker push "$BACKEND_IMAGE"

# Step 6: Enable ACR admin user and retrieve credentials
echo "Retrieving ACR credentials..."
az acr update -n "$ACR_NAME" --admin-enabled true
ACR_USERNAME=$(az acr credential show -n "$ACR_NAME" --query "username" -o tsv)
ACR_PASSWORD=$(az acr credential show -n "$ACR_NAME" --query "passwords[0].value" -o tsv)

# Step 6.5: Create App Service plan if not exists
az appservice plan show --name "$APP_SERVICE_PLAN" --resource-group "$RESOURCE_GROUP" &> /dev/null || {
  echo "Creating App Service plan..."
  az appservice plan create \
    --name "$APP_SERVICE_PLAN" \
    --resource-group "$RESOURCE_GROUP" \
    --is-linux \
    --sku B1 \
    --location "$LOCATION"
}

# Step 7: Create the Web App (if it does not exist yet)
if ! az webapp show --resource-group "$RESOURCE_GROUP" --name "$WEBAPP_NAME" &> /dev/null; then
  echo "Creating Web App..."
  az webapp create \
    --resource-group "$RESOURCE_GROUP" \
    --plan "$APP_SERVICE_PLAN" \
    --name "$WEBAPP_NAME" \
    --deployment-container-image-name "$BACKEND_IMAGE"
else
  echo "Web App already exists. Skipping creation."
fi

# Step 8: Configure container settings
echo "Configuring Web App container..."
az webapp config container set \
  --name "$WEBAPP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --docker-custom-image-name "$BACKEND_IMAGE" \
  --docker-registry-server-url "https://$ACR_NAME.azurecr.io" \
  --docker-registry-server-user "$ACR_USERNAME" \
  --docker-registry-server-password "$ACR_PASSWORD"

# Step 9: Set application settings (port)
echo "Setting environment variable WEBSITES_PORT=8000..."
az webapp config appsettings set \
  --name "$WEBAPP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --settings WEBSITES_PORT=8000

# Step 10: Restart Web App
echo "Restarting Web App..."
az webapp restart \
  --name "$WEBAPP_NAME" \
  --resource-group "$RESOURCE_GROUP"

# Summary
echo ""
echo "Deployment complete."
echo "Backend available at: https://$WEBAPP_NAME.azurewebsites.net"
