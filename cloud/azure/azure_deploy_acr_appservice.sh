#!/bin/bash

# Make this file executable: chmod +x cloud/azure/azure_deploy_acr_appservice.sh
# Run it from the project root with:
# ./cloud/azure/azure_deploy_acr_appservice.sh

# === Optional: One-time registration of Microsoft.Web provider ===
# Launch manually before first deployment (only once per subscription):
# az provider register --namespace Microsoft.Web
az provider show --namespace Microsoft.Web --query "registrationState"

# === Resolve script path (in case it's run from project root) ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

set -e

# === Configuration ===
RESOURCE_GROUP="realestate-api-rg-north"
LOCATION="northeurope"
ACR_NAME="realestateacrneo2"
APP_SERVICE_PLAN="realestate-api-plan-north"
WEBAPP_NAME="realestate-api"

BACKEND_IMAGE="$ACR_NAME.azurecr.io/real-estate-backend:latest"
FRONTEND_IMAGE="$ACR_NAME.azurecr.io/real-estate-frontend:latest"
DOCKER_COMPOSE_FILE="cloud/azure/docker-compose-azure.yml"

echo "RG=$RESOURCE_GROUP / PLAN=$APP_SERVICE_PLAN / WEBAPP=$WEBAPP_NAME / ACR=$ACR_NAME"

# === Step 1: Create the resource group ===
echo "Creating Azure resource group..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION"

# === Step 2: Create Azure Container Registry (ACR) ===
echo "Creating Azure Container Registry..."
az acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic --location "$LOCATION"

# === Step 3: Login to ACR ===
echo "Logging in to ACR..."
az acr login --name "$ACR_NAME"

# === Step 4: Build Docker images ===
echo "Building Docker images..."
docker build -f app/backend/Dockerfile.azure -t "$BACKEND_IMAGE" app/backend
docker build -f app/frontend-streamlit/Dockerfile.azure -t "$FRONTEND_IMAGE" app/frontend-streamlit

# === Step 5: Push images to ACR ===
echo "Pushing Docker images to ACR..."
docker push "$BACKEND_IMAGE"
docker push "$FRONTEND_IMAGE"

# === Step 6: Create Linux App Service plan ===
echo "Creating Linux App Service plan..."
az appservice plan create --name "$APP_SERVICE_PLAN" --resource-group "$RESOURCE_GROUP" --sku B1 --is-linux

# === Step 7: Deploy multi-container app using docker-compose.yml ===
echo "Deploying Web App using docker-compose.yml..."
az webapp create \
  --resource-group "$RESOURCE_GROUP" \
  --plan "$APP_SERVICE_PLAN" \
  --name "$WEBAPP_NAME" \
  --multicontainer-config-type compose \
  --multicontainer-config-file "$DOCKER_COMPOSE_FILE"

# === Step 8: Summary ===
echo ""
echo "Deployment complete!"
echo "Backend:  https://$WEBAPP_NAME.azurewebsites.net:8000"
echo "Frontend: https://$WEBAPP_NAME.azurewebsites.net:8501"
