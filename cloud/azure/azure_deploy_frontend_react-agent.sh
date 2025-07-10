#!/bin/bash

# Make this script executable: chmod +x cloud/azure/azure_deploy_frontend_react-agent.sh
# Run with: ./cloud/azure/azure_deploy_frontend_react-agent.sh

clear

set -e

# === Azure CLI detection ===
AZ_CLI="/c/Program Files/Microsoft SDKs/Azure/CLI2/wbin/az.cmd"

if [ ! -f "$AZ_CLI" ]; then
  echo "❌ Azure CLI not found at: $AZ_CLI"
  exit 1
fi

# === Directories ===
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# === Configuration ===
RESOURCE_GROUP="realestate-api-rg-north"
LOCATION="northeurope"
ACR_NAME="realestateacrneo2"
APP_SERVICE_PLAN="realestate-api-plan-north"
WEBAPP_NAME="realestate-react-ui-agent"

FRONTEND_IMAGE="$ACR_NAME.azurecr.io/real-estate-react-frontend:agent"
DOCKERFILE_PATH="app/frontend-react/Dockerfile.azure"
FRONTEND_DIR="app/frontend-react"

echo "Creating ACR if needed..."
"$AZ_CLI" acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null || {
  "$AZ_CLI" acr create --resource-group "$RESOURCE_GROUP" --name "$ACR_NAME" --sku Basic --location "$LOCATION"
}

echo "Logging into ACR..."
"$AZ_CLI" acr login --name "$ACR_NAME"

echo "Building Docker image (agent)..."
docker build -f "$DOCKERFILE_PATH" -t "$FRONTEND_IMAGE" "$FRONTEND_DIR"

echo "Pushing image to ACR..."
docker push "$FRONTEND_IMAGE"

echo "Retrieving ACR credentials..."
"$AZ_CLI" acr update -n "$ACR_NAME" --admin-enabled true
ACR_USERNAME=$("$AZ_CLI" acr credential show -n "$ACR_NAME" --query "username" -o tsv)
ACR_PASSWORD=$("$AZ_CLI" acr credential show -n "$ACR_NAME" --query "passwords[0].value" -o tsv)

echo "Creating Web App for React frontend (agent)..."
"$AZ_CLI" webapp create \
  --resource-group "$RESOURCE_GROUP" \
  --plan "$APP_SERVICE_PLAN" \
  --name "$WEBAPP_NAME" \
  --deployment-container-image-name "$FRONTEND_IMAGE"

echo "Configuring container credentials..."
"$AZ_CLI" webapp config container set \
  --name "$WEBAPP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --docker-custom-image-name "$FRONTEND_IMAGE" \
  --docker-registry-server-url "https://$ACR_NAME.azurecr.io" \
  --docker-registry-server-user "$ACR_USERNAME" \
  --docker-registry-server-password "$ACR_PASSWORD"

echo "Setting required environment variables (API_URL)..."
"$AZ_CLI" webapp config appsettings set \
  --name "$WEBAPP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --settings API_URL=https://realestate-api.azurewebsites.net

echo "Restarting React frontend Web App (agent)..."
"$AZ_CLI" webapp restart \
  --name "$WEBAPP_NAME" \
  --resource-group "$RESOURCE_GROUP"

echo ""
echo "✅ React frontend (agent) deployed:"
echo "🔗 https://$WEBAPP_NAME.azurewebsites.net"

# Optional: enable logging (useful during debugging)
"$AZ_CLI" webapp log config \
  --name "$WEBAPP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --docker-container-logging filesystem
