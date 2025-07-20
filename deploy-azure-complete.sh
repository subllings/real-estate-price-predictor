#!/bin/bash
# deploy-azure-complete.sh - Déploiement automatisé de l'application Real Estate ML sur Azure

set -e

# Variables de configuration
RESOURCE_GROUP="rg-real-estate-ml"
LOCATION="westeurope"
ACR_NAME="acrrealestate"
COSMOS_NAME="cosmos-real-estate-ml"

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Vérifier si Azure CLI est installé et connecté
check_prerequisites() {
    print_status "Vérification des prérequis..."
    
    if ! command -v az &> /dev/null; then
        print_error "Azure CLI n'est pas installé. Veuillez l'installer : https://docs.microsoft.com/en-us/cli/azure/install-azure-cli"
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker n'est pas installé. Veuillez l'installer : https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Vérifier la connexion Azure
    if ! az account show &> /dev/null; then
        print_error "Vous n'êtes pas connecté à Azure. Exécutez 'az login' d'abord."
        exit 1
    fi
    
    print_success "Prérequis validés"
}

# Créer les ressources Azure
create_azure_resources() {
    print_status "Création des ressources Azure..."
    
    # Créer le groupe de ressources
    print_status "Création du groupe de ressources $RESOURCE_GROUP..."
    az group create --name $RESOURCE_GROUP --location $LOCATION --output table
    
    # Créer Azure Container Registry
    print_status "Création d'Azure Container Registry $ACR_NAME..."
    az acr create --resource-group $RESOURCE_GROUP \
                  --name $ACR_NAME \
                  --sku Basic \
                  --admin-enabled true \
                  --output table
    
    # Créer Azure Cosmos DB
    print_status "Création d'Azure Cosmos DB $COSMOS_NAME..."
    az cosmosdb create --name $COSMOS_NAME \
                       --resource-group $RESOURCE_GROUP \
                       --kind GlobalDocumentDB \
                       --locations regionName=$LOCATION \
                       --output table
    
    print_success "Ressources Azure créées avec succès"
}

# Build et push des images Docker
build_and_push_images() {
    print_status "Build et push des images Docker..."
    
    # Se connecter au registry
    print_status "Connexion à Azure Container Registry..."
    az acr login --name $ACR_NAME
    
    # Vérifier que les Dockerfiles existent
    if [ ! -f "app/backend-api-price-prediction/Dockerfile" ]; then
        print_warning "Dockerfile backend non trouvé. Création..."
        create_backend_dockerfile
    fi
    
    if [ ! -f "app/frontend-react/Dockerfile" ]; then
        print_warning "Dockerfile frontend non trouvé. Création..."
        create_frontend_dockerfile
    fi
    
    # Build et push backend
    print_status "Build de l'image backend..."
    cd app/backend-api-price-prediction
    docker build -t $ACR_NAME.azurecr.io/backend-api:latest .
    
    print_status "Push de l'image backend..."
    docker push $ACR_NAME.azurecr.io/backend-api:latest
    
    # Build et push frontend
    print_status "Build de l'image frontend..."
    cd ../frontend-react
    docker build -t $ACR_NAME.azurecr.io/frontend-react:latest .
    
    print_status "Push de l'image frontend..."
    docker push $ACR_NAME.azurecr.io/frontend-react:latest
    
    cd ../..
    
    print_success "Images Docker créées et poussées avec succès"
}

# Créer le Dockerfile pour le backend
create_backend_dockerfile() {
    cat > app/backend-api-price-prediction/Dockerfile << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Créer le répertoire scripts et copier les scripts tuner
RUN mkdir -p /app/scripts
COPY ../../run_loop_tuner_agent.sh /app/scripts/ 2>/dev/null || echo "Script tuner non trouvé, sera ignoré"
COPY ../../loop_tuner_agent.py /app/scripts/ 2>/dev/null || echo "Script tuner non trouvé, sera ignoré"
COPY ../../agents/ /app/agents/ 2>/dev/null || echo "Dossier agents non trouvé, sera ignoré"

# Rendre les scripts exécutables
RUN chmod +x /app/scripts/*.sh 2>/dev/null || true

# Variables d'environnement
ENV PYTHONPATH=/app
ENV COSMOS_DB_ENDPOINT=""
ENV COSMOS_DB_KEY=""

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
EOF
}

# Créer le Dockerfile pour le frontend
create_frontend_dockerfile() {
    cat > app/frontend-react/Dockerfile << 'EOF'
FROM node:18-alpine as build

WORKDIR /app

# Copier package.json et package-lock.json
COPY package*.json ./

# Installer les dépendances
RUN npm ci --only=production

# Copier le code source
COPY . .

# Build de l'application
RUN npm run build

# Servir avec nginx
FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html

# Configuration nginx
RUN echo 'server { \
    listen 80; \
    location / { \
        root /usr/share/nginx/html; \
        index index.html index.htm; \
        try_files $uri $uri/ /index.html; \
    } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
EOF
}

# Déployer les containers
deploy_containers() {
    print_status "Déploiement des containers Azure..."
    
    # Obtenir les credentials
    print_status "Récupération des credentials Azure..."
    COSMOS_ENDPOINT=$(az cosmosdb show --name $COSMOS_NAME --resource-group $RESOURCE_GROUP --query documentEndpoint --output tsv)
    COSMOS_KEY=$(az cosmosdb keys list --name $COSMOS_NAME --resource-group $RESOURCE_GROUP --query primaryMasterKey --output tsv)
    ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv)
    
    # Déployer le backend
    print_status "Déploiement du container backend..."
    az container create \
      --resource-group $RESOURCE_GROUP \
      --name backend-api \
      --image $ACR_NAME.azurecr.io/backend-api:latest \
      --cpu 2 \
      --memory 4 \
      --registry-login-server $ACR_NAME.azurecr.io \
      --registry-username $ACR_NAME \
      --registry-password $ACR_PASSWORD \
      --ip-address Public \
      --ports 8002 \
      --environment-variables \
        COSMOS_DB_ENDPOINT=$COSMOS_ENDPOINT \
        COSMOS_DB_KEY=$COSMOS_KEY \
      --output table
    
    # Attendre que le backend soit prêt
    print_status "Attente du démarrage du backend..."
    sleep 30
    
    # Récupérer l'IP du backend
    BACKEND_IP=$(az container show --resource-group $RESOURCE_GROUP --name backend-api --query ipAddress.ip --output tsv)
    print_success "Backend déployé sur IP: $BACKEND_IP"
    
    # Déployer le frontend
    print_status "Déploiement du container frontend..."
    az container create \
      --resource-group $RESOURCE_GROUP \
      --name frontend-react \
      --image $ACR_NAME.azurecr.io/frontend-react:latest \
      --cpu 1 \
      --memory 2 \
      --registry-login-server $ACR_NAME.azurecr.io \
      --registry-username $ACR_NAME \
      --registry-password $ACR_PASSWORD \
      --ip-address Public \
      --ports 80 \
      --environment-variables \
        REACT_APP_API_URL=http://$BACKEND_IP:8002 \
      --output table
    
    # Récupérer l'IP du frontend
    FRONTEND_IP=$(az container show --resource-group $RESOURCE_GROUP --name frontend-react --query ipAddress.ip --output tsv)
    print_success "Frontend déployé sur IP: $FRONTEND_IP"
}

# Afficher le résumé du déploiement
show_deployment_summary() {
    print_success "🎉 Déploiement terminé avec succès!"
    echo ""
    echo "📋 Résumé du déploiement:"
    echo "========================"
    
    # Récupérer les IPs
    FRONTEND_IP=$(az container show --resource-group $RESOURCE_GROUP --name frontend-react --query ipAddress.ip --output tsv 2>/dev/null || echo "Non disponible")
    BACKEND_IP=$(az container show --resource-group $RESOURCE_GROUP --name backend-api --query ipAddress.ip --output tsv 2>/dev/null || echo "Non disponible")
    COSMOS_ENDPOINT=$(az cosmosdb show --name $COSMOS_NAME --resource-group $RESOURCE_GROUP --query documentEndpoint --output tsv 2>/dev/null || echo "Non disponible")
    
    echo "🌐 Application Web:     http://$FRONTEND_IP"
    echo "🔧 API Backend:        http://$BACKEND_IP:8002"
    echo "📊 Cosmos DB Endpoint: $COSMOS_ENDPOINT"
    echo "📦 Container Registry: $ACR_NAME.azurecr.io"
    echo "🏷️  Groupe de ressources: $RESOURCE_GROUP"
    echo ""
    echo "🚀 Pour lancer des tuner agents:"
    echo "   1. Accédez à http://$FRONTEND_IP"
    echo "   2. Allez dans 'Model Training'"
    echo "   3. Cliquez 'New Training'"
    echo "   4. Configurez et lancez votre tuner agent!"
    echo ""
    echo "📝 Commandes utiles:"
    echo "   - Logs backend:  az container logs --resource-group $RESOURCE_GROUP --name backend-api"
    echo "   - Logs frontend: az container logs --resource-group $RESOURCE_GROUP --name frontend-react"
    echo "   - Supprimer tout: az group delete --name $RESOURCE_GROUP --yes --no-wait"
}

# Fonction de nettoyage en cas d'erreur
cleanup_on_error() {
    print_error "Une erreur s'est produite pendant le déploiement."
    read -p "Voulez-vous supprimer les ressources créées? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Suppression des ressources..."
        az group delete --name $RESOURCE_GROUP --yes --no-wait
        print_success "Ressources supprimées"
    fi
}

# Fonction principale
main() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║               🚀 Déploiement Azure Real Estate ML           ║"
    echo "║                                                              ║"
    echo "║  Ce script va déployer votre application complète sur Azure ║"
    echo "║  avec support des tuner agents en arrière-plan              ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    
    # Confirmer le déploiement
    read -p "Voulez-vous continuer? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Déploiement annulé"
        exit 0
    fi
    
    # Piège pour gérer les erreurs
    trap cleanup_on_error ERR
    
    # Exécuter les étapes
    check_prerequisites
    create_azure_resources
    build_and_push_images
    deploy_containers
    show_deployment_summary
    
    print_success "Déploiement terminé! 🎉"
}

# Lancer le script
main "$@"
