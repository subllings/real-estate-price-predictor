# Guide de Déploiement Azure - Tuner Agent avec React Containerisé

## Vue d'ensemble de l'Architecture

Cette solution déploie sur Azure :
- **Frontend React** : Application web containerisée
- **Backend FastAPI** : API de gestion des modèles et tuner agents
- **Tuner Agents** : Processus d'optimisation des hyperparamètres en arrière-plan
- **Azure Container Instances** : Hébergement des containers
- **Azure Cosmos DB** : Suivi des jobs et métriques

## 1. Prérequis Azure

### Services Azure requis :
```bash
# Connexion à Azure
az login

# Créer un groupe de ressources
az group create --name rg-real-estate-ml --location westeurope

# Créer Azure Container Registry
az acr create --resource-group rg-real-estate-ml \
              --name acrrealestate \
              --sku Basic --admin-enabled true

# Créer Azure Cosmos DB
az cosmosdb create --name cosmos-real-estate-ml \
                   --resource-group rg-real-estate-ml \
                   --kind GlobalDocumentDB \
                   --locations regionName=westeurope
```

## 2. Configuration des Containers

### A. Dockerfile pour le Backend FastAPI

```dockerfile
# backend-api-price-prediction/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copier les requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Copier les scripts tuner depuis la racine
COPY ../run_loop_tuner_agent.sh /app/scripts/
COPY ../loop_tuner_agent.py /app/scripts/
COPY ../agents/ /app/agents/

# Rendre les scripts exécutables
RUN chmod +x /app/scripts/run_loop_tuner_agent.sh

# Variables d'environnement
ENV PYTHONPATH=/app
ENV COSMOS_DB_ENDPOINT=""
ENV COSMOS_DB_KEY=""

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
```

### B. Dockerfile pour le Frontend React

```dockerfile
# app/frontend-react/Dockerfile
FROM node:18-alpine

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
COPY --from=0 /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/nginx.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

## 3. Build et Push des Images

```bash
# Se connecter au registry
az acr login --name acrrealestate

# Build et push backend
cd app/backend-api-price-prediction
docker build -t acrrealestate.azurecr.io/backend-api:latest .
docker push acrrealestate.azurecr.io/backend-api:latest

# Build et push frontend
cd ../frontend-react
docker build -t acrrealestate.azurecr.io/frontend-react:latest .
docker push acrrealestate.azurecr.io/frontend-react:latest
```

## 4. Déploiement avec Azure Container Instances

### A. Déployer le Backend

```bash
# Obtenir les credentials Cosmos DB
COSMOS_ENDPOINT=$(az cosmosdb show --name cosmos-real-estate-ml --resource-group rg-real-estate-ml --query documentEndpoint --output tsv)
COSMOS_KEY=$(az cosmosdb keys list --name cosmos-real-estate-ml --resource-group rg-real-estate-ml --query primaryMasterKey --output tsv)

# Déployer le container backend
az container create \
  --resource-group rg-real-estate-ml \
  --name backend-api \
  --image acrrealestate.azurecr.io/backend-api:latest \
  --cpu 2 \
  --memory 4 \
  --registry-login-server acrrealestate.azurecr.io \
  --registry-username acrrealestate \
  --registry-password $(az acr credential show --name acrrealestate --query passwords[0].value --output tsv) \
  --ip-address Public \
  --ports 8002 \
  --environment-variables \
    COSMOS_DB_ENDPOINT=$COSMOS_ENDPOINT \
    COSMOS_DB_KEY=$COSMOS_KEY
```

### B. Déployer le Frontend

```bash
# Récupérer l'IP du backend
BACKEND_IP=$(az container show --resource-group rg-real-estate-ml --name backend-api --query ipAddress.ip --output tsv)

# Déployer le container frontend
az container create \
  --resource-group rg-real-estate-ml \
  --name frontend-react \
  --image acrrealestate.azurecr.io/frontend-react:latest \
  --cpu 1 \
  --memory 2 \
  --registry-login-server acrrealestate.azurecr.io \
  --registry-username acrrealestate \
  --registry-password $(az acr credential show --name acrrealestate --query passwords[0].value --output tsv) \
  --ip-address Public \
  --ports 80 \
  --environment-variables \
    REACT_APP_API_URL=http://$BACKEND_IP:8002
```

## 5. Configuration React pour les Tuner Agents

### Modification du service API React

```javascript
// app/frontend-react/src/services/api.js

// Ajouter les endpoints tuner
export const tunerApi = {
  // Lancer un tuner agent
  startTuner: async (config) => {
    const response = await fetch(`${API_BASE_URL}/api/tuner/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config)
    });
    return response.json();
  },

  // Récupérer le statut d'un job
  getTunerStatus: async (jobId) => {
    const response = await fetch(`${API_BASE_URL}/api/tuner/status/${jobId}`);
    return response.json();
  },

  // Lister tous les jobs tuner
  listTunerJobs: async () => {
    const response = await fetch(`${API_BASE_URL}/api/tuner/list`);
    return response.json();
  }
};
```

### Intégration dans ModelTrainingPage.jsx

```javascript
// Dans ModelTrainingPage.jsx, modifier handleNewTrainingSubmit :

const handleNewTrainingSubmit = async (e) => {
  e.preventDefault();
  
  try {
    setIsSubmitting(true);
    
    // Préparer la configuration pour le tuner agent
    const tunerConfig = {
      model_type: newTrainingFormData.modelType.toLowerCase().replace('+optuna', ''),
      termination_type: newTrainingFormData.terminationType
    };
    
    // Ajouter les paramètres selon le type de terminaison
    switch (newTrainingFormData.terminationType) {
      case 'duration':
        tunerConfig.duration_hours = parseFloat(newTrainingFormData.durationHours);
        break;
      case 'end_time':
        tunerConfig.end_time = newTrainingFormData.endTime;
        if (newTrainingFormData.stopHour) tunerConfig.stop_hour = parseInt(newTrainingFormData.stopHour);
        if (newTrainingFormData.stopMinute) tunerConfig.stop_minute = parseInt(newTrainingFormData.stopMinute);
        break;
      case 'max_trials':
        tunerConfig.max_trials = parseInt(newTrainingFormData.maxTrials);
        break;
    }
    
    // Lancer le tuner agent via l'API
    const result = await tunerApi.startTuner(tunerConfig);
    
    if (result.status === 'success') {
      setShowNewTrainingModal(false);
      setAlerts([{
        type: 'success',
        message: `Tuner agent lancé avec succès! Job ID: ${result.job_id}`
      }]);
      
      // Optionnel : ajouter le job à la liste des training jobs
      const newJob = {
        id: result.job_id,
        name: `Tuner ${tunerConfig.model_type}`,
        status: 'running',
        created_at: new Date().toISOString(),
        type: 'tuner_agent'
      };
      
      setTrainingJobs(prev => [newJob, ...prev]);
    }
    
  } catch (error) {
    setAlerts([{
      type: 'error', 
      message: `Erreur lors du lancement: ${error.message}`
    }]);
  } finally {
    setIsSubmitting(false);
  }
};
```

## 6. Script de Déploiement Automatisé

```bash
#!/bin/bash
# deploy-azure-complete.sh

set -e

# Variables
RESOURCE_GROUP="rg-real-estate-ml"
LOCATION="westeurope"
ACR_NAME="acrrealestate"
COSMOS_NAME="cosmos-real-estate-ml"

echo "🚀 Déploiement de l'application Real Estate ML sur Azure"

# 1. Créer les ressources Azure
echo "📋 Création des ressources Azure..."
az group create --name $RESOURCE_GROUP --location $LOCATION

az acr create --resource-group $RESOURCE_GROUP \
              --name $ACR_NAME \
              --sku Basic --admin-enabled true

az cosmosdb create --name $COSMOS_NAME \
                   --resource-group $RESOURCE_GROUP \
                   --kind GlobalDocumentDB \
                   --locations regionName=$LOCATION

# 2. Build et push des images
echo "🔨 Build et push des containers..."
az acr login --name $ACR_NAME

# Backend
cd app/backend-api-price-prediction
docker build -t $ACR_NAME.azurecr.io/backend-api:latest .
docker push $ACR_NAME.azurecr.io/backend-api:latest

# Frontend  
cd ../frontend-react
docker build -t $ACR_NAME.azurecr.io/frontend-react:latest .
docker push $ACR_NAME.azurecr.io/frontend-react:latest

cd ../..

# 3. Déployer les containers
echo "🚢 Déploiement des containers..."

# Obtenir les credentials
COSMOS_ENDPOINT=$(az cosmosdb show --name $COSMOS_NAME --resource-group $RESOURCE_GROUP --query documentEndpoint --output tsv)
COSMOS_KEY=$(az cosmosdb keys list --name $COSMOS_NAME --resource-group $RESOURCE_GROUP --query primaryMasterKey --output tsv)
ACR_PASSWORD=$(az acr credential show --name $ACR_NAME --query passwords[0].value --output tsv)

# Backend
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
    COSMOS_DB_KEY=$COSMOS_KEY

# Attendre que le backend soit prêt
echo "⏳ Attente du démarrage du backend..."
sleep 30

# Récupérer l'IP du backend
BACKEND_IP=$(az container show --resource-group $RESOURCE_GROUP --name backend-api --query ipAddress.ip --output tsv)

# Frontend
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
    REACT_APP_API_URL=http://$BACKEND_IP:8002

# Récupérer l'IP du frontend
FRONTEND_IP=$(az container show --resource-group $RESOURCE_GROUP --name frontend-react --query ipAddress.ip --output tsv)

echo "✅ Déploiement terminé!"
echo "🌐 Frontend: http://$FRONTEND_IP"
echo "🔧 Backend API: http://$BACKEND_IP:8002"
echo "📊 Cosmos DB: $COSMOS_ENDPOINT"
```

## 7. Utilisation depuis React

Une fois déployé, vous pouvez lancer des tuner agents directement depuis l'interface React :

1. **Accéder à l'application** : `http://[FRONTEND_IP]`
2. **Aller dans Model Training**
3. **Cliquer sur "New Training"**
4. **Configurer** :
   - Model Type : CatBoost+Optuna, XGBoost+Optuna, etc.
   - Termination Type : Endless, Duration, End Time, Max Trials
   - Paramètres spécifiques selon le type
5. **Lancer** : Le tuner agent s'exécute en arrière-plan sur Azure
6. **Suivre** : Via l'API de statut et Cosmos DB

## 8. Monitoring et Logs

```bash
# Voir les logs du backend
az container logs --resource-group rg-real-estate-ml --name backend-api

# Voir les logs du frontend  
az container logs --resource-group rg-real-estate-ml --name frontend-react

# Surveiller les métriques
az monitor metrics list --resource /subscriptions/[SUBSCRIPTION]/resourceGroups/rg-real-estate-ml/providers/Microsoft.ContainerInstance/containerGroups/backend-api
```

## Points Clés

✅ **Architecture scalable** : Containers séparés pour frontend/backend
✅ **Exécution asynchrone** : Tuner agents en arrière-plan
✅ **Suivi complet** : Jobs trackés dans Cosmos DB
✅ **Interface unifiée** : Lancement depuis React, API REST
✅ **Déploiement automatisé** : Script bash complet

Cette architecture vous permet de lancer et gérer vos tuner agents directement depuis votre interface React déployée sur Azure !
