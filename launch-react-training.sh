#!/bin/bash

# Script de lancement de l'interface React - Model Training
# Avec données réelles depuis CosmosDB

echo "🚀 Lancement de l'interface React - Model Training"
echo "================================================"

# Vérifier si l'API est accessible
echo "🔍 Vérification de l'API..."
if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    echo "✅ API accessible sur le port 8000"
else
    echo "❌ API non accessible sur le port 8000"
    echo "   Assurez-vous que l'API est démarrée avec:"
    echo "   python app/backend-api-price-prediction/main.py"
    echo "   ou"
    echo "   ./run-backend-api-price-prediction.sh"
    exit 1
fi

# Vérifier les données
echo "🧪 Vérification des données d'expériences..."
EXPERIMENTS=$(curl -s http://127.0.0.1:8000/experiments | jq -r '.experiments | length' 2>/dev/null)
if [ "$EXPERIMENTS" -gt 0 ]; then
    echo "✅ $EXPERIMENTS expériences trouvées"
else
    echo "⚠️  Aucune expérience trouvée - l'interface sera vide"
fi

# Naviguer vers le dossier React
echo "📁 Navigation vers le dossier React..."
cd app/frontend-react

# Vérifier si node_modules existe
if [ ! -d "node_modules" ]; then
    echo "📦 Installation des dépendances..."
    npm install
fi

# Démarrer l'application React
echo "⚛️  Démarrage de l'application React..."
echo "   Interface disponible sur: http://localhost:3000"
echo "   Page Model Training: http://localhost:3000/training"
echo ""
echo "🎯 L'interface affichera les données réelles depuis CosmosDB"
echo "   - Tableau des expériences avec R² scores"
echo "   - Statistiques de résumé"
echo "   - État en temps réel des modèles"
echo ""
echo "📱 Naviguez vers 'Model Training' dans le menu pour voir l'interface"
echo ""

# Lancer React
npm start
