#!/bin/bash
echo "🚀 Lancement de l'application React..."
echo ""

cd "$(dirname "$0")/app/frontend-react"

echo "📦 Vérification des dépendances..."
if [ ! -d "node_modules" ]; then
    echo "📥 Installation des dépendances..."
    npm install
fi

echo "🌐 Lancement de l'application React..."
echo "📍 URL: http://localhost:3000"
npm start
