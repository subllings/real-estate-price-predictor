#!/bin/bash

echo "🔍 Test de compilation React..."
echo "=============================="

cd app/frontend-react

echo "📦 Vérification des dépendances..."
if [ ! -d "node_modules" ]; then
    echo "Installation des dépendances..."
    npm install
fi

echo "🧪 Test de compilation..."
npm run build

if [ $? -eq 0 ]; then
    echo "✅ Compilation réussie ! L'erreur a été corrigée."
else
    echo "❌ Erreur de compilation persistante."
    exit 1
fi
