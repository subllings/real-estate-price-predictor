#!/bin/bash

# Make this file executable: chmod +x run-backend-api.sh
# Run it with: ./run-backend-api.sh

clear

# Définir l'hôte par défaut si non défini
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}

echo "Starting API at http://${HOST}:${PORT} ..."
uvicorn app.backend.main:app --host $HOST --port $PORT --reload

sleep 2

print_blue "Opening browser..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    explorer.exe "http://localhost:8000/docs"
else
    xdg-open http://localhost:8000/docs >/dev/null 2>&1 &
    
fi