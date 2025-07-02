#!/bin/bash

# Make this file executable: chmod +x run_api.sh
# Run it with: ./run_api.sh

clear

# Définir l'hôte par défaut si non défini
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}

echo "Starting API at http://${HOST}:${PORT} ..."
uvicorn app.backend.main:app --host $HOST --port $PORT --reload