#!/bin/bash

# Make this file executable: chmod +x run-backend-api-price-prediction.sh
# Run it with: ./run-backend-api-price-prediction.sh

clear

# === Config ===
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
BACKEND_DIR="app/backend-api-price-prediction"
ENTRYPOINT="main:app"

# === Activate venv (Windows Git Bash)
if [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
else
    echo "❌ .venv not found or not accessible. Run setup-env.sh first."
    exit 1
fi

# === Go to backend directory ===
cd "$BACKEND_DIR" || {
    echo "❌ Cannot find directory: $BACKEND_DIR"
    exit 1
}

# === Start FastAPI backend ===
echo -e "\033[34m>>> Starting API at http://${HOST}:${PORT} ...\033[0m"
uvicorn "$ENTRYPOINT" --host "$HOST" --port "$PORT" --reload &

# === Wait and open browser ===
sleep 2
echo -e "\033[34m>>> Opening browser at http://${HOST}:${PORT}/docs ...\033[0m"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    explorer.exe "http://${HOST}:${PORT}/docs"
else
    xdg-open "http://${HOST}:${PORT}/docs" >/dev/null 2>&1 &
fi
