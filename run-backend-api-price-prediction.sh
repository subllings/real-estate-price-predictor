#!/bin/bash

clear

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8010}
BACKEND_DIR="app/backend-api-llm-v2"
ENTRYPOINT="main:app"

# === Setup conda for Git Bash Windows ===
CONDA_BASE="/c/ProgramData/anaconda3"

# Add conda to PATH if conda command is not found
if ! command -v conda &> /dev/null; then
    export PATH="$CONDA_BASE:$CONDA_BASE/Scripts:$CONDA_BASE/condabin:$PATH"
fi

# Source conda.sh to enable 'conda activate'
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    echo "❌ Could not find conda.sh to initialize conda shell integration."
    exit 1
fi

# === Activate Conda env local ===
ENV_PATH="./environment/conda-env"
if [ -d "$ENV_PATH" ]; then
    conda activate "$ENV_PATH"
else
    echo "❌ Conda environment not found at $ENV_PATH. Run setup-conda-env.sh first."
    exit 1
fi

# === Change to backend directory ===
cd "$BACKEND_DIR" || {
    echo "❌ Backend directory not found: $BACKEND_DIR"
    exit 1
}

# === Start FastAPI backend ===
echo -e "\033[34m>>> Starting API at http://${HOST}:${PORT} ...\033[0m"
uvicorn "$ENTRYPOINT" --host "$HOST" --port "$PORT" --reload &

# === Open browser on docs ===
sleep 2
echo -e "\033[34m>>> Opening browser at http://${HOST}:${PORT}/docs ...\033[0m"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    explorer.exe "http://${HOST}:${PORT}/docs"
else
    xdg-open "http://${HOST}:${PORT}/docs" >/dev/null 2>&1 &
fi
