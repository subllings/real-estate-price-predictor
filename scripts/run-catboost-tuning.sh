#!/bin/bash


# This script assumes it was already made executable with: chmod +x scripts/run-xgboost-tuning.sh
# Run with: ./scripts/run-xgboost-tuning.sh

set -e

# Détection du chemin du projet
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Active le bon interpréteur Python
PYTHON_EXEC="$PROJECT_ROOT/.venv/Scripts/python.exe"

echo "[Shell] Starting CatBoost Optuna tuning..."

"$PYTHON_EXEC" model_training/train_xgboost_optuna_global.py

echo "CatBoost tuning completed."
