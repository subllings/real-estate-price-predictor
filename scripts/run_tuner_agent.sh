#!/bin/bash

# This script assumes it was already made executable with: chmod +x scripts/run_tuner_agent.sh
# Run with: ./scripts/run_tuner_agent.sh <model_name>
# Example: ./scripts/run_tuner_agent.sh catboost

clear

set -e

# Detect project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# Python interpreter (adjust if needed)
PYTHON_EXEC="$PROJECT_ROOT/.venv/Scripts/python.exe"

# Check if model name is passed
if [ $# -ne 1 ]; then
  echo "[Shell] ❌ Missing model name. Usage: ./scripts/run_tuner_agent.sh <model_name>"
  exit 1
fi

MODEL_NAME=$1

echo "[Shell] Launching TunerAgentOrchestrator for model: $MODEL_NAME"

"$PYTHON_EXEC" scripts/run_tuner_agent.py "$MODEL_NAME"

echo "[Shell] ✅ Tuning completed for model: $MODEL_NAME"
