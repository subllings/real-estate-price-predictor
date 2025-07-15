#!/bin/bash

# This script assumes it was already made executable with: chmod +x scripts/run_loop_tuner_agent.sh
# Run with: ./scripts/run_loop_tuner_agent.sh <model_name> [--no-time-limit | --stop-hour <H> --stop-minute <M>]
# Run with: ./scripts/run_loop_tuner_agent.sh catboost --no-time-limit
# Run with: ./scripts/run_loop_tuner_agent.sh xgboost --no-time-limit

clear
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

PYTHON_EXEC="$PROJECT_ROOT/.venv/Scripts/python.exe"

if [ $# -lt 1 ]; then
  echo "[Shell] ❌ Missing model name. Usage: ./scripts/run_loop_tuner_agent.sh <model_name> [--no-time-limit | --stop-hour <H> --stop-minute <M>]"
  exit 1
fi

MODEL_NAME=$1
shift

echo "[Shell] Launching tuner loop for model: $MODEL_NAME"
"$PYTHON_EXEC" -m agents.tuner_agent.loop_tuner_agent "$MODEL_NAME" "$@"

echo "[Shell] ✅ Tuning loop ended for model: $MODEL_NAME"