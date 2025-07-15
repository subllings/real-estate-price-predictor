#!/bin/bash

# Quick activation script
# Usage: source activate-env.sh

# === Define color codes ===
GREEN_BG="\033[42m"
RED_BG="\033[41m"
WHITE_TEXT="\033[97m"
RESET="\033[0m"

print_green() {
    echo -e "${GREEN_BG}${WHITE_TEXT}>>> $1${RESET}"
}

print_error() {
    echo -e "${RED_BG}${WHITE_TEXT}>>> ERROR: $1${RESET}"
    return 1
}

# === Check if venv exists ===
if [ ! -d ".venv" ]; then
    print_error "No virtual environment found. Run ./setup-smart.sh first."
    return 1
fi

# === Activate venv ===
source .venv/Scripts/activate || {
    print_error "Failed to activate venv."
    return 1
}

print_green "✅ Virtual environment activated! Python version: $(python --version)"

# Show current directory and remind about available commands
echo ""
echo "📁 Current directory: $(pwd)"
echo "🐍 Python path: $(which python)"
echo ""
echo "Available quick commands:"
echo "  • ./run_all.sh                    # Start all services"
echo "  • ./update-deps.sh               # Update dependencies only"
echo "  • ./setup-smart.sh --force-reinstall  # Full reinstall if needed"
echo ""
