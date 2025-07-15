#!/bin/bash

# Quick update script - only updates dependencies
# Usage: ./update-deps.sh

# === Define color codes ===
GREEN_BG="\033[42m"
BLUE_BG="\033[44m"
RED_BG="\033[41m"
WHITE_TEXT="\033[97m"
RESET="\033[0m"

print_green() {
    echo -e "${GREEN_BG}${WHITE_TEXT}>>> $1${RESET}"
}

print_blue() {
    echo -e "${BLUE_BG}${WHITE_TEXT}>>> $1${RESET}"
}

print_error() {
    echo -e "${RED_BG}${WHITE_TEXT}>>> ERROR: $1${RESET}"
    exit 1
}

# === Check if venv exists ===
if [ ! -d ".venv" ]; then
    print_error "No virtual environment found. Run ./setup-smart.sh first."
fi

# === Activate venv ===
print_blue "Activating virtual environment..."
source .venv/Scripts/activate || print_error "Failed to activate venv."

# === Quick dependency update ===
print_blue "Updating dependencies (only new/changed packages)..."
pip install -r requirements.txt --upgrade

# === Update timestamp ===
echo "$(date +%s)" > .venv/.install_timestamp

print_green "✅ Dependencies updated successfully!"
