#!/bin/bash

# Make this file executable: chmod +x predict_price.sh
# Run it with: ./predict_price.sh
# From root directory: chmod +x scripts/predict_price.sh && ./scripts/predict_price.sh

clear

# === Define color codes ===
BLUE_BG="\033[44m"
GREEN_BG="\033[42m"
RED_BG="\033[41m"
WHITE_TEXT="\033[97m"
BLACK_TEXT="\033[30m"
RESET="\033[0m"

# === Define print helpers ===
print_blue() {
    echo ""
    echo -e "${BLUE_BG}${WHITE_TEXT}>>> $1${RESET}"
    echo ""
}

print_green() {
    echo ""
    echo -e "${GREEN_BG}${WHITE_TEXT}>>> $1${RESET}"
    echo ""
}

print_error() {
    echo ""
    echo -e "${RED_BG}${WHITE_TEXT}>>> ERROR: $1${RESET}"
    echo ""
    exit 1
}

# === Activate virtual environment ===
print_blue "Activating virtual environment..."
source .venv/Scripts/activate || print_error "Failed to activate virtual environment. Is it created?"

# === Run the prediction script ===
print_blue "Predicting price from sample input..."
PYTHONPATH=. python scripts/predict_price.py || print_error "Prediction failed."

# === Done ===
print_green "Prediction completed successfully."
