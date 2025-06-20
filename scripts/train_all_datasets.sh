#!/bin/bash

# Make this file executable: chmod +x train_all_datasets.sh
# Run it with: ./train_all_datasets.sh
# from root directory: chmod +x scripts/train_all_datasets.sh && ./scripts/train_all_datasets.sh


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

# === Run the training script ===
print_blue "Launching dataset-wide training for all models..."
PYTHONPATH=. python scripts/train_all_datasets.py || print_error "Training failed."

# === Done ===
print_green "All models trained successfully for all datasets."
