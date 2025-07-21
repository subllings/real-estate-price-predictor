#!/bin/bash

# Make executable: chmod +x setup-conda-env.sh
# Run with: ./setup-conda-env.sh

clear

# === Define color codes ===
BLUE_BG="\033[44m"
GREEN_BG="\033[42m"
RED_BG="\033[41m"
WHITE_TEXT="\033[97m"
RESET="\033[0m"

print_blue() {
    echo -e "\n${BLUE_BG}${WHITE_TEXT}>>> $1${RESET}\n"
}

print_green() {
    echo -e "\n${GREEN_BG}${WHITE_TEXT}>>> $1${RESET}\n"
}

print_error() {
    echo -e "\n${RED_BG}${WHITE_TEXT}>>> ERROR: $1${RESET}\n"
    exit 1
}

# === Remove existing conda env ===
ENV_NAME="realestate-env"

print_blue "Removing existing conda environment ($ENV_NAME) if it exists..."
conda env remove -n "$ENV_NAME" -y >/dev/null 2>&1

# === Create new conda environment ===
print_blue "Creating new conda environment ($ENV_NAME) with Python 3.12..."
conda create -y -n "$ENV_NAME" python=3.12 || print_error "Failed to create conda environment."

# === Activate environment ===
print_blue "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME" || print_error "Failed to activate conda environment."

# === Install requirements ===
print_blue "Installing dependencies from requirements.txt..."
pip install -r requirements.txt || print_error "Failed to install dependencies."

# === Optional: register kernel for Jupyter ===
print_blue "Registering kernel for Jupyter..."
python -m ipykernel install --user --name="$ENV_NAME" --display-name "Python ($ENV_NAME)"

# === VS Code config ===
print_blue "Writing VS Code settings..."
mkdir -p .vscode
echo '{
  "python.defaultInterpreterPath": "${workspaceFolder}/.conda/envs/'"$ENV_NAME"'/bin/python"
}' > .vscode/settings.json

# === Done ===
print_green "Setup complete. Your Conda environment '$ENV_NAME' is ready!"
