#!/bin/bash

# Make this file executable: chmod +x setup-env.sh
# Run it with: ./setup-env.sh
# chmod +x setup-env.sh && ./setup-env.sh

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

# === Show Python versions ===
print_blue "Available Python versions:"
py -0 || where python

# === Remove existing venv ===
if [ -d ".venv" ]; then
    print_blue "Removing existing virtual environment (.venv)..."
    rm -rf .venv
fi

# === Create new venv ===
print_blue "Creating new virtual environment with Python 3.12.10..."
py -3.12 -m venv .venv || print_error "Python 3.12 not found. Please install it first."

# === Activate venv ===
echo ">>> Activating virtual environment..."
source .venv/Scripts/activate || print_error "Failed to activate venv. Are you in Git Bash?"

# === Show versions ===
print_blue "Python version:"
python --version

print_blue "pip version:"pyt
pip --version

# === Upgrade pip ===
print_blue "Upgrading pip..."
python -m pip install --upgrade pip

# === Install dependencies ===
print_blue "Installing dependencies from requirements.txt..."
pip install -r requirements.txt
#pip install -r requirements-dev.txt
npm install -g pyright
pip install --upgrade flake8 flake8-docstrings pydocstyle

# === Register venv in Jupyter ===
python -m ipykernel install --user --name=venv --display-name "Python (.venv)"
print_blue "Writing .vscode/settings.json to use .venv Python interpreter..."
mkdir -p .vscode
echo '{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe"
}' > .vscode/settings.json

# === Done ===
print_green "Setup complete. Your virtual environment is ready!"
