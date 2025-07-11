#!/bin/bash

# Make this file executable: chmod +x setup-env.sh
# Run it with: ./setup-env.sh

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
print_blue "Creating new virtual environment with Python 3.12..."
py -3.12 -m venv .venv || print_error "Python 3.12 not found. Please install it first."

# === Activate venv ===
print_blue "Activating virtual environment..."
source .venv/Scripts/activate || print_error "Failed to activate venv. Are you in Git Bash?"

# === Show Python and pip versions ===
print_blue "Python version:"
python --version

print_blue "pip version:"
pip --version

# === Upgrade pip only if needed (version < 23) ===
PIP_VERSION=$(pip --version | awk '{print $2}')
PIP_MAJOR=$(echo "$PIP_VERSION" | cut -d. -f1)
if (( PIP_MAJOR < 23 )); then
    print_blue "Upgrading pip (current: $PIP_VERSION)..."
    python -m pip install --upgrade pip
else
    print_blue "pip is up-to-date ($PIP_VERSION), skipping upgrade."
fi

# === Install dependencies ===
print_blue "Installing dependencies from requirements.txt..."
pip install -r requirements.txt || print_error "Failed to install dependencies."

# === Register venv in Jupyter ===
print_blue "Registering virtual environment in Jupyter..."
python -m ipykernel install --user --name=venv --display-name "Python (.venv)"

# === Configure VS Code interpreter ===
print_blue "Writing VS Code settings..."
mkdir -p .vscode
echo '{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe"
}' > .vscode/settings.json

# === Optional: XGBoost GPU Support (NVIDIA) ===
print_blue "Installing GPU-enabled XGBoost (optional)..."
pip uninstall -y xgboost
pip install --upgrade --extra-index-url https://pypi.nvidia.com xgboost

pip install "uvicorn[standard]==0.29.0"

# === Done ===
print_green "Setup complete. Your virtual environment is ready!"
