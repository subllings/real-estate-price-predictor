#!/bin/bash

# Smart setup script - only installs what's needed
# Usage: ./setup-smart.sh [--force-reinstall]

clear

# === Define color codes ===
BLUE_BG="\033[44m"
GREEN_BG="\033[42m"
YELLOW_BG="\033[43m"
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

print_yellow() {
    echo ""
    echo -e "${YELLOW_BG}${BLACK_TEXT}>>> $1${RESET}"
    echo ""
}

print_error() {
    echo ""
    echo -e "${RED_BG}${WHITE_TEXT}>>> ERROR: $1${RESET}"
    echo ""
    exit 1
}

# === Check for force reinstall flag ===
FORCE_REINSTALL=false
if [ "$1" = "--force-reinstall" ]; then
    FORCE_REINSTALL=true
    print_yellow "Force reinstall mode enabled. Will recreate environment."
fi

# === Check if venv exists and is valid ===
if [ -d ".venv" ] && [ "$FORCE_REINSTALL" = false ]; then
    print_blue "Checking existing virtual environment..."
    
    # Try to activate and check Python version
    if source .venv/Scripts/activate 2>/dev/null; then
        PYTHON_VERSION=$(python --version 2>/dev/null)
        if [[ $PYTHON_VERSION == *"3.12"* ]]; then
            print_green "✅ Valid Python 3.12 environment found: $PYTHON_VERSION"
            
            # Check if requirements are up to date
            print_blue "Checking if dependencies need updates..."
            
            # Get modification time of requirements.txt
            REQ_TIME=$(stat -c %Y requirements.txt 2>/dev/null || stat -f %m requirements.txt 2>/dev/null)
            
            # Check if we have a timestamp file
            if [ -f ".venv/.install_timestamp" ]; then
                INSTALL_TIME=$(cat .venv/.install_timestamp)
                
                if [ "$REQ_TIME" -le "$INSTALL_TIME" ]; then
                    print_green "✅ Dependencies are up to date. Nothing to install!"
                    print_yellow "To force reinstall, run: ./setup-smart.sh --force-reinstall"
                    exit 0
                else
                    print_yellow "⚠️ requirements.txt has been modified. Installing updates only..."
                    pip install -r requirements.txt --upgrade
                    echo "$(date +%s)" > .venv/.install_timestamp
                    print_green "✅ Dependencies updated successfully!"
                    exit 0
                fi
            else
                print_yellow "⚠️ No install timestamp found. Installing dependencies..."
                pip install -r requirements.txt
                echo "$(date +%s)" > .venv/.install_timestamp
                print_green "✅ Dependencies installed successfully!"
                exit 0
            fi
        else
            print_yellow "⚠️ Wrong Python version ($PYTHON_VERSION). Recreating environment..."
        fi
    else
        print_yellow "⚠️ Cannot activate existing environment. Recreating..."
    fi
    
    # If we reach here, we need to recreate
    rm -rf .venv
else
    if [ "$FORCE_REINSTALL" = true ]; then
        print_blue "Force reinstall: Removing existing environment..."
        rm -rf .venv
    else
        print_blue "No virtual environment found. Creating new one..."
    fi
fi

# === Create new venv ===
print_blue "Creating new virtual environment with Python 3.12..."
py -3.12 -m venv .venv || print_error "Python 3.12 not found. Please install it first."

# === Activate venv ===
print_blue "Activating virtual environment..."
source .venv/Scripts/activate || print_error "Failed to activate venv. Are you sure you're in Git Bash?"

# === Show Python and pip versions ===
print_blue "Python version:"
python --version

print_blue "pip version:"
pip --version

# === Upgrade pip only if needed ===
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

# === Create timestamp file ===
echo "$(date +%s)" > .venv/.install_timestamp

# === Register venv in Jupyter ===
print_blue "Registering virtual environment in Jupyter..."
python -m ipykernel install --user --name=venv --display-name "Python (.venv)"

# === Configure VS Code interpreter ===
print_blue "Writing VS Code settings..."
mkdir -p .vscode
echo '{
  "python.defaultInterpreterPath": "${workspaceFolder}/.venv/Scripts/python.exe"
}' > .vscode/settings.json

# === Optional: XGBoost GPU Support ===
print_blue "Installing GPU-enabled XGBoost (optional)..."
pip uninstall -y xgboost
pip install --upgrade --extra-index-url https://pypi.nvidia.com xgboost

pip install "uvicorn[standard]==0.29.0"

# === Done ===
print_green "Setup complete! Your virtual environment is ready!"
print_yellow "💡 Next time, this script will be much faster as it will only update what's needed."
