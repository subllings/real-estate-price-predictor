#!/bin/bash

# cd e:/_SoftEng/_BeCode/real-estate-price-predictor
# chmod +x setup-env.sh
# ./setup-env.sh

clear

# === Navigate to project root ===
cd "e:\_SoftEng\_BeCode\real-estate-price-predictor" || exit 1

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

# === Remove existing venv AND conda-env ===
if [ -d ".venv" ]; then
    print_blue "Removing existing virtual environment (.venv)..."
    rm -rf .venv
fi

if [ -d "conda-env" ]; then
    print_blue "Removing existing conda-env directory..."
    rm -rf conda-env
fi

# === Use conda base environment instead ===
print_blue "Using conda base environment for consistency..."

# === Activate conda base ===
print_blue "Activating conda base environment..."
eval "$(conda shell.bash hook)"
conda activate base

# === Show Python and pip versions ===
print_blue "Python version:"
python --version

print_blue "pip version:"
pip --version

# === Install dependencies in conda base ===
print_blue "Installing dependencies in conda base environment..."
conda install -c conda-forge pandas numpy scikit-learn matplotlib seaborn plotly jupyter jupyterlab -y
pip install catboost xgboost lightgbm optuna azure-cosmos azure-core azure-identity --no-warn-script-location

# === Register conda base in Jupyter ===
print_blue "Registering conda base environment in Jupyter..."
python -m ipykernel install --user --name=conda-base --display-name "Python (conda-base)"

# === Configure VS Code interpreter ===
print_blue "Writing VS Code settings for conda..."
mkdir -p .vscode
echo '{
  "python.defaultInterpreterPath": "python"
}' > .vscode/settings.json

# === Optional: XGBoost GPU Support (NVIDIA) ===
print_blue "Installing GPU-enabled XGBoost (optional)..."
pip uninstall -y xgboost
pip install --upgrade --extra-index-url https://pypi.nvidia.com xgboost --no-warn-script-location

pip install "uvicorn[standard]==0.29.0" --no-warn-script-location

# === Done ===
print_green "Setup complete. Using unified conda base environment!"
# === Optional: XGBoost GPU Support (NVIDIA) ===
print_blue "Installing GPU-enabled XGBoost (optional)..."
pip uninstall -y xgboost
pip install --upgrade --extra-index-url https://pypi.nvidia.com xgboost

pip install "uvicorn[standard]==0.29.0"

# === Done ===
print_green "Setup complete. Your virtual environment is ready!"
# === Done ===
print_green "Setup complete. Your virtual environment is ready!"
