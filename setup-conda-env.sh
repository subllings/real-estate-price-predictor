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

# === Ensure Conda is in PATH (for Git Bash on Windows) ===
if ! command -v conda &> /dev/null; then
    CONDA_WIN_PATH="/c/ProgramData/anaconda3"
    echo ">>> Conda not found in PATH — adding it temporarily..."
    export PATH="$CONDA_WIN_PATH:$CONDA_WIN_PATH/Scripts:$CONDA_WIN_PATH/condabin:$PATH"

    if ! command -v conda &> /dev/null; then
        print_error "conda not found. Please install Anaconda or Miniconda and try again."
    fi

    if ! grep -q "$CONDA_WIN_PATH" ~/.bashrc; then
        echo ">>> Adding Conda to PATH permanently in ~/.bashrc"
        echo "" >> ~/.bashrc
        echo "# Added by setup-conda-env.sh" >> ~/.bashrc
        echo "export PATH=\"$CONDA_WIN_PATH:$CONDA_WIN_PATH/Scripts:$CONDA_WIN_PATH/condabin:\$PATH\"" >> ~/.bashrc
    fi
fi

# === Define local environment path inside project ===
ENV_PATH="./environment/conda-env"

print_blue "Removing existing conda environment at $ENV_PATH if it exists..."
conda env remove -p "$ENV_PATH" -y >/dev/null 2>&1

if [ -d "$ENV_PATH" ]; then
    print_blue "Conda environment directory still exists, removing manually..."
    rm -rf "$ENV_PATH"
fi

print_blue "Creating new conda environment at $ENV_PATH with Python 3.12..."
conda create -p "$ENV_PATH" python=3.12 -y || print_error "Failed to create conda environment."

print_blue "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate "$ENV_PATH" || print_error "Failed to activate conda environment."

print_blue "Installing faiss-cpu via conda (Windows)..."
conda install -y -c conda-forge faiss-cpu || print_error "Failed to install faiss-cpu via conda"

print_blue "Installing Python dependencies from requirements.txt..."
pip install -r requirements.txt || print_error "Failed to install dependencies"

print_blue "Installing uvicorn..."
pip install "uvicorn[standard]==0.29.0" || print_error "Failed to install uvicorn"

print_blue "Registering kernel for Jupyter..."
python -m ipykernel install --user --name="conda-env" --display-name "Python (conda-env)"

print_blue "Writing VS Code settings..."
mkdir -p .vscode

echo "{\"python.defaultInterpreterPath\": \"$PWD/environment/conda-env/Scripts/python.exe\"}" > .vscode/settings.json


print_green "Setup complete. Your Conda environment at $ENV_PATH is ready!"
