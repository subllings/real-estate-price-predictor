#!/bin/bash

# cd e:/_SoftEng/_BeCode/real-estate-price-predictor
# chmod +x setup-conda-env-project.sh
# ./setup-conda-env-project.sh

ENV_PATH="./conda-env"  # Local environment path at project root
PYTHON_VERSION="3.11"

echo -e "\033[34m>>> Setting up ROOT conda environment at: $ENV_PATH\033[0m"
echo -e "\033[33m🏠 This is the main project environment for data science, ML training, and analysis\033[0m"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "\033[31m❌ Conda is not installed or not in PATH\033[0m"
    exit 1
fi

# Remove existing environment folder if it exists
if [ -d "${ENV_PATH}" ]; then
    echo -e "\033[33m>>> Environment path '$ENV_PATH' already exists. Removing it first...\033[0m"
    rm -rf "${ENV_PATH}"
fi

# Remove .venv if it exists
if [ -d ".venv" ]; then
    echo -e "\033[33m>>> Removing old .venv directory...\033[0m"
    rm -rf ".venv"
fi

# Create the environment locally using the -p option
echo -e "\033[33m>>> Creating local conda environment with Python $PYTHON_VERSION...\033[0m"
conda create -p "${ENV_PATH}" python="$PYTHON_VERSION" -y

# Activate the environment
echo -e "\033[33m>>> Activating environment: $ENV_PATH...\033[0m"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PATH}"

# Install core dependencies
echo -e "\033[33m>>> Installing core dependencies...\033[0m"
python -m pip install --upgrade pip

# Data Science & ML
echo -e "\033[33m>>> Installing data science and ML libraries...\033[0m"
python -m pip install pandas==2.1.3
python -m pip install numpy==1.24.3
python -m pip install scikit-learn==1.3.2
python -m pip install matplotlib==3.8.2
python -m pip install seaborn==0.13.0
python -m pip install plotly==5.17.0

# Machine Learning models
echo -e "\033[33m>>> Installing ML model libraries...\033[0m"
python -m pip install catboost==1.2.2
python -m pip install xgboost==2.0.2
python -m pip install lightgbm==4.1.0
python -m pip install optuna==3.4.0

# Jupyter and analysis
echo -e "\033[33m>>> Installing Jupyter and analysis tools...\033[0m"
python -m pip install jupyter==1.0.0
python -m pip install jupyterlab==4.0.8
python -m pip install ipykernel==6.26.0

# Data processing
echo -e "\033[33m>>> Installing data processing libraries...\033[0m"
python -m pip install requests==2.31.0
python -m pip install beautifulsoup4==4.12.2
python -m pip install lxml==4.9.3

# Utilities
echo -e "\033[33m>>> Installing utility libraries...\033[0m"
python -m pip install python-dotenv==1.0.0
python -m pip install tqdm==4.66.1
python -m pip install joblib==1.3.2

# Development tools
echo -e "\033[33m>>> Installing development tools...\033[0m"
python -m pip install pytest==7.4.3
python -m pip install black==23.11.0
python -m pip install flake8==6.1.0

# Verify installation
echo -e "\033[33m>>> Verifying installation...\033[0m"
python -c "
try:
    import pandas, numpy, sklearn, catboost, xgboost, lightgbm, optuna, jupyter
    print('✅ All core dependencies installed successfully')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    exit(1)
"

# Create requirements.txt for future reference
echo -e "\033[33m>>> Creating requirements.txt...\033[0m"
python -m pip freeze > requirements.txt

# Setup Jupyter kernel
echo -e "\033[33m>>> Setting up Jupyter kernel...\033[0m"
python -m ipykernel install --user --name=real-estate-ml --display-name="Real Estate ML"

echo -e "\033[32m✅ Root project environment '$ENV_PATH' created successfully!\033[0m"
echo ""
echo -e "\033[34m📋 Available commands:\033[0m"
echo -e "\033[32m  • Activate environment: conda activate $ENV_PATH\033[0m"
echo -e "\033[32m  • Start Jupyter Lab: jupyter lab\033[0m"
echo -e "\033[32m  • Run ML training: python train_models.py\033[0m"
echo -e "\033[32m  • Data analysis: python data_analysis.py\033[0m"
echo ""
echo -e "\033[33m💡 Environment info:\033[0m"
echo -e "\033[33m   Purpose: Data Science, ML Training, Analysis\033[0m"
echo -e "\033[33m   Path: $ENV_PATH\033[0m"
echo -e "\033[33m   Python: $PYTHON_VERSION\033[0m"
echo -e "\033[33m   Location: $(realpath $ENV_PATH)\033[0m"
echo ""
echo -e "\033[34m🚀 API-specific environments:\033[0m"
echo -e "\033[33m   • LLM API: app/backend-api-llm-v2/conda-env\033[0m"
echo -e "\033[33m   • Price Prediction API: app/backend-api-price-prediction/conda-env\033[0m"
