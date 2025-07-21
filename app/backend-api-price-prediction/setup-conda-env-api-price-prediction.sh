#!/bin/bash

# cd e:/_SoftEng/_BeCode/real-estate-price-predictor/app/backend-api-price-prediction
# chmod +x setup-conda-env-api-price-prediction.sh
# ./setup-conda-env-api-price-prediction.sh

ENV_PATH="./conda-env"
PYTHON_VERSION="3.11"

echo -e "\033[34m>>> Setting up Price Prediction API conda environment at: $ENV_PATH\033[0m"
echo -e "\033[33m🔧 This API will use its own isolated price prediction environment\033[0m"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "\033[31m❌ Conda is not installed or not in PATH\033[0m"
    exit 1
fi

# Force cleanup - always remove existing environment
echo -e "\033[33m>>> Force cleanup: removing any existing environment...\033[0m"
rm -rf "${ENV_PATH}"

# Create the environment locally using the -p option
echo -e "\033[33m>>> Creating local conda environment with Python $PYTHON_VERSION...\033[0m"
conda create -p "${ENV_PATH}" python="$PYTHON_VERSION" -y

# Activate the environment (ignore errors)
echo -e "\033[33m>>> Activating environment: $ENV_PATH...\033[0m"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "${ENV_PATH}" 2>/dev/null || true

# Install everything in one go to avoid activation issues
echo -e "\033[33m>>> Installing all dependencies...\033[0m"
conda install -c conda-forge fastapi uvicorn python-multipart pandas numpy scikit-learn xgboost lightgbm joblib requests python-dotenv pydantic -y

# Install CatBoost via pip
echo -e "\033[33m>>> Installing CatBoost via pip...\033[0m"
"${ENV_PATH}/python" -m pip install catboost

# Verify installation
echo -e "\033[33m>>> Verifying installation...\033[0m"
"${ENV_PATH}/python" -c "
try:
    import fastapi, uvicorn, pandas, numpy, sklearn, catboost, xgboost, lightgbm, joblib
    print('✅ All price prediction dependencies installed successfully')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    exit(1)
"

echo -e "\033[32m✅ Price Prediction API environment '$ENV_PATH' created successfully!\033[0m"
echo ""
echo -e "\033[34m📋 Next steps:\033[0m"
echo -e "\033[32m  1. Run the API: ./start-api-price-prediction.sh\033[0m"
echo ""
echo -e "\033[33m💡 Environment info:\033[0m"
echo -e "\033[33m   Path: $ENV_PATH\033[0m"
echo -e "\033[33m   Python: $PYTHON_VERSION\033[0m"
echo -e "\033[33m   Purpose: Real Estate Price Prediction API\033[0m"
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    exit(1)
"

# Create requirements.txt for future reference
echo -e "\033[33m>>> Creating requirements.txt...\033[0m"
python -m pip freeze > requirements.txt

echo -e "\033[32m✅ Price Prediction API environment '$ENV_PATH' created successfully!\033[0m"
echo ""
echo -e "\033[34m📋 Next steps:\033[0m"
echo -e "\033[32m  1. Activate environment: conda activate $ENV_PATH\033[0m"
echo -e "\033[32m  2. Configure .env file with API settings\033[0m"
echo -e "\033[32m  3. Run the API: ./start-api-price-prediction.sh\033[0m"
echo ""
echo -e "\033[33m💡 Environment info:\033[0m"
echo -e "\033[33m   Path: $ENV_PATH\033[0m"
echo -e "\033[33m   Python: $PYTHON_VERSION\033[0m"
echo -e "\033[33m   Purpose: Real Estate Price Prediction API\033[0m"
echo -e "\033[33m   Location: $(realpath $ENV_PATH 2>/dev/null || echo $ENV_PATH)\033[0m"
echo -e "\033[33m   Location: $(realpath $ENV_PATH 2>/dev/null || echo $ENV_PATH)\033[0m"
echo -e "\033[33m   Location: $(realpath $ENV_PATH 2>/dev/null || echo $ENV_PATH)\033[0m" conda activate $ENV_PATH\033[0m"
echo -e "\033[33m   Purpose: Real Estate Price Prediction API\033[0m"settings\033[0m"
echo -e "\033[33m   Location: $(realpath $ENV_PATH 2>/dev/null || echo $ENV_PATH)\033[0m"3[0m"
echo ""
echo -e "\033[33m💡 Environment info:\033[0m"
echo -e "\033[33m   Path: $ENV_PATH\033[0m"
echo -e "\033[33m   Python: $PYTHON_VERSION\033[0m"
echo -e "\033[33m   Purpose: Real Estate Price Prediction API\033[0m"
echo -e "\033[33m   Location: $(realpath $ENV_PATH 2>/dev/null || echo $ENV_PATH)\033[0m"
echo -e "\033[33m   Path: $ENV_PATH\033[0m"
echo -e "\033[33m   Python: $PYTHON_VERSION\033[0m"
echo -e "\033[33m   Purpose: Real Estate Price Prediction API\033[0m"
echo -e "\033[33m   Location: $(realpath $ENV_PATH 2>/dev/null || echo $ENV_PATH)\033[0m"
echo -e "\033[33m   Purpose: Real Estate Price Prediction API\033[0m"
echo -e "\033[33m   Location: $(realpath $ENV_PATH 2>/dev/null || echo $ENV_PATH)\033[0m"
echo -e "\033[33m   Purpose: Real Estate Price Prediction API\033[0m"
echo -e "\033[33m   Location: $(realpath $ENV_PATH 2>/dev/null || echo $ENV_PATH)\033[0m"
echo -e "\033[33m   Location: $(realpath $ENV_PATH 2>/dev/null || echo $ENV_PATH)\033[0m"
echo -e "\033[33m   Purpose: Real Estate Price Prediction API\033[0m"
echo -e "\033[33m   Location: $(realpath $ENV_PATH 2>/dev/null || echo $ENV_PATH)\033[0m"
