#!/bin/bash

# cd e:/_SoftEng/_BeCode/real-estate-price-predictor
# chmod +x setup-conda-env-project.sh
# ./setup-conda-env-project.sh

ENV_NAME="real-estate-ml-project"
PYTHON_VERSION="3.11"

echo "🚀 Setting up main ML project environment: $ENV_NAME"

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Remove old environment
conda env remove -n "${ENV_NAME}" -y 2>/dev/null || true

# Create new environment
conda create -n "${ENV_NAME}" python="$PYTHON_VERSION" -y

# Activate environment
conda activate "${ENV_NAME}"

# Install packages
conda install -c conda-forge pandas numpy scikit-learn matplotlib seaborn plotly jupyter jupyterlab -y
pip install catboost xgboost lightgbm optuna python-dotenv tqdm pytest black flake8

# Test
python -c "import pandas, sklearn, jupyter; print('✅ Setup OK')"

echo "✅ Done! Activate with: conda activate ${ENV_NAME}"
echo "✅ Start Jupyter: jupyter lab"
